"""
Обучение модели 144M параметров со сложными промптами.

Архитектура:
  - UNet: 136.6M (base_ch=224, ch_mult=1,2,3,4 → каналы [224,448,672,896])
  - SharpVAE: 6.2M (base_ch=128, ÷4 compression, skip connections)
  - TextEncoder: 1.0M (vocab=5000, embed=128, context=512, 2 layers)
  - ИТОГО: ~144M параметров

Сложный датасет:
  - Несколько объектов на одном холсте
  - Градиенты, наложения, тени
  - Текстуры (шахматная доска, полосы, точки)
  - Композиции "объект А на фоне с паттерном Б"

VRAM: ~2.7 ГБ (batch=2) — влезает в RTX 2050 (4 ГБ)

Использование::

    # Обучить (стандарт: VAE 1000 + UNet 2000 шагов, ~8-10 мин)
    python -m train.train_own_v3

    # Больше шагов
    python -m train.train_own_v3 --steps 5000 --steps-vae 2000

    # Загрузить и сгенерировать
    python -m train.train_own_v3 --load train/own_model_v3.pt --prompt "red circle and blue square on gradient background"
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from train.train_mini import MiniUNet, MiniTextEncoder, SimpleScheduler
from train.train_own import DDIMSampler, _save_bmp
from train.train_own_v2 import (
    SharpVAE, PatchDiscriminator,
    sobel_edges, edge_loss, _flush, _vram_mb,
)


# ================================================================== #
#  Complex Synthetic Dataset — multi-object scenes
# ================================================================== #

PALETTE = {
    "red": (220, 50, 50), "green": (50, 200, 50), "blue": (50, 50, 220),
    "yellow": (230, 220, 50), "purple": (150, 50, 200), "orange": (240, 150, 30),
    "cyan": (50, 210, 210), "white": (240, 240, 240), "pink": (240, 130, 170),
    "brown": (140, 80, 30), "lime": (100, 240, 60), "magenta": (200, 50, 180),
    "teal": (50, 160, 160), "gold": (230, 190, 50), "silver": (180, 180, 190),
    "coral": (240, 100, 80), "navy": (30, 30, 120), "maroon": (128, 20, 20),
}

BG_STYLES = {
    "solid_black": "black background",
    "solid_white": "white background",
    "solid_gray": "gray background",
    "solid_dark_blue": "dark blue background",
    "gradient_h": "horizontal gradient background",
    "gradient_v": "vertical gradient background",
    "checkerboard": "checkerboard background",
    "stripes_h": "horizontal striped background",
    "stripes_v": "vertical striped background",
    "dots": "dotted background",
    "noise": "noisy background",
}

SHAPE_NAMES = [
    "circle", "square", "triangle", "diamond", "cross",
    "ring", "star", "hexagon", "arrow_up", "arrow_right",
    "half_circle", "crescent",
]

POSITIONS = {
    "center": (0.5, 0.5),
    "top_left": (0.3, 0.3),
    "top_right": (0.7, 0.3),
    "bottom_left": (0.3, 0.7),
    "bottom_right": (0.7, 0.7),
    "top": (0.5, 0.3),
    "bottom": (0.5, 0.7),
    "left": (0.3, 0.5),
    "right": (0.7, 0.5),
}

SIZES = {"small": 0.15, "medium": 0.25, "large": 0.35}


def make_background(size: int, style: str, color1: tuple, color2: tuple) -> np.ndarray:
    """Create a complex background."""
    img = np.zeros((size, size, 3), dtype=np.uint8)

    if style == "solid_black":
        img[:] = (10, 10, 10)
    elif style == "solid_white":
        img[:] = (240, 240, 240)
    elif style == "solid_gray":
        img[:] = (128, 128, 128)
    elif style == "solid_dark_blue":
        img[:] = (20, 20, 80)
    elif style == "gradient_h":
        for x in range(size):
            f = x / size
            img[:, x] = tuple(int(color1[i] * (1-f) + color2[i] * f) for i in range(3))
    elif style == "gradient_v":
        for y in range(size):
            f = y / size
            img[y, :] = tuple(int(color1[i] * (1-f) + color2[i] * f) for i in range(3))
    elif style == "checkerboard":
        sq = max(size // 8, 4)
        for y in range(size):
            for x in range(size):
                if ((x // sq) + (y // sq)) % 2 == 0:
                    img[y, x] = color1
                else:
                    img[y, x] = color2
    elif style == "stripes_h":
        stripe_w = max(size // 10, 3)
        for y in range(size):
            img[y, :] = color1 if (y // stripe_w) % 2 == 0 else color2
    elif style == "stripes_v":
        stripe_w = max(size // 10, 3)
        for x in range(size):
            img[:, x] = color1 if (x // stripe_w) % 2 == 0 else color2
    elif style == "dots":
        img[:] = color1
        dot_spacing = max(size // 8, 4)
        dot_r = max(dot_spacing // 4, 1)
        yy, xx = np.ogrid[:size, :size]
        for cy in range(dot_spacing // 2, size, dot_spacing):
            for cx in range(dot_spacing // 2, size, dot_spacing):
                mask = ((xx - cx)**2 + (yy - cy)**2) <= dot_r**2
                img[mask] = color2
    elif style == "noise":
        base = np.array(color1, dtype=np.float32)
        noise = np.random.randint(-30, 31, (size, size, 3)).astype(np.float32)
        img = np.clip(base + noise, 0, 255).astype(np.uint8)

    return img


def draw_shape_at(img: np.ndarray, shape: str, color: tuple,
                  cx: int, cy: int, r: int):
    """Draw a shape at a specific position."""
    size = img.shape[0]

    if shape == "circle":
        yy, xx = np.ogrid[:size, :size]
        mask = ((xx - cx)**2 + (yy - cy)**2) <= r**2
        img[mask] = color

    elif shape == "square":
        y1, y2 = max(0, cy-r), min(size, cy+r)
        x1, x2 = max(0, cx-r), min(size, cx+r)
        img[y1:y2, x1:x2] = color

    elif shape == "triangle":
        for y in range(max(0, cy-r), min(size, cy+r)):
            frac = (y - (cy-r)) / (2*r + 1e-8)
            hw = int(frac * r)
            x1, x2 = max(0, cx-hw), min(size, cx+hw+1)
            img[y, x1:x2] = color

    elif shape == "diamond":
        for y in range(max(0, cy-r), min(size, cy+r)):
            d = abs(y - cy)
            hw = max(0, r - d)
            x1, x2 = max(0, cx-hw), min(size, cx+hw+1)
            img[y, x1:x2] = color

    elif shape == "cross":
        t = max(r // 3, 2)
        y1, y2 = max(0, cy-r), min(size, cy+r)
        x1, x2 = max(0, cx-t), min(size, cx+t)
        img[y1:y2, x1:x2] = color
        y1, y2 = max(0, cy-t), min(size, cy+t)
        x1, x2 = max(0, cx-r), min(size, cx+r)
        img[y1:y2, x1:x2] = color

    elif shape == "ring":
        yy, xx = np.ogrid[:size, :size]
        dist2 = (xx - cx)**2 + (yy - cy)**2
        inner = r * 0.6
        mask = (dist2 <= r**2) & (dist2 >= inner**2)
        img[mask] = color

    elif shape == "star":
        # 5-pointed star via overlapping triangles
        for angle_offset in [0, 72, 144, 216, 288]:
            rad = math.radians(angle_offset - 90)
            px = cx + int(r * math.cos(rad))
            py = cy + int(r * math.sin(rad))
            sr = r // 2
            for y in range(max(0, py-sr), min(size, py+sr)):
                frac = 1.0 - abs(y - py) / (sr + 1e-8)
                hw = int(frac * sr * 0.5)
                x1, x2 = max(0, px-hw), min(size, px+hw+1)
                img[y, x1:x2] = color
        # Center fill
        yy, xx = np.ogrid[:size, :size]
        mask = ((xx-cx)**2 + (yy-cy)**2) <= (r//2)**2
        img[mask] = color

    elif shape == "hexagon":
        for y in range(max(0, cy-r), min(size, cy+r)):
            dy = abs(y - cy) / (r + 1e-8)
            if dy < 0.5:
                hw = r
            else:
                hw = int(r * (1.0 - (dy - 0.5) * 2))
            x1, x2 = max(0, cx-hw), min(size, cx+hw+1)
            img[y, x1:x2] = color

    elif shape == "arrow_up":
        # Triangle top + rect bottom
        head_h = r
        for y in range(max(0, cy-r), min(size, cy)):
            frac = (y - (cy-r)) / (head_h + 1e-8)
            hw = int(frac * r * 0.8)
            x1, x2 = max(0, cx-hw), min(size, cx+hw+1)
            img[y, x1:x2] = color
        t = max(r // 3, 2)
        img[max(0,cy):min(size,cy+r), max(0,cx-t):min(size,cx+t)] = color

    elif shape == "arrow_right":
        head_w = r
        for x in range(max(0, cx), min(size, cx+r)):
            frac = 1.0 - (x - cx) / (head_w + 1e-8)
            hw = int(frac * r * 0.8)
            y1, y2 = max(0, cy-hw), min(size, cy+hw+1)
            img[y1:y2, x] = color
        t = max(r // 3, 2)
        img[max(0,cy-t):min(size,cy+t), max(0,cx-r):min(size,cx)] = color

    elif shape == "half_circle":
        yy, xx = np.ogrid[:size, :size]
        mask = ((xx-cx)**2 + (yy-cy)**2 <= r**2) & (yy <= cy)
        img[mask] = color

    elif shape == "crescent":
        yy, xx = np.ogrid[:size, :size]
        outer = (xx-cx)**2 + (yy-cy)**2 <= r**2
        inner = (xx-(cx+r//3))**2 + (yy-cy)**2 <= (r*0.8)**2
        mask = outer & ~inner
        img[mask] = color


class ComplexDataset(Dataset):
    """Complex synthetic dataset with multi-object scenes."""

    def __init__(self, num_samples: int = 1000, img_size: int = 64,
                 text_encoder: MiniTextEncoder | None = None):
        self.num_samples = num_samples
        self.img_size = img_size
        self.text_encoder = text_encoder
        self.samples = []

        color_names = list(PALETTE.keys())
        bg_styles = list(BG_STYLES.keys())
        pos_names = list(POSITIONS.keys())
        size_names = list(SIZES.keys())

        rng = random.Random(42)

        for i in range(num_samples):
            # Pick background
            bg_style = bg_styles[i % len(bg_styles)]
            bg_c1 = PALETTE[color_names[rng.randint(0, len(color_names)-1)]]
            bg_c2 = PALETTE[color_names[rng.randint(0, len(color_names)-1)]]
            bg_desc = BG_STYLES[bg_style]

            img = make_background(img_size, bg_style, bg_c1, bg_c2)

            # 1-3 objects
            num_obj = rng.choice([1, 1, 2, 2, 2, 3])
            obj_descs = []

            used_positions = set()
            for _ in range(num_obj):
                shape = rng.choice(SHAPE_NAMES)
                color_name = rng.choice(color_names)
                color = PALETTE[color_name]
                sz_name = rng.choice(size_names)
                r = int(SIZES[sz_name] * img_size)

                # Pick unused position
                avail_pos = [p for p in pos_names if p not in used_positions]
                if not avail_pos:
                    avail_pos = pos_names
                pos_name = rng.choice(avail_pos)
                used_positions.add(pos_name)
                fx, fy = POSITIONS[pos_name]

                cx = int(fx * img_size)
                cy = int(fy * img_size)

                draw_shape_at(img, shape, color, cx, cy, r)

                pos_text = pos_name.replace("_", " ")
                obj_descs.append(f"{sz_name} {color_name} {shape} at {pos_text}")

            prompt = ", ".join(obj_descs) + f" on {bg_desc}"

            # To tensor
            img_t = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
            tokens = text_encoder.simple_tokenize(prompt) if text_encoder else torch.zeros(32, dtype=torch.long)

            self.samples.append({
                "image": img_t,
                "tokens": tokens,
                "prompt": prompt,
            })

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    return {
        "images": torch.stack([b["image"] for b in batch]),
        "tokens": torch.stack([b["tokens"] for b in batch]),
        "prompts": [b["prompt"] for b in batch],
    }


# ================================================================== #
#  Training
# ================================================================== #

def train_vae(vae: SharpVAE, disc: PatchDiscriminator,
              dataset, device, steps: int = 1000,
              lr: float = 1e-3, batch_size: int = 4):
    """Train VAE with L1 + edge + progressive GAN."""
    print(f"\n{'='*60}")
    print(f"  Phase 1: Training SharpVAE (L1 + Edge + GAN)")
    print(f"{'='*60}")

    vae.train()
    disc.train()

    opt_vae = torch.optim.Adam(vae.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_disc = torch.optim.Adam(disc.parameters(), lr=lr * 0.2, betas=(0.5, 0.999))

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        collate_fn=collate_fn, drop_last=True)

    use_amp = torch.cuda.is_available()
    scaler_vae = torch.amp.GradScaler("cuda", enabled=use_amp)
    scaler_disc = torch.amp.GradScaler("cuda", enabled=use_amp)

    gan_start = max(steps // 3, 100)
    step = 0
    t0 = time.time()

    print(f"  {'step':>6} | {'L1':>8} | {'edge':>8} | {'GAN':>8} | "
          f"{'D_loss':>8} | {'VRAM':>8}")
    print(f"  {'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")

    d_loss = torch.tensor(0.0)

    while step < steps:
        for batch in loader:
            if step >= steps:
                break

            images = batch["images"].to(device)

            # ---- Discriminator (every 2 steps after warmup) ----
            if step >= gan_start and step % 2 == 0:
                opt_disc.zero_grad()
                with torch.amp.autocast("cuda", enabled=use_amp):
                    with torch.no_grad():
                        drop = random.random() < 0.2
                        recon_d, _, _ = vae(images, drop_skips=drop)
                    real_pred = disc(images)
                    fake_pred = disc(recon_d.detach())
                    d_loss = 0.5 * (
                        F.binary_cross_entropy_with_logits(
                            real_pred, 0.9 * torch.ones_like(real_pred)) +
                        F.binary_cross_entropy_with_logits(
                            fake_pred, 0.1 * torch.ones_like(fake_pred))
                    )
                scaler_disc.scale(d_loss).backward()
                scaler_disc.step(opt_disc)
                scaler_disc.update()

            # ---- VAE ----
            opt_vae.zero_grad()
            drop = random.random() < 0.15
            with torch.amp.autocast("cuda", enabled=use_amp):
                recon, mu, logvar = vae(images, drop_skips=drop)
                l1 = F.l1_loss(recon, images)
                e_loss = edge_loss(recon, images)
                kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                vae_loss = l1 + 0.5 * e_loss + 0.0005 * kl

                if step >= gan_start:
                    progress = min(1.0, (step - gan_start) / max(1, steps - gan_start))
                    gw = 0.01 + 0.09 * progress
                    g_loss = F.binary_cross_entropy_with_logits(
                        disc(recon), torch.ones_like(disc(recon)))
                    vae_loss = vae_loss + gw * g_loss
                else:
                    g_loss = torch.tensor(0.0)

            scaler_vae.scale(vae_loss).backward()
            scaler_vae.step(opt_vae)
            scaler_vae.update()
            step += 1

            if step % 100 == 0 or step == 1:
                print(f"  {step:6d} | {l1.item():8.4f} | {e_loss.item():8.4f} | "
                      f"{g_loss.item():8.4f} | {d_loss.item():8.4f} | "
                      f"{_vram_mb():8.0f}MB")

    elapsed = time.time() - t0
    print(f"\n  SharpVAE trained: {steps} steps in {elapsed:.1f}s")
    vae.eval()
    return vae


def train_unet(unet, vae: SharpVAE, text_enc, dataset, device,
               steps: int = 2000, lr: float = 2e-4,
               batch_size: int = 2):
    """Train UNet to denoise 16×16 latents."""
    print(f"\n{'='*60}")
    print(f"  Phase 2: Training UNet 136.6M (denoising)")
    print(f"{'='*60}")

    scheduler = SimpleScheduler(num_timesteps=1000)
    unet.train()
    vae.eval()
    text_enc.eval()

    optimizer = torch.optim.AdamW(unet.parameters(), lr=lr, weight_decay=0.01)
    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps, eta_min=lr * 0.01)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        collate_fn=collate_fn, drop_last=True)

    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    step = 0
    losses = []
    t0 = time.time()

    print(f"  {'step':>6} | {'loss':>10} | {'avg':>10} | {'VRAM':>8}")
    print(f"  {'-'*6}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}")

    while step < steps:
        for batch in loader:
            if step >= steps:
                break

            images = batch["images"].to(device)
            tokens = batch["tokens"].to(device)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=use_amp):
                with torch.no_grad():
                    z, _, _, _, _ = vae.encode(images)

                t = torch.randint(0, 1000, (z.shape[0],), device=device)
                noise = torch.randn_like(z)
                noisy_z = scheduler.add_noise(z, noise, t)

                with torch.no_grad():
                    context = text_enc(tokens)

                pred = unet(noisy_z, t, context)
                loss = F.l1_loss(pred, noise)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            lr_sched.step()

            step += 1
            losses.append(loss.item())

            if step % 100 == 0 or step == 1:
                avg = sum(losses[-100:]) / min(len(losses), 100)
                print(f"  {step:6d} | {loss.item():10.6f} | "
                      f"{avg:10.6f} | {_vram_mb():8.0f}MB")

    elapsed = time.time() - t0
    initial = sum(losses[:30]) / min(len(losses), 30)
    final = sum(losses[-30:]) / min(len(losses), 30)
    print(f"\n  UNet trained: {steps} steps in {elapsed:.1f}s")
    print(f"  Loss: {initial:.4f} → {final:.4f} ({(1-final/initial)*100:+.1f}%)")

    return unet, scheduler


@torch.no_grad()
def generate(unet, vae, text_enc, scheduler, prompts, device,
             latent_shape, num_steps=50,
             output_dir="output/own_model_v3"):
    """Generate images."""
    print(f"\n{'='*60}")
    print(f"  Generating images (144M model)")
    print(f"{'='*60}")

    os.makedirs(output_dir, exist_ok=True)
    unet.eval(); vae.eval(); text_enc.eval()

    sampler = DDIMSampler(scheduler)

    for i, prompt in enumerate(prompts):
        print(f"\n  [{i+1}/{len(prompts)}] \"{prompt}\"")
        tokens = text_enc.simple_tokenize(prompt).unsqueeze(0).to(device)
        context = text_enc(tokens)

        shape = (1, latent_shape[1], latent_shape[2], latent_shape[3])
        z = sampler.sample(unet, shape, context, device, num_steps=num_steps)

        img = vae.decode(z, s1=None, s2=None)
        img = img[0].clamp(0, 1)
        img_np = (img.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        path = os.path.join(output_dir, f"gen_{i:03d}.bmp")
        _save_bmp(path, img_np)
        print(f"  Saved: {path} ({img_np.shape[1]}×{img_np.shape[0]})")

    print(f"\n  All images saved to: {os.path.abspath(output_dir)}")
    return output_dir


# ================================================================== #
#  Default complex prompts for generation
# ================================================================== #

COMPLEX_PROMPTS = [
    # Multi-object compositions
    "large red circle at center, small blue square at top right on checkerboard background",
    "medium green triangle at top, medium orange diamond at bottom on horizontal gradient background",
    "small cyan star at top left, small magenta ring at bottom right, large yellow hexagon at center on dark blue background",
    "large purple cross at center, small white circle at top left on vertical striped background",

    # Interesting patterns
    "medium gold crescent at center on dotted background",
    "small red arrow_up at top, small blue arrow_right at right on noisy background",
    "large teal half_circle at center, small coral square at top right on horizontal gradient background",
    "medium maroon hexagon at left, medium lime star at right on gray background",

    # Complex multi-object
    "small red circle at top left, small green circle at top right, small blue circle at bottom left, large yellow diamond at center on black background",
    "large silver ring at center, small navy square at top on vertical gradient background",
    "medium pink triangle at center, small brown cross at bottom right on checkerboard background",
    "large orange star at center, small purple ring at top left, small cyan diamond at bottom right on white background",
]


# ================================================================== #
#  Full pipeline
# ================================================================== #

def run_pipeline(
    steps_vae=1000, steps_unet=2000,
    batch_size=2, img_size=64, num_samples=1000,
    save_path="train/own_model_v3.pt",
    output_dir="output/own_model_v3",
    generate_prompts=None,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*60}")
    print(f"  Training 144M Parameter Model (v3)")
    print(f"{'='*60}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**2:.0f} MB")
    print(f"  Image: {img_size}×{img_size}  |  Latent: {img_size//4}×{img_size//4}")
    print(f"  Dataset: {num_samples} complex scenes")
    print(f"  VAE: {steps_vae} steps  |  UNet: {steps_unet} steps")
    print(f"  Batch: {batch_size}")
    print(f"{'='*60}\n")

    t_total = time.time()

    # Text encoder (1.0M)
    context_dim = 512
    text_enc = MiniTextEncoder(
        vocab_size=5000, embed_dim=128,
        context_dim=context_dim, max_len=32, num_layers=2,
    ).to(device)
    text_enc.eval()
    text_enc.requires_grad_(False)

    # Dataset
    print(f"  Creating complex dataset ({num_samples} scenes) ...")
    dataset = ComplexDataset(
        num_samples=num_samples,
        img_size=img_size,
        text_encoder=text_enc,
    )
    print(f"  Example prompts:")
    for s in dataset.samples[:5]:
        print(f"    - {s['prompt']}")
    print()

    # VAE (6.2M) + Discriminator
    latent_ch = 4
    vae = SharpVAE(in_channels=3, latent_channels=latent_ch, base_ch=128).to(device)
    disc = PatchDiscriminator(in_channels=3, base_ch=48).to(device)

    vae = train_vae(vae, disc, dataset, device,
                    steps=steps_vae, lr=1e-3, batch_size=min(batch_size * 2, 8))

    del disc
    _flush()

    # UNet (136.6M)
    latent_size = img_size // 4
    unet = MiniUNet(
        in_channels=latent_ch, out_channels=latent_ch,
        context_dim=context_dim,
        base_ch=224, ch_mult=(1, 2, 3, 4), time_dim=256,
    ).to(device)

    unet, scheduler = train_unet(
        unet, vae, text_enc, dataset, device,
        steps=steps_unet, lr=2e-4, batch_size=batch_size,
    )

    # Save
    total = (sum(p.numel() for p in unet.parameters()) +
             sum(p.numel() for p in vae.parameters()) +
             sum(p.numel() for p in text_enc.parameters()))

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    torch.save({
        "unet": unet.state_dict(),
        "vae": vae.state_dict(),
        "text_encoder": text_enc.state_dict(),
        "config": {
            "version": 3,
            "img_size": img_size,
            "latent_ch": latent_ch,
            "latent_size": latent_size,
            "context_dim": context_dim,
            "base_ch_unet": 224,
            "base_ch_vae": 128,
            "ch_mult": [1, 2, 3, 4],
            "time_dim": 256,
            "total_params": total,
            "text_enc": {"vocab": 5000, "embed": 128, "layers": 2, "max_len": 32},
        },
    }, save_path)

    elapsed = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"  Model saved: {os.path.abspath(save_path)}")
    print(f"  Size: {os.path.getsize(save_path)/1024/1024:.1f} MB")
    print(f"  Total params: {total:,} ({total/1e6:.1f}M)")
    print(f"  Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"{'='*60}\n")

    # Generate
    prompts = generate_prompts or COMPLEX_PROMPTS
    generate(unet, vae, text_enc, scheduler, prompts, device,
             latent_shape=(1, latent_ch, latent_size, latent_size),
             num_steps=50, output_dir=output_dir)

    return save_path


def load_and_generate(model_path, prompts, output_dir="output/own_model_v3",
                      num_steps=50):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Load] {model_path} ...")
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    te_cfg = cfg.get("text_enc", {})

    text_enc = MiniTextEncoder(
        vocab_size=te_cfg.get("vocab", 5000),
        embed_dim=te_cfg.get("embed", 128),
        context_dim=cfg["context_dim"],
        max_len=te_cfg.get("max_len", 32),
        num_layers=te_cfg.get("layers", 2),
    ).to(device)
    text_enc.load_state_dict(ckpt["text_encoder"])
    text_enc.eval()

    vae = SharpVAE(3, cfg["latent_ch"], cfg.get("base_ch_vae", 128)).to(device)
    vae.load_state_dict(ckpt["vae"])
    vae.eval()

    unet = MiniUNet(
        cfg["latent_ch"], cfg["latent_ch"], cfg["context_dim"],
        cfg.get("base_ch_unet", 224),
        tuple(cfg.get("ch_mult", [1,2,3,4])),
        cfg.get("time_dim", 256),
    ).to(device)
    unet.load_state_dict(ckpt["unet"])
    unet.eval()

    scheduler = SimpleScheduler(num_timesteps=1000)
    print(f"[Load] {cfg['total_params']:,} params ({cfg['total_params']/1e6:.1f}M)\n")

    generate(unet, vae, text_enc, scheduler, prompts, device,
             (1, cfg["latent_ch"], cfg["latent_size"], cfg["latent_size"]),
             num_steps, output_dir)


def main():
    p = argparse.ArgumentParser(description="Train 144M text-to-image model (v3)")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--steps-vae", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--img-size", type=int, default=64)
    p.add_argument("--num-samples", type=int, default=1000)
    p.add_argument("--prompt", type=str, default=None)
    p.add_argument("--load", type=str, default=None)
    p.add_argument("--save-path", type=str, default="train/own_model_v3.pt")
    p.add_argument("--output-dir", type=str, default="output/own_model_v3")
    args = p.parse_args()

    if args.load:
        prompts = [args.prompt] if args.prompt else COMPLEX_PROMPTS[:4]
        load_and_generate(args.load, prompts, args.output_dir)
    else:
        gen_prompts = [args.prompt] if args.prompt else None
        run_pipeline(
            steps_vae=args.steps_vae, steps_unet=args.steps,
            batch_size=args.batch_size, img_size=args.img_size,
            num_samples=args.num_samples,
            save_path=args.save_path, output_dir=args.output_dir,
            generate_prompts=gen_prompts,
        )


if __name__ == "__main__":
    main()
