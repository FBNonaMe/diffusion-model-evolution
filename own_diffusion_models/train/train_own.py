"""
Обучение СВОЕЙ модели генерации изображений из текста.

Полный цикл: датасет → обучение → генерация — на RTX 2050 (4 ГБ VRAM).

Как это работает:
  1. Создаём синтетический датасет (цветные паттерны с текстовыми описаниями)
  2. Кодируем изображения через VAE → латенты
  3. Обучаем маленький UNet (~30-50M) предсказывать шум на латентах
  4. При генерации: промпт → текстовый энкодер → UNet деноизинг → VAE декодер → изображение

Результат: модель, которая по текстовому промпту генерирует картинки!
(Качество будет простое — но это ВАША обученная модель.)

Использование::

    # Обучить модель (200 шагов, ~30 сек на RTX 2050)
    python -m train.train_own --steps 200

    # Больше шагов = лучше качество
    python -m train.train_own --steps 1000

    # Сгенерировать изображения обученной моделью
    python -m train.train_own --generate --prompt "red circle on blue background"

    # Обучить + сразу сгенерировать
    python -m train.train_own --steps 500 --generate
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from train.train_mini import MiniUNet, MiniTextEncoder, SimpleScheduler


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _flush():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()


def _vram_mb():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    return 0


# ------------------------------------------------------------------ #
# Synthetic Dataset — colored shapes with text descriptions
# ------------------------------------------------------------------ #

COLORS = {
    "red": (220, 50, 50),
    "green": (50, 200, 50),
    "blue": (50, 50, 220),
    "yellow": (230, 220, 50),
    "purple": (150, 50, 200),
    "orange": (240, 150, 30),
    "cyan": (50, 210, 210),
    "white": (240, 240, 240),
    "pink": (240, 130, 170),
}

BG_COLORS = {
    "black": (10, 10, 10),
    "white": (240, 240, 240),
    "gray": (128, 128, 128),
    "dark blue": (20, 20, 80),
    "dark green": (20, 60, 20),
}

SHAPES = ["circle", "square", "triangle", "diamond", "cross",
          "horizontal stripe", "vertical stripe", "diagonal",
          "gradient", "dots"]


def draw_shape(size: int, shape: str, color: tuple, bg_color: tuple) -> np.ndarray:
    """Draw a simple shape on a solid background. Pure numpy, no PIL needed."""
    img = np.full((size, size, 3), bg_color, dtype=np.uint8)
    cx, cy = size // 2, size // 2
    r = size // 3

    if shape == "circle":
        yy, xx = np.ogrid[:size, :size]
        mask = ((xx - cx)**2 + (yy - cy)**2) <= r**2
        img[mask] = color

    elif shape == "square":
        s = r
        img[cy-s:cy+s, cx-s:cx+s] = color

    elif shape == "triangle":
        for y in range(cy - r, cy + r):
            frac = (y - (cy - r)) / (2 * r)
            half_w = int(frac * r)
            img[y, cx-half_w:cx+half_w+1] = color

    elif shape == "diamond":
        for y in range(cy - r, cy + r):
            dist = abs(y - cy)
            half_w = r - dist
            if half_w > 0:
                img[y, cx-half_w:cx+half_w+1] = color

    elif shape == "cross":
        t = max(r // 4, 2)
        img[cy-r:cy+r, cx-t:cx+t] = color
        img[cy-t:cy+t, cx-r:cx+r] = color

    elif shape == "horizontal stripe":
        t = r // 2
        img[cy-t:cy+t, :] = color

    elif shape == "vertical stripe":
        t = r // 2
        img[:, cx-t:cx+t] = color

    elif shape == "diagonal":
        for i in range(size):
            t = max(r // 4, 2)
            for d in range(-t, t):
                y = i + d
                if 0 <= y < size and 0 <= i < size:
                    img[y, i] = color

    elif shape == "gradient":
        for x in range(size):
            frac = x / size
            c = tuple(int(bg_color[i] * (1-frac) + color[i] * frac) for i in range(3))
            img[:, x] = c

    elif shape == "dots":
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                for ox, oy in [(0,0), (-r, -r), (r, -r), (-r, r), (r, r)]:
                    yy, xx = cy + oy + dy, cx + ox + dx
                    if 0 <= yy < size and 0 <= xx < size:
                        img[yy, xx] = color

    return img


class SyntheticDataset(Dataset):
    """Generates colored shapes with text descriptions on-the-fly."""

    def __init__(self, num_samples: int = 500, img_size: int = 64,
                 text_encoder: MiniTextEncoder = None):
        self.num_samples = num_samples
        self.img_size = img_size
        self.text_encoder = text_encoder

        # Pre-generate all samples for consistency
        self.samples = []
        color_names = list(COLORS.keys())
        bg_names = list(BG_COLORS.keys())

        for i in range(num_samples):
            shape = SHAPES[i % len(SHAPES)]
            color_name = color_names[i % len(color_names)]
            bg_name = bg_names[i % len(bg_names)]

            # Avoid same color for shape and background
            if color_name == bg_name:
                bg_name = bg_names[(i + 1) % len(bg_names)]

            prompt = f"{color_name} {shape} on {bg_name} background"
            img = draw_shape(img_size, shape, COLORS[color_name], BG_COLORS[bg_name])

            # (H,W,3) uint8 → (3,H,W) float [0,1]
            img_t = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0

            # Tokenize prompt
            if text_encoder:
                tokens = text_encoder.simple_tokenize(prompt)
            else:
                tokens = torch.zeros(32, dtype=torch.long)

            self.samples.append({
                "image": img_t,
                "tokens": tokens,
                "prompt": prompt,
            })

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_images(batch):
    return {
        "images": torch.stack([b["image"] for b in batch]),
        "tokens": torch.stack([b["tokens"] for b in batch]),
        "prompts": [b["prompt"] for b in batch],
    }


# ------------------------------------------------------------------ #
# Simple VAE (lightweight, trainable, no pretrained model needed)
# ------------------------------------------------------------------ #

class MiniVAE(nn.Module):
    """Tiny VAE for encoding/decoding 64×64 images to 8×8 latents.

    ~2M parameters. Learns jointly with the UNet.
    """

    def __init__(self, in_channels=3, latent_channels=4, base_ch=64):
        super().__init__()
        # Encoder: 64→32→16→8
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_ch, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_ch, base_ch, 3, stride=2, padding=1),  # 32
            nn.SiLU(),
            nn.Conv2d(base_ch, base_ch * 2, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_ch * 2, base_ch * 2, 3, stride=2, padding=1),  # 16
            nn.SiLU(),
            nn.Conv2d(base_ch * 2, base_ch * 4, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_ch * 4, base_ch * 4, 3, stride=2, padding=1),  # 8
            nn.SiLU(),
        )
        self.enc_mu = nn.Conv2d(base_ch * 4, latent_channels, 1)
        self.enc_logvar = nn.Conv2d(base_ch * 4, latent_channels, 1)

        # Decoder: 8→16→32→64
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, base_ch * 4, 3, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(base_ch * 4, base_ch * 4, 4, stride=2, padding=1),  # 16
            nn.SiLU(),
            nn.Conv2d(base_ch * 4, base_ch * 2, 3, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(base_ch * 2, base_ch * 2, 4, stride=2, padding=1),  # 32
            nn.SiLU(),
            nn.Conv2d(base_ch * 2, base_ch, 3, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(base_ch, base_ch, 4, stride=2, padding=1),  # 64
            nn.SiLU(),
            nn.Conv2d(base_ch, in_channels, 3, padding=1),
            nn.Sigmoid(),
        )

        total = sum(p.numel() for p in self.parameters())
        print(f"[MiniVAE] Parameters: {total:,} ({total/1e6:.1f}M)")

    def encode(self, x):
        h = self.encoder(x)
        mu = self.enc_mu(h)
        logvar = self.enc_logvar(h)
        # Reparameterize
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z, mu, logvar

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        recon = self.decode(z)
        return recon, mu, logvar


# ------------------------------------------------------------------ #
# DDIM Sampler (fast generation)
# ------------------------------------------------------------------ #

class DDIMSampler:
    """Deterministic DDIM sampler for fast generation."""

    def __init__(self, scheduler: SimpleScheduler):
        self.scheduler = scheduler
        self.num_timesteps = scheduler.num_timesteps
        self.alphas_cumprod = scheduler.alphas_cumprod

    @torch.no_grad()
    def sample(self, model, shape, context, device,
               num_steps=20, eta=0.0):
        """Generate samples using DDIM."""
        # Create timestep schedule
        step_size = self.num_timesteps // num_steps
        timesteps = list(range(self.num_timesteps - 1, -1, -step_size))

        # Start from pure noise
        x = torch.randn(shape, device=device)
        acp = self.alphas_cumprod.to(device)

        for i, t in enumerate(timesteps):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)

            # Predict noise
            pred_noise = model(x, t_batch, context)

            # DDIM update
            alpha_t = acp[t]
            alpha_prev = acp[timesteps[i+1]] if i + 1 < len(timesteps) else torch.tensor(1.0, device=device)

            # Predicted x0
            x0_pred = (x - (1 - alpha_t).sqrt() * pred_noise) / alpha_t.sqrt()
            x0_pred = x0_pred.clamp(-3, 3)  # stability

            # Direction pointing to x_t
            dir_xt = (1 - alpha_prev).sqrt() * pred_noise

            # No noise for DDIM (eta=0)
            x = alpha_prev.sqrt() * x0_pred + dir_xt

        return x


# ------------------------------------------------------------------ #
# Phase 1: Train VAE (learn to compress images)
# ------------------------------------------------------------------ #

def train_vae(vae, dataset, device, steps=200, lr=1e-3, batch_size=16):
    """Train VAE to encode/decode images."""
    print(f"\n{'='*60}")
    print(f"  Phase 1: Training VAE")
    print(f"{'='*60}")

    vae.train()
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        collate_fn=collate_images, drop_last=True)

    step = 0
    t0 = time.time()
    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    while step < steps:
        for batch in loader:
            if step >= steps:
                break

            images = batch["images"].to(device)
            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=use_amp):
                recon, mu, logvar = vae(images)
                # Reconstruction loss
                recon_loss = F.mse_loss(recon, images)
                # KL divergence
                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + 0.001 * kl_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            step += 1

            if step % 50 == 0 or step == 1:
                print(f"  step {step:4d} | recon={recon_loss.item():.4f} | "
                      f"kl={kl_loss.item():.4f} | VRAM={_vram_mb():.0f}MB")

    elapsed = time.time() - t0
    print(f"  VAE trained: {steps} steps in {elapsed:.1f}s\n")
    vae.eval()
    return vae


# ------------------------------------------------------------------ #
# Phase 2: Train UNet (learn to denoise latents)
# ------------------------------------------------------------------ #

def train_unet(unet, vae, text_enc, dataset, device,
               steps=500, lr=3e-4, batch_size=8):
    """Train UNet to denoise VAE latents conditioned on text."""
    print(f"\n{'='*60}")
    print(f"  Phase 2: Training UNet (denoising)")
    print(f"{'='*60}")

    scheduler = SimpleScheduler(num_timesteps=1000)
    unet.train()
    vae.eval()
    text_enc.eval()

    optimizer = torch.optim.AdamW(unet.parameters(), lr=lr, weight_decay=0.01)
    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps, eta_min=lr * 0.01)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        collate_fn=collate_images, drop_last=True)

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

            images = batch["images"].to(device)  # B,3,H,W
            tokens = batch["tokens"].to(device)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=use_amp):
                # Encode images to latents (frozen VAE)
                with torch.no_grad():
                    z, _, _ = vae.encode(images)

                # Random timesteps
                t = torch.randint(0, 1000, (z.shape[0],), device=device)

                # Add noise
                noise = torch.randn_like(z)
                noisy_z = scheduler.add_noise(z, noise, t)

                # Text conditioning
                with torch.no_grad():
                    context = text_enc(tokens)

                # Predict noise
                pred = unet(noisy_z, t, context)
                loss = F.mse_loss(pred, noise)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            lr_sched.step()

            step += 1
            losses.append(loss.item())

            if step % 50 == 0 or step == 1:
                avg = sum(losses[-50:]) / min(len(losses), 50)
                print(f"  {step:6d} | {loss.item():10.6f} | "
                      f"{avg:10.6f} | {_vram_mb():8.0f}MB")

    elapsed = time.time() - t0
    initial = sum(losses[:20]) / min(len(losses), 20)
    final = sum(losses[-20:]) / min(len(losses), 20)
    print(f"\n  UNet trained: {steps} steps in {elapsed:.1f}s")
    print(f"  Loss: {initial:.4f} → {final:.4f} ({(1-final/initial)*100:+.1f}%)\n")

    return unet, scheduler


# ------------------------------------------------------------------ #
# Phase 3: Generate images!
# ------------------------------------------------------------------ #

@torch.no_grad()
def generate(unet, vae, text_enc, scheduler, prompts: list[str],
             device, latent_shape=(1, 4, 8, 8), num_steps=30,
             output_dir="output/own_model"):
    """Generate images from text prompts using trained model."""
    print(f"\n{'='*60}")
    print(f"  Generating images with YOUR model!")
    print(f"{'='*60}")

    os.makedirs(output_dir, exist_ok=True)
    unet.eval()
    vae.eval()
    text_enc.eval()

    sampler = DDIMSampler(scheduler)

    for i, prompt in enumerate(prompts):
        print(f"\n  [{i+1}/{len(prompts)}] \"{prompt}\"")

        # Encode prompt
        tokens = text_enc.simple_tokenize(prompt).unsqueeze(0).to(device)
        context = text_enc(tokens)

        # Sample latents via DDIM
        shape = (1, latent_shape[1], latent_shape[2], latent_shape[3])
        z = sampler.sample(unet, shape, context, device, num_steps=num_steps)

        # Decode to image
        img = vae.decode(z)
        img = img[0].clamp(0, 1)  # (3, H, W)

        # Save as numpy → PNG (no PIL needed)
        img_np = (img.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        # Save as BMP (simpler than PNG, no extra libs)
        path = os.path.join(output_dir, f"gen_{i:03d}.bmp")
        _save_bmp(path, img_np)
        print(f"  Saved: {path} ({img_np.shape[1]}×{img_np.shape[0]})")

    print(f"\n  All images saved to: {os.path.abspath(output_dir)}\n")
    return output_dir


def _save_bmp(path: str, img: np.ndarray):
    """Save RGB image as BMP file (no PIL required)."""
    h, w, c = img.shape
    # BMP is stored bottom-up, BGR
    img_bgr = img[::-1, :, ::-1].copy()
    row_size = (w * 3 + 3) & ~3  # rows must be 4-byte aligned
    pixel_size = row_size * h
    file_size = 54 + pixel_size

    with open(path, "wb") as f:
        # File header (14 bytes)
        f.write(b"BM")
        f.write(file_size.to_bytes(4, "little"))
        f.write(b"\x00\x00\x00\x00")
        f.write((54).to_bytes(4, "little"))
        # Info header (40 bytes)
        f.write((40).to_bytes(4, "little"))
        f.write(w.to_bytes(4, "little"))
        f.write(h.to_bytes(4, "little"))
        f.write((1).to_bytes(2, "little"))   # planes
        f.write((24).to_bytes(2, "little"))  # bpp
        f.write(b"\x00" * 4)                 # compression
        f.write(pixel_size.to_bytes(4, "little"))
        f.write(b"\x00" * 16)                # resolution + colors
        # Pixel data
        for y in range(h):
            row = img_bgr[y].tobytes()
            padding = b"\x00" * (row_size - w * 3)
            f.write(row + padding)


# ------------------------------------------------------------------ #
# Full pipeline
# ------------------------------------------------------------------ #

def run_full_pipeline(
    steps_vae: int = 200,
    steps_unet: int = 500,
    batch_size: int = 8,
    img_size: int = 64,
    num_samples: int = 500,
    base_ch_unet: int = 96,
    do_generate: bool = True,
    generate_prompts: list[str] | None = None,
    save_path: str = "train/own_model.pt",
    output_dir: str = "output/own_model",
):
    """Full training pipeline: dataset → VAE → UNet → generation."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*60}")
    print(f"  Training YOUR OWN Model")
    print(f"{'='*60}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Image size: {img_size}×{img_size}")
    print(f"  Dataset: {num_samples} synthetic images")
    print(f"  VAE steps: {steps_vae}")
    print(f"  UNet steps: {steps_unet}")
    print(f"{'='*60}\n")

    t_total = time.time()

    # ---- Text encoder (frozen) ----
    context_dim = 256
    text_enc = MiniTextEncoder(
        vocab_size=3000, embed_dim=64,
        context_dim=context_dim, max_len=16, num_layers=1,
    ).to(device)
    text_enc.eval()
    text_enc.requires_grad_(False)

    # ---- Dataset ----
    print(f"  Creating synthetic dataset ({num_samples} images) ...")
    dataset = SyntheticDataset(
        num_samples=num_samples,
        img_size=img_size,
        text_encoder=text_enc,
    )
    print(f"  Example prompts:")
    for s in dataset.samples[:5]:
        print(f"    - {s['prompt']}")
    print()

    # ---- VAE ----
    latent_ch = 4
    vae = MiniVAE(in_channels=3, latent_channels=latent_ch, base_ch=48).to(device)
    vae = train_vae(vae, dataset, device, steps=steps_vae,
                    lr=1e-3, batch_size=min(batch_size, 16))

    # ---- UNet ----
    # latent spatial size: img_size / 8
    latent_size = img_size // 8
    unet = MiniUNet(
        in_channels=latent_ch,
        out_channels=latent_ch,
        context_dim=context_dim,
        base_ch=base_ch_unet,
        ch_mult=(1, 2, 3),
        time_dim=128,
    ).to(device)

    unet, scheduler = train_unet(
        unet, vae, text_enc, dataset, device,
        steps=steps_unet, lr=3e-4,
        batch_size=min(batch_size, 8),
    )

    # ---- Save model ----
    total_params = (
        sum(p.numel() for p in unet.parameters()) +
        sum(p.numel() for p in vae.parameters()) +
        sum(p.numel() for p in text_enc.parameters())
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        "unet": unet.state_dict(),
        "vae": vae.state_dict(),
        "text_encoder": text_enc.state_dict(),
        "config": {
            "img_size": img_size,
            "latent_ch": latent_ch,
            "latent_size": latent_size,
            "context_dim": context_dim,
            "base_ch_unet": base_ch_unet,
            "ch_mult": [1, 2, 3],
            "total_params": total_params,
        },
        "training": {
            "steps_vae": steps_vae,
            "steps_unet": steps_unet,
            "num_samples": num_samples,
        },
    }, save_path)

    elapsed = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"  Model saved: {os.path.abspath(save_path)}")
    print(f"  Size: {os.path.getsize(save_path)/1024/1024:.1f} MB")
    print(f"  Total params: {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"{'='*60}\n")

    # ---- Generate ----
    if do_generate:
        if generate_prompts is None:
            generate_prompts = [
                "red circle on black background",
                "blue square on white background",
                "green triangle on gray background",
                "yellow diamond on dark blue background",
                "purple cross on white background",
                "orange horizontal stripe on black background",
                "cyan vertical stripe on gray background",
                "pink dots on dark green background",
            ]

        generate(
            unet, vae, text_enc, scheduler,
            prompts=generate_prompts,
            device=device,
            latent_shape=(1, latent_ch, latent_size, latent_size),
            num_steps=30,
            output_dir=output_dir,
        )

    return save_path


# ------------------------------------------------------------------ #
# Load and generate from saved model
# ------------------------------------------------------------------ #

def load_and_generate(
    model_path: str,
    prompts: list[str],
    output_dir: str = "output/own_model",
    num_steps: int = 30,
):
    """Load a saved model and generate images."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[Load] Loading model from {model_path} ...")
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]

    # Rebuild models
    text_enc = MiniTextEncoder(
        vocab_size=3000, embed_dim=64,
        context_dim=cfg["context_dim"], max_len=16, num_layers=1,
    ).to(device)
    text_enc.load_state_dict(ckpt["text_encoder"])
    text_enc.eval()

    vae = MiniVAE(
        in_channels=3, latent_channels=cfg["latent_ch"], base_ch=48,
    ).to(device)
    vae.load_state_dict(ckpt["vae"])
    vae.eval()

    unet = MiniUNet(
        in_channels=cfg["latent_ch"],
        out_channels=cfg["latent_ch"],
        context_dim=cfg["context_dim"],
        base_ch=cfg["base_ch_unet"],
        ch_mult=tuple(cfg["ch_mult"]),
        time_dim=128,
    ).to(device)
    unet.load_state_dict(ckpt["unet"])
    unet.eval()

    scheduler = SimpleScheduler(num_timesteps=1000)

    print(f"[Load] Model loaded: {cfg['total_params']:,} params\n")

    generate(
        unet, vae, text_enc, scheduler,
        prompts=prompts,
        device=device,
        latent_shape=(1, cfg["latent_ch"],
                      cfg["latent_size"], cfg["latent_size"]),
        num_steps=num_steps,
        output_dir=output_dir,
    )


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(
        description="Train your own text-to-image model (~30M params)"
    )
    parser.add_argument("--steps", type=int, default=500,
                        help="UNet training steps (default: 500)")
    parser.add_argument("--steps-vae", type=int, default=200,
                        help="VAE training steps (default: 200)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--img-size", type=int, default=64,
                        help="Image size (default: 64)")
    parser.add_argument("--num-samples", type=int, default=500,
                        help="Dataset size")
    parser.add_argument("--base-ch", type=int, default=96,
                        help="UNet base channels (96=~18M, 128=~30M)")
    parser.add_argument("--generate", action="store_true",
                        help="Generate images after training")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Custom prompt for generation")
    parser.add_argument("--load", type=str, default=None,
                        help="Load saved model and generate (skip training)")
    parser.add_argument("--save-path", type=str,
                        default="train/own_model.pt")
    parser.add_argument("--output-dir", type=str,
                        default="output/own_model")
    args = parser.parse_args()

    if args.load:
        # Generate from saved model
        prompts = [args.prompt] if args.prompt else [
            "red circle on black background",
            "blue square on white background",
            "green triangle on gray background",
        ]
        load_and_generate(args.load, prompts, args.output_dir)
    else:
        # Train and optionally generate
        gen_prompts = [args.prompt] if args.prompt else None
        run_full_pipeline(
            steps_vae=args.steps_vae,
            steps_unet=args.steps,
            batch_size=args.batch_size,
            img_size=args.img_size,
            num_samples=args.num_samples,
            base_ch_unet=args.base_ch,
            do_generate=args.generate,
            generate_prompts=gen_prompts,
            save_path=args.save_path,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
