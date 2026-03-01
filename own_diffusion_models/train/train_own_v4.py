"""
Train on REAL photos (CIFAR-10) — v4 fixed.

Root causes of gray squares in previous version:
  1. SharpVAE skip connections: encoder passes s1/s2 to decoder during
     training (85% of time), but at generation s1=s2=None → decoder gets
     zeros for half its input → gray blobs.
  2. UNet 136M is way too large for 3000 training steps on 30K images
     (only 0.2 epochs!). Model can't converge.
  3. Hash tokenizer maps unrelated words to random vectors — text
     conditioning is useless.

Fixes:
  - GenVAE: NO skip connections. Decoder works only from latent z.
    8 latent channels (instead of 4) → more information capacity.
  - Compact UNet ~15M params: converges in reasonable number of steps.
  - Class embedding: nn.Embedding(10, dim) → reliable conditioning.
  - More training: VAE 5000 steps, UNet 15000 steps.
  - Total ~19M params, ~15 min on RTX 2050, 2.5 GB VRAM.

Usage::

    python -m train.train_own_v4
    python -m train.train_own_v4 --animals-only
    python -m train.train_own_v4 --steps 20000 --steps-vae 8000
    python -m train.train_own_v4 --load train/own_model_v4.pt --prompt "cat"
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import random
import struct
import sys
import time
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ================================================================== #
#  CIFAR-10 metadata
# ================================================================== #

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

ANIMAL_CLASSES = {"bird", "cat", "deer", "dog", "frog", "horse"}


# ================================================================== #
#  Helpers
# ================================================================== #

def _flush():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _vram_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 ** 2
    return 0.0


def _save_bmp(path: str, img_np: np.ndarray):
    """Save RGB uint8 HWC array as 24-bit BMP."""
    h, w, _ = img_np.shape
    row_bytes = w * 3
    pad = (4 - row_bytes % 4) % 4
    stride = row_bytes + pad
    pixel_size = stride * h
    file_size = 54 + pixel_size
    with open(path, "wb") as f:
        f.write(b"BM")
        f.write(struct.pack("<I", file_size))
        f.write(struct.pack("<HH", 0, 0))
        f.write(struct.pack("<I", 54))
        f.write(struct.pack("<I", 40))
        f.write(struct.pack("<i", w))
        f.write(struct.pack("<i", h))
        f.write(struct.pack("<HH", 1, 24))
        f.write(struct.pack("<I", 0))
        f.write(struct.pack("<I", pixel_size))
        f.write(struct.pack("<i", 2835))
        f.write(struct.pack("<i", 2835))
        f.write(struct.pack("<I", 0))
        f.write(struct.pack("<I", 0))
        for y in range(h - 1, -1, -1):
            row = img_np[y]
            bgr = row[:, ::-1].tobytes()
            f.write(bgr)
            if pad:
                f.write(b"\x00" * pad)


# ================================================================== #
#  GenVAE — VAE without skip connections (works for generation!)
# ================================================================== #

class ResBlock(nn.Module):
    """Residual block with GroupNorm."""

    def __init__(self, ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(min(32, ch), ch),
            nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.GroupNorm(min(32, ch), ch),
            nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)


class GenVAE(nn.Module):
    """VAE designed for generation — NO skip connections.

    The decoder must reconstruct everything from the latent code alone.
    Uses more latent channels (8) to compensate for lack of skip info.
    Deeper encoder/decoder with residual blocks for better features.
    """

    def __init__(self, in_channels: int = 3, latent_channels: int = 8,
                 base_ch: int = 64):
        super().__init__()
        ch = base_ch

        # ---- Encoder ----
        # 3→ch (full res)
        self.enc_in = nn.Conv2d(in_channels, ch, 3, padding=1)

        # Level 1: full → ÷2
        self.enc1 = nn.Sequential(
            ResBlock(ch),
            ResBlock(ch),
        )
        self.down1 = nn.Conv2d(ch, ch * 2, 3, stride=2, padding=1)

        # Level 2: ÷2 → ÷4
        ch2 = ch * 2
        self.enc2 = nn.Sequential(
            ResBlock(ch2),
            ResBlock(ch2),
        )
        self.down2 = nn.Conv2d(ch2, ch2, 3, stride=2, padding=1)

        # Bottleneck at ÷4
        self.enc_mid = nn.Sequential(
            ResBlock(ch2),
            ResBlock(ch2),
            ResBlock(ch2),
        )

        # To latent (mu, logvar)
        self.to_mu = nn.Conv2d(ch2, latent_channels, 1)
        self.to_logvar = nn.Conv2d(ch2, latent_channels, 1)

        # ---- Decoder ----
        self.from_z = nn.Conv2d(latent_channels, ch2, 3, padding=1)

        # Bottleneck
        self.dec_mid = nn.Sequential(
            ResBlock(ch2),
            ResBlock(ch2),
            ResBlock(ch2),
        )

        # Level 1: ÷4 → ÷2
        self.up1 = nn.ConvTranspose2d(ch2, ch2, 4, stride=2, padding=1)
        self.dec1 = nn.Sequential(
            ResBlock(ch2),
            ResBlock(ch2),
        )

        # Level 2: ÷2 → full
        self.up2 = nn.ConvTranspose2d(ch2, ch, 4, stride=2, padding=1)
        self.dec2 = nn.Sequential(
            ResBlock(ch),
            ResBlock(ch),
        )

        # Output
        self.dec_out = nn.Sequential(
            nn.GroupNorm(min(32, ch), ch),
            nn.SiLU(),
            nn.Conv2d(ch, in_channels, 3, padding=1),
            nn.Sigmoid(),
        )

        total = sum(p.numel() for p in self.parameters())
        print(f"[GenVAE] Parameters: {total:,} ({total/1e6:.1f}M)")

    def encode(self, x):
        h = self.enc_in(x)
        h = self.enc1(h)
        h = self.down1(h)
        h = self.enc2(h)
        h = self.down2(h)
        h = self.enc_mid(h)

        mu = self.to_mu(h)
        logvar = self.to_logvar(h).clamp(-10, 10)

        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)

        return z, mu, logvar

    def decode(self, z):
        h = F.silu(self.from_z(z))
        h = self.dec_mid(h)
        h = self.up1(h)
        h = self.dec1(h)
        h = self.up2(h)
        h = self.dec2(h)
        return self.dec_out(h)

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        recon = self.decode(z)
        return recon, mu, logvar


# ================================================================== #
#  PatchDiscriminator
# ================================================================== #

class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=3, base_ch=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.utils.spectral_norm(
                nn.Conv2d(in_channels, base_ch, 4, 2, 1)),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(
                nn.Conv2d(base_ch, base_ch * 2, 4, 2, 1)),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(
                nn.Conv2d(base_ch * 2, base_ch * 4, 4, 2, 1)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_ch * 4, 1, 4, 1, 1),
        )
        total = sum(p.numel() for p in self.parameters())
        print(f"[PatchDisc] Parameters: {total:,} ({total/1e6:.1f}M)")

    def forward(self, x):
        return self.net(x)


# ================================================================== #
#  Class-conditional embedding (replaces broken hash tokenizer)
# ================================================================== #

class ClassConditioner(nn.Module):
    """Maps class labels to context vectors for cross-attention.

    Output shape: (B, num_tokens, context_dim) — compatible with
    MiniUNet's CrossAttention.

    Also supports text-to-class mapping for inference.
    """

    def __init__(self, num_classes: int = 10, context_dim: int = 256,
                 num_tokens: int = 4):
        super().__init__()
        self.num_classes = num_classes
        self.context_dim = context_dim
        self.num_tokens = num_tokens

        self.embed = nn.Embedding(num_classes, context_dim * num_tokens)
        self.norm = nn.LayerNorm(context_dim)

        # Text → class mapping
        self._text_map = {}
        for i, cls in enumerate(CIFAR10_CLASSES):
            self._text_map[cls] = i
            self._text_map[cls + "s"] = i  # plural

        # Common synonyms
        self._text_map.update({
            "puppy": 5, "puppies": 5, "doggy": 5,
            "kitten": 3, "kitty": 3, "kittens": 3,
            "plane": 0, "jet": 0, "aircraft": 0,
            "car": 1, "auto": 1, "vehicle": 1, "cars": 1,
            "boat": 8, "ships": 8, "vessel": 8, "sailing": 8,
            "lorry": 9, "trucks": 9,
            "stag": 4, "doe": 4, "fawn": 4,
            "toad": 6, "frogs": 6,
            "pony": 7, "stallion": 7, "mare": 7, "horses": 7,
            "sparrow": 2, "robin": 2, "eagle": 2, "parrot": 2,
            "owl": 2, "pigeon": 2, "crow": 2, "finch": 2,
            "animal": 3, "pet": 3,
        })

        total = sum(p.numel() for p in self.parameters())
        print(f"[ClassCond] Parameters: {total:,} ({total/1e6:.2f}M)")

    def forward(self, class_ids: torch.Tensor) -> torch.Tensor:
        """class_ids: (B,) long → (B, num_tokens, context_dim)"""
        h = self.embed(class_ids)  # (B, context_dim * num_tokens)
        h = h.view(-1, self.num_tokens, self.context_dim)
        h = self.norm(h)
        return h

    def text_to_class(self, text: str) -> int:
        """Map text prompt to class ID."""
        text_lower = text.lower()
        for word, cls_id in self._text_map.items():
            if word in text_lower:
                return cls_id
        # Default to random
        return random.randint(0, self.num_classes - 1)


# ================================================================== #
#  Compact UNet for diffusion (reuse MiniUNet with smaller config)
# ================================================================== #

# We import MiniUNet but use smaller hyperparameters
from train.train_mini import MiniUNet, SimpleScheduler


# ================================================================== #
#  CIFAR-10 Dataset
# ================================================================== #

class CIFAR10Dataset(Dataset):
    """CIFAR-10 with class labels for conditioning."""

    def __init__(self, root="data/cifar10", img_size=64,
                 animals_only=False, max_samples=0):
        import torchvision
        import torchvision.transforms as T

        self.img_size = img_size

        print(f"  Loading CIFAR-10 ...")
        self.transform = T.Compose([
            T.Resize(img_size, interpolation=T.InterpolationMode.BILINEAR),
            T.RandomHorizontalFlip(0.5),
            T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05),
            T.RandomRotation(5),
            T.ToTensor(),
        ])
        # Simpler transform for VAE (no rotation — easier to learn)
        self.transform_vae = T.Compose([
            T.Resize(img_size, interpolation=T.InterpolationMode.BILINEAR),
            T.RandomHorizontalFlip(0.5),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            T.ToTensor(),
        ])

        cifar = torchvision.datasets.CIFAR10(
            root=root, train=True, download=True,
        )

        if animals_only:
            animal_ids = {i for i, c in enumerate(CIFAR10_CLASSES)
                          if c in ANIMAL_CLASSES}
            indices = [i for i, (_, label) in enumerate(cifar)
                       if label in animal_ids]
            print(f"  Filtered to animals: {len(indices)} images")
        else:
            indices = list(range(len(cifar)))

        if max_samples > 0 and max_samples < len(indices):
            random.seed(42)
            indices = random.sample(indices, max_samples)

        self.cifar = cifar
        self.indices = indices
        self.use_vae_transform = True

        print(f"  Dataset: {len(self.indices)} images, {img_size}×{img_size}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        pil_img, label = self.cifar[real_idx]
        if self.use_vae_transform:
            img_t = self.transform_vae(pil_img)
        else:
            img_t = self.transform(pil_img)
        return {"image": img_t, "label": label}


def collate_fn(batch):
    return {
        "images": torch.stack([b["image"] for b in batch]),
        "labels": torch.tensor([b["label"] for b in batch], dtype=torch.long),
    }


# ================================================================== #
#  Edge loss
# ================================================================== #

def sobel_edges(x: torch.Tensor) -> torch.Tensor:
    """Compute edge magnitude using Sobel filters."""
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                            dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                            dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    gray = x.mean(dim=1, keepdim=True)
    gx = F.conv2d(gray, sobel_x, padding=1)
    gy = F.conv2d(gray, sobel_y, padding=1)
    return (gx ** 2 + gy ** 2).sqrt()


def edge_loss(recon, target):
    return F.l1_loss(sobel_edges(recon), sobel_edges(target))


# ================================================================== #
#  DDIM Sampler
# ================================================================== #

class DDIMSampler:
    def __init__(self, scheduler: SimpleScheduler):
        self.num_timesteps = scheduler.num_timesteps
        self.alphas_cumprod = scheduler.alphas_cumprod

    def sample(self, unet, shape, context, device,
               num_steps=50, cfg_scale=3.0, uncond_context=None):
        """DDIM sampling with classifier-free guidance."""
        acp = self.alphas_cumprod.to(device)
        step_size = max(1, self.num_timesteps // num_steps)
        timesteps = list(range(self.num_timesteps - 1, -1, -step_size))

        x = torch.randn(shape, device=device)

        for j, t in enumerate(timesteps):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)

            pred_cond = unet(x, t_batch, context)

            if cfg_scale > 1.0 and uncond_context is not None:
                pred_unc = unet(x, t_batch, uncond_context)
                pred = pred_unc + cfg_scale * (pred_cond - pred_unc)
            else:
                pred = pred_cond

            alpha_t = acp[t]
            alpha_prev = acp[timesteps[j + 1]] if j + 1 < len(timesteps) \
                else torch.tensor(1.0, device=device)

            x0_pred = (x - (1 - alpha_t).sqrt() * pred) / alpha_t.sqrt()
            x0_pred = x0_pred.clamp(-5, 5)
            dir_xt = (1 - alpha_prev).sqrt() * pred
            x = alpha_prev.sqrt() * x0_pred + dir_xt

        return x


# ================================================================== #
#  Phase 1: Train VAE
# ================================================================== #

def train_vae(vae: GenVAE, disc: PatchDiscriminator, dataset, device,
              steps=5000, lr=1e-4, batch_size=16):
    print(f"\n{'='*60}")
    print(f"  Phase 1: Train GenVAE (NO skip connections)")
    print(f"  Steps: {steps}, Batch: {batch_size}")
    print(f"{'='*60}")

    vae.train()
    disc.train()

    opt_vae = torch.optim.Adam(vae.parameters(), lr=lr, betas=(0.9, 0.999))
    opt_disc = torch.optim.Adam(disc.parameters(), lr=lr * 0.5, betas=(0.5, 0.999))

    # LR warmup + decay
    warmup = min(500, steps // 5)

    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        return max(0.1, 1.0 - (step - warmup) / max(1, steps - warmup))

    sched_vae = torch.optim.lr_scheduler.LambdaLR(opt_vae, lr_lambda)

    dataset.use_vae_transform = True
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        collate_fn=collate_fn, drop_last=True,
                        num_workers=0, pin_memory=True)

    # Disable AMP for VAE — fp16 causes NaN in KL divergence
    use_amp_d = torch.cuda.is_available()
    scaler_d = torch.amp.GradScaler("cuda", enabled=use_amp_d)

    gan_start = max(steps // 4, 500)
    step = 0
    t0 = time.time()
    d_loss_val = 0.0

    print(f"  GAN activates at step {gan_start}")
    print(f"  {'step':>6} | {'L1':>8} | {'edge':>8} | {'KL':>8} | "
          f"{'G':>8} | {'D':>8} | {'VRAM':>7}")
    print(f"  {'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-"
          f"{'-'*8}-+-{'-'*8}-+-{'-'*7}")

    while step < steps:
        for batch in loader:
            if step >= steps:
                break

            images = batch["images"].to(device)

            # ---- Discriminator ----
            if step >= gan_start and step % 2 == 0:
                opt_disc.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=use_amp_d):
                    with torch.no_grad():
                        recon_d, _, _ = vae(images)
                    real_p = disc(images)
                    fake_p = disc(recon_d.detach())
                    d_loss = 0.5 * (
                        F.binary_cross_entropy_with_logits(
                            real_p, 0.9 * torch.ones_like(real_p)) +
                        F.binary_cross_entropy_with_logits(
                            fake_p, 0.1 * torch.ones_like(fake_p))
                    )
                scaler_d.scale(d_loss).backward()
                scaler_d.step(opt_disc)
                scaler_d.update()
                d_loss_val = d_loss.item()

            # ---- VAE (fp32 to avoid NaN in KL) ----
            opt_vae.zero_grad(set_to_none=True)
            recon, mu, logvar = vae(images)

            l1 = F.l1_loss(recon, images)
            e_loss = edge_loss(recon, images)
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

            # Multi-scale reconstruction
            ms = torch.tensor(0.0, device=device)
            for s in [2, 4]:
                ms = ms + F.l1_loss(F.avg_pool2d(recon, s),
                                    F.avg_pool2d(images, s))
            ms = ms * 0.5

            # KL warmup: 0 → 0.0001 over first 2000 steps
            kl_w = min(0.0001, 0.0001 * step / max(1, min(2000, steps // 3)))
            vae_loss = l1 + 0.3 * e_loss + 0.2 * ms + kl_w * kl

            if step >= gan_start:
                progress = min(1.0, (step - gan_start) /
                               max(1, steps - gan_start))
                gw = 0.005 + 0.03 * progress
                g_loss = F.binary_cross_entropy_with_logits(
                    disc(recon), torch.ones_like(disc(recon)))
                vae_loss = vae_loss + gw * g_loss
                g_val = g_loss.item()
            else:
                g_val = 0.0

            vae_loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
            opt_vae.step()
            sched_vae.step()

            step += 1

            if step % 250 == 0 or step == 1:
                print(f"  {step:6d} | {l1.item():8.4f} | {e_loss.item():8.4f} | "
                      f"{kl.item():8.4f} | {g_val:8.4f} | {d_loss_val:8.4f} | "
                      f"{_vram_mb():7.0f}MB")

    elapsed = time.time() - t0
    print(f"\n  VAE done: {steps} steps, {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Final L1: {l1.item():.4f}")
    vae.eval()
    return vae


# ================================================================== #
#  Phase 2: Train UNet
# ================================================================== #

def train_unet(unet, vae, cond, dataset, device,
               steps=15000, lr=2e-4, batch_size=8):
    print(f"\n{'='*60}")
    print(f"  Phase 2: Train UNet on real photo latents")
    print(f"  Steps: {steps}, Batch: {batch_size}")
    print(f"{'='*60}")

    scheduler = SimpleScheduler(num_timesteps=1000)
    unet.train()
    vae.eval()
    cond.train()

    params = list(unet.parameters()) + list(cond.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps, eta_min=lr * 0.01)

    dataset.use_vae_transform = False  # use augmented transform
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        collate_fn=collate_fn, drop_last=True,
                        num_workers=0, pin_memory=True)

    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    step = 0
    losses = []
    t0 = time.time()

    # 10% unconditional (for CFG at inference)
    uncond_drop_rate = 0.1

    print(f"  Unconditional drop: {uncond_drop_rate*100:.0f}%")
    print(f"  {'step':>6} | {'loss':>10} | {'avg':>10} | {'VRAM':>7}")
    print(f"  {'-'*6}-+-{'-'*10}-+-{'-'*10}-+-{'-'*7}")

    while step < steps:
        for batch in loader:
            if step >= steps:
                break

            images = batch["images"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                with torch.no_grad():
                    z, _, _ = vae.encode(images)

                # Class conditioning (with occasional uncond for CFG)
                if random.random() < uncond_drop_rate:
                    # Use "null" class = num_classes (need to handle)
                    context = torch.zeros(labels.shape[0], cond.num_tokens,
                                         cond.context_dim, device=device)
                else:
                    context = cond(labels)

                t = torch.randint(0, 1000, (z.shape[0],), device=device)
                noise = torch.randn_like(z)
                noisy_z = scheduler.add_noise(z, noise, t)

                pred = unet(noisy_z, t, context)
                loss = F.mse_loss(pred, noise) + 0.5 * F.l1_loss(pred, noise)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(optimizer)
            scaler.update()
            lr_sched.step()

            step += 1
            losses.append(loss.item())

            if step % 500 == 0 or step == 1:
                avg = sum(losses[-500:]) / min(len(losses), 500)
                elapsed = time.time() - t0
                eta = elapsed / step * (steps - step)
                print(f"  {step:6d} | {loss.item():10.5f} | "
                      f"{avg:10.5f} | {_vram_mb():7.0f}MB  "
                      f"[{elapsed:.0f}s / ~{eta:.0f}s ETA]")

    elapsed = time.time() - t0
    init = sum(losses[:100]) / min(len(losses), 100)
    fin = sum(losses[-100:]) / min(len(losses), 100)
    print(f"\n  UNet done: {steps} steps, {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Loss: {init:.5f} → {fin:.5f} ({(1-fin/init)*100:+.1f}%)")

    return unet, cond, scheduler


# ================================================================== #
#  Generation
# ================================================================== #

@torch.no_grad()
def generate(unet, vae, cond, scheduler, prompts, device,
             latent_shape, num_steps=50, cfg_scale=3.0,
             output_dir="output/own_model_v4"):
    print(f"\n{'='*60}")
    print(f"  Generating real images")
    print(f"{'='*60}")

    os.makedirs(output_dir, exist_ok=True)
    unet.eval()
    vae.eval()
    cond.eval()

    sampler = DDIMSampler(scheduler)

    # Unconditional context for CFG
    uncond_ctx = torch.zeros(1, cond.num_tokens, cond.context_dim,
                             device=device)

    for i, prompt in enumerate(prompts):
        cls_id = cond.text_to_class(prompt)
        cls_name = CIFAR10_CLASSES[cls_id]
        print(f"\n  [{i+1}/{len(prompts)}] \"{prompt}\" → {cls_name} (class {cls_id})")

        cls_t = torch.tensor([cls_id], device=device)
        context = cond(cls_t)

        x = sampler.sample(
            unet, latent_shape, context, device,
            num_steps=num_steps,
            cfg_scale=cfg_scale,
            uncond_context=uncond_ctx,
        )

        img = vae.decode(x)
        img = img[0].clamp(0, 1)
        img_np = (img.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        path = os.path.join(output_dir, f"gen_{i:03d}.bmp")
        _save_bmp(path, img_np)
        print(f"  Saved: {path} ({img_np.shape[1]}×{img_np.shape[0]})")

    print(f"\n  All saved to: {os.path.abspath(output_dir)}")
    return output_dir


# ================================================================== #
#  Default prompts
# ================================================================== #

DEFAULT_PROMPTS = [
    "a photo of a cat",
    "a cute puppy",
    "a photo of a bird",
    "a green frog",
    "a brown horse",
    "a deer in nature",
    "a fluffy dog",
    "a small kitten",
    "a colorful bird",
    "a horse running",
    "a photo of a frog",
    "a photo of a deer",
    "a photo of an airplane",
    "a red car",
    "a ship on water",
    "a truck",
]


# ================================================================== #
#  Main pipeline
# ================================================================== #

def run_pipeline(
    steps_vae=5000, steps_unet=15000,
    batch_vae=16, batch_unet=8,
    img_size=64, max_samples=0,
    animals_only=False,
    save_path="train/own_model_v4.pt",
    output_dir="output/own_model_v4",
    gen_prompts=None, cfg_scale=3.0,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    classes = "animals (6)" if animals_only else "all (10)"

    print(f"\n{'='*60}")
    print(f"  v4 — Train on REAL Photos (CIFAR-10)")
    print(f"{'='*60}")
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**2
        print(f"  GPU: {name} ({vram:.0f} MB)")
    print(f"  Image: {img_size}×{img_size}")
    print(f"  Classes: {classes}")
    print(f"  VAE: {steps_vae} steps (batch={batch_vae})")
    print(f"  UNet: {steps_unet} steps (batch={batch_unet})")
    print(f"  CFG: {cfg_scale}")
    print(f"{'='*60}\n")

    t_total = time.time()

    # Dataset
    dataset = CIFAR10Dataset(
        root="data/cifar10", img_size=img_size,
        animals_only=animals_only, max_samples=max_samples,
    )

    # Models
    latent_ch = 8   # More latent channels = more info for decoder
    context_dim = 256

    vae = GenVAE(3, latent_ch, base_ch=64).to(device)
    disc = PatchDiscriminator(3, 64).to(device)
    cond = ClassConditioner(num_classes=10, context_dim=context_dim,
                            num_tokens=4).to(device)

    # Phase 1: VAE
    vae = train_vae(vae, disc, dataset, device,
                    steps=steps_vae, lr=3e-4, batch_size=batch_vae)
    del disc
    _flush()

    # UNet — compact size for convergence
    latent_size = img_size // 4  # 16 for 64px images
    unet = MiniUNet(
        in_channels=latent_ch, out_channels=latent_ch,
        context_dim=context_dim,
        base_ch=96, ch_mult=(1, 2, 4), time_dim=128,
    ).to(device)

    # Phase 2: UNet
    unet, cond, scheduler = train_unet(
        unet, vae, cond, dataset, device,
        steps=steps_unet, lr=2e-4, batch_size=batch_unet,
    )

    # Save
    total = (sum(p.numel() for p in unet.parameters()) +
             sum(p.numel() for p in vae.parameters()) +
             sum(p.numel() for p in cond.parameters()))

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    torch.save({
        "unet": unet.state_dict(),
        "vae": vae.state_dict(),
        "cond": cond.state_dict(),
        "config": {
            "version": 4,
            "img_size": img_size,
            "latent_ch": latent_ch,
            "latent_size": latent_size,
            "context_dim": context_dim,
            "base_ch_unet": 96,
            "base_ch_vae": 64,
            "ch_mult": [1, 2, 4],
            "time_dim": 128,
            "num_classes": 10,
            "num_tokens": 4,
            "total_params": total,
            "animals_only": animals_only,
            "dataset": "CIFAR-10",
        },
    }, save_path)

    elapsed = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"  Saved: {os.path.abspath(save_path)}")
    print(f"  Size: {os.path.getsize(save_path)/1024/1024:.1f} MB")
    print(f"  Params: {total:,} ({total/1e6:.1f}M)")
    print(f"  Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"{'='*60}\n")

    # Generate
    prompts = gen_prompts or DEFAULT_PROMPTS
    latent_shape = (1, latent_ch, latent_size, latent_size)
    generate(unet, vae, cond, scheduler, prompts, device,
             latent_shape, 50, cfg_scale, output_dir)

    return save_path


def load_and_generate(path, prompts, output_dir="output/own_model_v4",
                      num_steps=50, cfg_scale=3.0):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Load] {path} ...")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]

    vae = GenVAE(3, cfg["latent_ch"], cfg.get("base_ch_vae", 64)).to(device)
    vae.load_state_dict(ckpt["vae"])
    vae.eval()

    cond = ClassConditioner(
        cfg.get("num_classes", 10), cfg["context_dim"],
        cfg.get("num_tokens", 4),
    ).to(device)
    cond.load_state_dict(ckpt["cond"])
    cond.eval()

    unet = MiniUNet(
        cfg["latent_ch"], cfg["latent_ch"], cfg["context_dim"],
        cfg.get("base_ch_unet", 96),
        tuple(cfg.get("ch_mult", [1, 2, 4])),
        cfg.get("time_dim", 128),
    ).to(device)
    unet.load_state_dict(ckpt["unet"])
    unet.eval()

    scheduler = SimpleScheduler(num_timesteps=1000)
    print(f"[Load] {cfg['total_params']:,} params ({cfg['total_params']/1e6:.1f}M)")

    latent_shape = (1, cfg["latent_ch"], cfg["latent_size"], cfg["latent_size"])
    generate(unet, vae, cond, scheduler, prompts, device,
             latent_shape, num_steps, cfg_scale, output_dir)


# ================================================================== #
#  CLI
# ================================================================== #

def main():
    p = argparse.ArgumentParser(description="Train on real CIFAR-10 photos")
    p.add_argument("--steps", type=int, default=15000)
    p.add_argument("--steps-vae", type=int, default=5000)
    p.add_argument("--batch-vae", type=int, default=16)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--img-size", type=int, default=64)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--animals-only", action="store_true")
    p.add_argument("--cfg-scale", type=float, default=3.0)
    p.add_argument("--prompt", type=str, default=None)
    p.add_argument("--load", type=str, default=None)
    p.add_argument("--save-path", type=str, default="train/own_model_v4.pt")
    p.add_argument("--output-dir", type=str, default="output/own_model_v4")
    args = p.parse_args()

    if args.load:
        prompts = [args.prompt] if args.prompt else DEFAULT_PROMPTS[:8]
        load_and_generate(args.load, prompts, args.output_dir,
                          cfg_scale=args.cfg_scale)
    else:
        gen = [args.prompt] if args.prompt else None
        run_pipeline(
            steps_vae=args.steps_vae, steps_unet=args.steps,
            batch_vae=args.batch_vae, batch_unet=args.batch,
            img_size=args.img_size, max_samples=args.max_samples,
            animals_only=args.animals_only,
            save_path=args.save_path, output_dir=args.output_dir,
            gen_prompts=gen, cfg_scale=args.cfg_scale,
        )


if __name__ == "__main__":
    main()
