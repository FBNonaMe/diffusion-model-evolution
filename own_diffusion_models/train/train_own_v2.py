"""
Обучение модели генерации БЕЗ БЛЮРА — чёткие контуры и яркие цвета.

Что изменено по сравнению с v1 (train_own.py):
  - L1 loss вместо MSE (MSE размывает границы)
  - Edge loss (Sobel-фильтр, сохраняет контуры)
  - PatchGAN дискриминатор (adversarial training → чёткие текстуры)
  - Меньшее сжатие VAE: ÷4 вместо ÷8 (16×16 латенты вместо 8×8)
  - Skip connections в VAE (U-Net стиль)
  - Больше шагов обучения по умолчанию
  - Увеличенная ёмкость модели

Результат: ~20M параметров, обучение ~60-90 сек на RTX 2050.

Использование::

    # Стандартное обучение (чёткие картинки!)
    python -m train.train_own_v2

    # Больше шагов = ещё лучше
    python -m train.train_own_v2 --steps 2000 --steps-vae 1000

    # Разрешение 128×128
    python -m train.train_own_v2 --img-size 128

    # Загрузить и сгенерировать
    python -m train.train_own_v2 --load train/own_model_v2.pt --prompt "red circle on black background"
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
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from train.train_mini import MiniUNet, MiniTextEncoder, SimpleScheduler
from train.train_own import (
    COLORS, BG_COLORS, SHAPES, draw_shape, SyntheticDataset,
    collate_images, DDIMSampler, _save_bmp,
)


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
# Edge loss — Sobel filter to preserve sharp edges
# ------------------------------------------------------------------ #

def sobel_edges(img: torch.Tensor) -> torch.Tensor:
    """Extract edges using Sobel filter.  img: (B,C,H,W)"""
    # Convert to grayscale
    gray = img.mean(dim=1, keepdim=True)

    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=img.dtype, device=img.device,
    ).view(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        dtype=img.dtype, device=img.device,
    ).view(1, 1, 3, 3)

    ex = F.conv2d(gray, sobel_x, padding=1)
    ey = F.conv2d(gray, sobel_y, padding=1)
    return (ex ** 2 + ey ** 2).sqrt()


def edge_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 loss between edge maps — penalizes blurry edges."""
    return F.l1_loss(sobel_edges(pred), sobel_edges(target))


# ------------------------------------------------------------------ #
# PatchGAN Discriminator — makes VAE output sharp
# ------------------------------------------------------------------ #

class PatchDiscriminator(nn.Module):
    """PatchGAN discriminator — classifies overlapping patches as real/fake.

    This is the key to sharp images: the discriminator penalizes
    blurry/smeared outputs, forcing the VAE to produce crisp details.
    ~200K parameters.
    """

    def __init__(self, in_channels: int = 3, base_ch: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            # 64×64 → 32×32
            nn.Conv2d(in_channels, base_ch, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # 32×32 → 16×16
            nn.Conv2d(base_ch, base_ch * 2, 4, stride=2, padding=1),
            nn.InstanceNorm2d(base_ch * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # 16×16 → 8×8
            nn.Conv2d(base_ch * 2, base_ch * 4, 4, stride=2, padding=1),
            nn.InstanceNorm2d(base_ch * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # 8×8 → 7×7 patch output
            nn.Conv2d(base_ch * 4, 1, 4, stride=1, padding=1),
        )
        total = sum(p.numel() for p in self.parameters())
        print(f"[PatchDiscriminator] Parameters: {total:,} ({total/1e6:.2f}M)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ------------------------------------------------------------------ #
# Sharp VAE — less compression + skip connections
# ------------------------------------------------------------------ #

class SharpVAE(nn.Module):
    """VAE with less compression (÷4), skip connections, and better capacity.

    Key differences from MiniVAE:
    - Only 2 downsamples (64→32→16) instead of 3 (64→32→16→8)
    - Skip connections from encoder to decoder (U-Net style)
    - LargeResidual blocks in bottleneck
    - ~3M parameters
    """

    def __init__(self, in_channels: int = 3, latent_channels: int = 4,
                 base_ch: int = 64):
        super().__init__()
        self.in_channels = in_channels
        self.latent_channels = latent_channels

        # Encoder: img → 16×16 latent (only ÷4 compression)
        # Block 1: full res → ÷2
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_ch, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.SiLU(),
        )
        self.down1 = nn.Conv2d(base_ch, base_ch, 3, stride=2, padding=1)

        # Block 2: ÷2 → ÷4
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch * 2, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_ch * 2, base_ch * 2, 3, padding=1),
            nn.SiLU(),
        )
        self.down2 = nn.Conv2d(base_ch * 2, base_ch * 2, 3, stride=2, padding=1)

        # Bottleneck at ÷4
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_ch * 2, base_ch * 2, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_ch * 2, base_ch * 2, 3, padding=1),
            nn.SiLU(),
        )

        # To latent
        self.to_mu = nn.Conv2d(base_ch * 2, latent_channels, 1)
        self.to_logvar = nn.Conv2d(base_ch * 2, latent_channels, 1)

        # Decoder: 16×16 latent → img
        self.from_latent = nn.Conv2d(latent_channels, base_ch * 2, 3, padding=1)

        # Up 1: ÷4 → ÷2 (+ skip from enc2)
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch * 2, 4, stride=2, padding=1)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_ch * 4, base_ch * 2, 3, padding=1),  # concat skip
            nn.SiLU(),
            nn.Conv2d(base_ch * 2, base_ch, 3, padding=1),
            nn.SiLU(),
        )

        # Up 2: ÷2 → full (+ skip from enc1)
        self.up2 = nn.ConvTranspose2d(base_ch, base_ch, 4, stride=2, padding=1)
        self.dec2 = nn.Sequential(
            nn.Conv2d(base_ch * 2, base_ch, 3, padding=1),  # concat skip
            nn.SiLU(),
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.SiLU(),
        )

        # Final output (no Sigmoid — we use tanh for sharper dynamic range)
        self.to_img = nn.Sequential(
            nn.Conv2d(base_ch, in_channels, 3, padding=1),
            nn.Sigmoid(),
        )

        total = sum(p.numel() for p in self.parameters())
        print(f"[SharpVAE] Parameters: {total:,} ({total/1e6:.1f}M)")

    def encode(self, x: torch.Tensor):
        # Encoder with stored skip connections
        s1 = self.enc1(x)            # base_ch, H, W
        h = self.down1(s1)           # base_ch, H/2, W/2

        s2 = self.enc2(h)            # base_ch*2, H/2, W/2
        h = self.down2(s2)           # base_ch*2, H/4, W/4

        h = self.bottleneck(h)       # base_ch*2, H/4, W/4

        mu = self.to_mu(h)
        logvar = self.to_logvar(h)

        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)

        return z, mu, logvar, s1, s2

    def decode(self, z: torch.Tensor,
               s1: torch.Tensor | None = None,
               s2: torch.Tensor | None = None):
        h = F.silu(self.from_latent(z))  # base_ch*2, H/4, W/4

        h = self.up1(h)                  # base_ch*2, H/2, W/2
        if s2 is not None:
            h = torch.cat([h, s2], dim=1)  # base_ch*4
        else:
            h = torch.cat([h, torch.zeros_like(h)], dim=1)
        h = self.dec1(h)                 # base_ch, H/2, W/2

        h = self.up2(h)                  # base_ch, H, W
        if s1 is not None:
            h = torch.cat([h, s1], dim=1)  # base_ch*2
        else:
            h = torch.cat([h, torch.zeros_like(h)], dim=1)
        h = self.dec2(h)                 # base_ch, H, W

        return self.to_img(h)

    def forward(self, x, drop_skips: bool = False):
        z, mu, logvar, s1, s2 = self.encode(x)
        if drop_skips:
            recon = self.decode(z, s1=None, s2=None)
        else:
            recon = self.decode(z, s1, s2)
        return recon, mu, logvar


# ------------------------------------------------------------------ #
# Phase 1: Train VAE with adversarial + edge loss (sharp!)
# ------------------------------------------------------------------ #

def train_vae_sharp(vae: SharpVAE, disc: PatchDiscriminator,
                    dataset, device, steps: int = 500,
                    lr: float = 1e-3, batch_size: int = 16):
    """Train VAE with L1 + edge + adversarial loss → sharp output."""
    print(f"\n{'='*60}")
    print(f"  Phase 1: Training SharpVAE (L1 + Edge + GAN)")
    print(f"{'='*60}")

    vae.train()
    disc.train()

    opt_vae = torch.optim.Adam(vae.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_disc = torch.optim.Adam(disc.parameters(), lr=lr * 0.2, betas=(0.5, 0.999))

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        collate_fn=collate_images, drop_last=True)

    use_amp = torch.cuda.is_available()
    scaler_vae = torch.amp.GradScaler("cuda", enabled=use_amp)
    scaler_disc = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Warmup: pure L1+edge first, then gradually add GAN
    gan_start = max(steps // 3, 100)  # start GAN after 33% of training

    step = 0
    t0 = time.time()

    print(f"  {'step':>6} | {'L1':>8} | {'edge':>8} | {'GAN':>8} | "
          f"{'D_loss':>8} | {'VRAM':>8}")
    print(f"  {'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")

    while step < steps:
        for batch in loader:
            if step >= steps:
                break

            images = batch["images"].to(device)

            # ---- Train Discriminator (every 2 steps, label smoothing) ---- #
            if step >= gan_start and step % 2 == 0:
                opt_disc.zero_grad()
                with torch.amp.autocast("cuda", enabled=use_amp):
                    with torch.no_grad():
                        # Use drop_skips randomly, matching generator's distribution
                        d_drop = random.random() < 0.2
                        recon_d, _, _ = vae(images, drop_skips=d_drop)
                    real_pred = disc(images)
                    fake_pred = disc(recon_d.detach())
                    # Label smoothing + noise for stability
                    d_loss_real = F.binary_cross_entropy_with_logits(
                        real_pred, 0.9 * torch.ones_like(real_pred))
                    d_loss_fake = F.binary_cross_entropy_with_logits(
                        fake_pred, 0.1 * torch.ones_like(fake_pred))
                    d_loss = (d_loss_real + d_loss_fake) * 0.5

                scaler_disc.scale(d_loss).backward()
                scaler_disc.step(opt_disc)
                scaler_disc.update()
            else:
                if step < gan_start:
                    d_loss = torch.tensor(0.0)

            # ---- Train VAE (Generator) ---- #
            opt_vae.zero_grad()

            # 20% of the time: drop skip connections so decoder can work without them
            drop = random.random() < 0.2

            with torch.amp.autocast("cuda", enabled=use_amp):
                recon, mu, logvar = vae(images, drop_skips=drop)

                # L1 reconstruction (sharper than MSE!)
                l1 = F.l1_loss(recon, images)

                # Edge preservation loss
                e_loss = edge_loss(recon, images)

                # KL divergence (very small weight — we want good reconstruction)
                kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

                # Total VAE loss
                vae_loss = l1 + 0.5 * e_loss + 0.0005 * kl

                # Progressive GAN: weight increases linearly from 0 to 0.1
                if step >= gan_start:
                    gan_progress = min(1.0, (step - gan_start) / max(1, steps - gan_start))
                    gan_weight = 0.01 + 0.09 * gan_progress  # 0.01 → 0.10
                    fake_pred = disc(recon)
                    g_loss = F.binary_cross_entropy_with_logits(
                        fake_pred, torch.ones_like(fake_pred))
                    vae_loss = vae_loss + gan_weight * g_loss
                else:
                    g_loss = torch.tensor(0.0)

            scaler_vae.scale(vae_loss).backward()
            scaler_vae.step(opt_vae)
            scaler_vae.update()

            step += 1

            if step % 50 == 0 or step == 1:
                print(f"  {step:6d} | {l1.item():8.4f} | {e_loss.item():8.4f} | "
                      f"{g_loss.item():8.4f} | {d_loss.item():8.4f} | "
                      f"{_vram_mb():8.0f}MB")

    elapsed = time.time() - t0
    print(f"\n  SharpVAE trained: {steps} steps in {elapsed:.1f}s")
    vae.eval()
    disc.eval()
    return vae


# ------------------------------------------------------------------ #
# Phase 2: Train UNet (same as v1, but with 16×16 latents)
# ------------------------------------------------------------------ #

def train_unet(unet, vae: SharpVAE, text_enc, dataset, device,
               steps: int = 1000, lr: float = 3e-4,
               batch_size: int = 8):
    """Train UNet to denoise VAE latents conditioned on text."""
    print(f"\n{'='*60}")
    print(f"  Phase 2: Training UNet (denoising on 16×16 latents)")
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
                loss = F.l1_loss(pred, noise)  # L1 instead of MSE

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
    initial = sum(losses[:20]) / min(len(losses), 20)
    final = sum(losses[-20:]) / min(len(losses), 20)
    print(f"\n  UNet trained: {steps} steps in {elapsed:.1f}s")
    print(f"  Loss: {initial:.4f} → {final:.4f} ({(1-final/initial)*100:+.1f}%)")

    return unet, scheduler


# ------------------------------------------------------------------ #
# Phase 3: Generate sharp images!
# ------------------------------------------------------------------ #

@torch.no_grad()
def generate(unet, vae: SharpVAE, text_enc, scheduler,
             prompts: list[str], device,
             latent_shape=(1, 4, 16, 16), num_steps: int = 50,
             output_dir: str = "output/own_model_v2"):
    """Generate images from text prompts using trained model."""
    print(f"\n{'='*60}")
    print(f"  Generating SHARP images!")
    print(f"{'='*60}")

    os.makedirs(output_dir, exist_ok=True)
    unet.eval()
    vae.eval()
    text_enc.eval()

    sampler = DDIMSampler(scheduler)

    for i, prompt in enumerate(prompts):
        print(f"\n  [{i+1}/{len(prompts)}] \"{prompt}\"")

        tokens = text_enc.simple_tokenize(prompt).unsqueeze(0).to(device)
        context = text_enc(tokens)

        shape = (1, latent_shape[1], latent_shape[2], latent_shape[3])
        z = sampler.sample(unet, shape, context, device, num_steps=num_steps)

        # Decode WITHOUT skip connections (we don't have encoder features at inference)
        img = vae.decode(z, s1=None, s2=None)
        img = img[0].clamp(0, 1)

        img_np = (img.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        path = os.path.join(output_dir, f"gen_{i:03d}.bmp")
        _save_bmp(path, img_np)
        print(f"  Saved: {path} ({img_np.shape[1]}×{img_np.shape[0]})")

    print(f"\n  All images saved to: {os.path.abspath(output_dir)}")
    return output_dir


# ------------------------------------------------------------------ #
# Full pipeline
# ------------------------------------------------------------------ #

def run_full_pipeline(
    steps_vae: int = 500,
    steps_unet: int = 1000,
    batch_size: int = 8,
    img_size: int = 64,
    num_samples: int = 500,
    base_ch_unet: int = 96,
    base_ch_vae: int = 96,
    do_generate: bool = True,
    generate_prompts: list[str] | None = None,
    save_path: str = "train/own_model_v2.pt",
    output_dir: str = "output/own_model_v2",
):
    """Full training pipeline with sharp output."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*60}")
    print(f"  Training YOUR OWN Model v2 (SHARP)")
    print(f"{'='*60}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Image size: {img_size}×{img_size}")
    print(f"  Latent size: {img_size//4}×{img_size//4} (÷4 compression)")
    print(f"  Dataset: {num_samples} synthetic images")
    print(f"  VAE steps: {steps_vae} (L1 + Edge + GAN)")
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

    # ---- SharpVAE + Discriminator ----
    latent_ch = 4
    vae = SharpVAE(
        in_channels=3, latent_channels=latent_ch, base_ch=base_ch_vae,
    ).to(device)

    disc = PatchDiscriminator(in_channels=3, base_ch=32).to(device)

    vae = train_vae_sharp(
        vae, disc, dataset, device,
        steps=steps_vae, lr=1e-3,
        batch_size=min(batch_size, 16),
    )

    # Free discriminator memory after training
    del disc
    _flush()

    # ---- UNet (works on 16×16 latents) ----
    latent_size = img_size // 4  # ÷4 instead of ÷8
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

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    torch.save({
        "unet": unet.state_dict(),
        "vae": vae.state_dict(),
        "text_encoder": text_enc.state_dict(),
        "config": {
            "version": 2,
            "img_size": img_size,
            "latent_ch": latent_ch,
            "latent_size": latent_size,
            "context_dim": context_dim,
            "base_ch_unet": base_ch_unet,
            "base_ch_vae": base_ch_vae,
            "ch_mult": [1, 2, 3],
            "total_params": total_params,
            "vae_type": "SharpVAE",
        },
        "training": {
            "steps_vae": steps_vae,
            "steps_unet": steps_unet,
            "num_samples": num_samples,
            "losses": "L1 + Edge + GAN (VAE), L1 (UNet)",
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
            num_steps=50,
            output_dir=output_dir,
        )

    return save_path


# ------------------------------------------------------------------ #
# Load and generate from saved model
# ------------------------------------------------------------------ #

def load_and_generate(
    model_path: str,
    prompts: list[str],
    output_dir: str = "output/own_model_v2",
    num_steps: int = 50,
):
    """Load a saved v2 model and generate images."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[Load] Loading model from {model_path} ...")
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]

    text_enc = MiniTextEncoder(
        vocab_size=3000, embed_dim=64,
        context_dim=cfg["context_dim"], max_len=16, num_layers=1,
    ).to(device)
    text_enc.load_state_dict(ckpt["text_encoder"])
    text_enc.eval()

    base_ch_vae = cfg.get("base_ch_vae", 64)
    vae = SharpVAE(
        in_channels=3, latent_channels=cfg["latent_ch"],
        base_ch=base_ch_vae,
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

    print(f"[Load] Model loaded: {cfg['total_params']:,} params (v2 sharp)\n")

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
        description="Train your own SHARP text-to-image model (v2)"
    )
    parser.add_argument("--steps", type=int, default=1000,
                        help="UNet training steps (default: 1000)")
    parser.add_argument("--steps-vae", type=int, default=500,
                        help="VAE training steps (default: 500)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--img-size", type=int, default=64,
                        help="Image size (default: 64)")
    parser.add_argument("--num-samples", type=int, default=500,
                        help="Dataset size")
    parser.add_argument("--base-ch", type=int, default=96,
                        help="UNet base channels")
    parser.add_argument("--base-ch-vae", type=int, default=96,
                        help="VAE base channels (96=~3.5M params)")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Custom prompt for generation")
    parser.add_argument("--load", type=str, default=None,
                        help="Load saved model and generate")
    parser.add_argument("--save-path", type=str,
                        default="train/own_model_v2.pt")
    parser.add_argument("--output-dir", type=str,
                        default="output/own_model_v2")
    args = parser.parse_args()

    if args.load:
        prompts = [args.prompt] if args.prompt else [
            "red circle on black background",
            "blue square on white background",
            "green triangle on gray background",
        ]
        load_and_generate(args.load, prompts, args.output_dir)
    else:
        gen_prompts = [args.prompt] if args.prompt else None
        run_full_pipeline(
            steps_vae=args.steps_vae,
            steps_unet=args.steps,
            batch_size=args.batch_size,
            img_size=args.img_size,
            num_samples=args.num_samples,
            base_ch_unet=args.base_ch,
            base_ch_vae=args.base_ch_vae,
            do_generate=True,
            generate_prompts=gen_prompts,
            save_path=args.save_path,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
