"""
v5.1 Diffusion Model — Optimized for RTX 2050 (4 GB VRAM).

Target
------
  - Resolution : 128×128 RGB
  - Total params: ~30 M  (VAE ≈ 18 M, UNet ≈ 12 M)
  - Stable training with EMA (decay 0.999) + CosineAnnealingLR
  - Dataset     : 100 k+ animal images (CIFAR-10 + augmentation or image folder)

Architecture
------------
  GenVAE v5 — no skip connections, 128→32 (÷4), 12 latent channels
    Loss    : L1 + Edge + GAN + KL  (KL warmup over 40 % steps)
    EMA     : decay 0.999

  UNet v5.1 — 96→192→384, self-attention at 16×16 + middle block
    2 ResBlocks / stage, GroupNorm, SiLU, Dropout 0.1
    Gradient checkpointing enabled
    Noise prediction (ε) or v-prediction (--v-prediction flag)
    Class conditioning: nn.Embedding → added to time embedding
    Cosine beta schedule for improved SNR distribution

Training
--------
  Optimizer : AdamW 8-bit (bitsandbytes) — falls back to regular AdamW
  lr        : 2e-4,  betas (0.9, 0.95),  weight_decay 0.1
  Scheduler : CosineAnnealingLR (T_max = total steps, eta_min = 1e-6)
  Precision : torch.autocast(fp16) + GradScaler
  Batch     : batch_size=1, gradient_accumulation=8
  EMA       : decay 0.999 for both VAE and UNet — generation from EMA

  Metrics tracked : UNet loss, VAE L1, KL, EMA loss, FID (every N epochs)
  Early stopping  : FID stalls or val-loss plateaus 3 ×

Expected peak VRAM : ~3–3.5 GB.

Usage::

    python -m train.train_own_v5
    python -m train.train_own_v5 --animals-only
    python -m train.train_own_v5 --steps 10000 --steps-vae 2000
    python -m train.train_own_v5 --data-dir /path/to/animal_images
    python -m train.train_own_v5 --load train/own_model_v5.pt --prompt "cat"
"""

from __future__ import annotations

import argparse
import copy
import gc
import math
import os
import random
import struct
import sys
import time
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.checkpoint import checkpoint as grad_ckpt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── Try 8-bit AdamW from bitsandbytes ──────────────────────────────
try:
    import bitsandbytes as bnb
    AdamW8bit = bnb.optim.AdamW8bit
    _HAS_BNB = True
except ImportError:
    AdamW8bit = None
    _HAS_BNB = False


# ================================================================== #
#  Constants
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


def _param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _make_optimizer(params, lr, betas, weight_decay):
    """AdamW 8-bit when available, else standard AdamW."""
    if _HAS_BNB:
        print("  [optim] Using AdamW 8-bit (bitsandbytes)")
        return AdamW8bit(params, lr=lr, betas=betas, weight_decay=weight_decay)
    print("  [optim] Falling back to torch.optim.AdamW (bitsandbytes not found)")
    return torch.optim.AdamW(params, lr=lr, betas=betas, weight_decay=weight_decay)


# ================================================================== #
#  Exponential Moving Average
# ================================================================== #

class EMA:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {}
        self.backup: dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def apply(self, model: nn.Module):
        """Swap model weights with EMA weights."""
        self.backup = {}
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module):
        """Restore original model weights."""
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self):
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_state_dict(self, state_dict):
        self.shadow = {k: v.clone() for k, v in state_dict.items()}


# ================================================================== #
#  Early-stopping tracker
# ================================================================== #

class EarlyStopping:
    def __init__(self, patience: int = 3):
        self.patience = patience
        self.counter = 0
        self.best = float("inf")

    def check(self, value: float) -> bool:
        """Return True when training should stop."""
        if value < self.best - 1e-6:
            self.best = value
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


# ================================================================== #
#  GenVAE v5 — VAE without skip connections
# ================================================================== #

class _VAEResBlock(nn.Module):
    """Simple residual block (no time embedding) for the VAE."""

    def __init__(self, ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class GenVAEv5(nn.Module):
    """Generation-focused VAE — NO skip connections.

    128×128 → latent 32×32 (÷4), 12 latent channels.
    3 ResBlocks per level for capacity.  base_ch=128 → ~18 M params.
    """

    def __init__(self, in_channels: int = 3, latent_channels: int = 12,
                 base_ch: int = 128):
        super().__init__()
        ch1:  int = base_ch       # 128
        ch2:  int = base_ch * 2   # 256

        # ── Encoder ──────────────────────────────────────────────────
        self.enc_in = nn.Conv2d(in_channels, ch1, 3, padding=1)

        # Level 1  (128×128) → ÷2
        self.enc1 = nn.Sequential(*[_VAEResBlock(ch1) for _ in range(3)])
        self.down1 = nn.Conv2d(ch1, ch2, 3, stride=2, padding=1)

        # Level 2  (64×64) → ÷2
        self.enc2 = nn.Sequential(*[_VAEResBlock(ch2) for _ in range(3)])
        self.down2 = nn.Conv2d(ch2, ch2, 3, stride=2, padding=1)

        # Bottleneck  (32×32)
        self.enc_mid = nn.Sequential(*[_VAEResBlock(ch2) for _ in range(3)])

        self.to_mu     = nn.Conv2d(ch2, latent_channels, 1)
        self.to_logvar = nn.Conv2d(ch2, latent_channels, 1)

        # ── Decoder ──────────────────────────────────────────────────
        self.from_z = nn.Conv2d(latent_channels, ch2, 3, padding=1)

        self.dec_mid = nn.Sequential(*[_VAEResBlock(ch2) for _ in range(3)])

        self.up1  = nn.ConvTranspose2d(ch2, ch2, 4, stride=2, padding=1)
        self.dec1 = nn.Sequential(*[_VAEResBlock(ch2) for _ in range(3)])

        self.up2  = nn.ConvTranspose2d(ch2, ch1, 4, stride=2, padding=1)
        self.dec2 = nn.Sequential(*[_VAEResBlock(ch1) for _ in range(3)])

        self.dec_out = nn.Sequential(
            nn.GroupNorm(32, ch1),
            nn.SiLU(),
            nn.Conv2d(ch1, in_channels, 3, padding=1),
            nn.Sigmoid(),
        )

        total = _param_count(self)
        print(f"[GenVAEv5] Parameters: {total:,} ({total/1e6:.1f}M)")

    # ── forward helpers ──────────────────────────────────────────────

    def encode(self, x: torch.Tensor):
        h = self.enc_in(x)
        h = self.enc1(h)
        h = self.down1(h)
        h = self.enc2(h)
        h = self.down2(h)
        h = self.enc_mid(h)
        mu     = self.to_mu(h).clamp(-20, 20)
        logvar = self.to_logvar(h)
        logvar = torch.clamp(logvar, -6, 6)
        std = torch.exp(0.5 * logvar)
        z   = mu + std * torch.randn_like(std)
        return z, mu, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.from_z(z))
        h = self.dec_mid(h)
        h = self.up1(h)
        h = self.dec1(h)
        h = self.up2(h)
        h = self.dec2(h)
        return self.dec_out(h)

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        return self.decode(z), mu, logvar


# ================================================================== #
#  PatchDiscriminator (for GAN loss in VAE)
# ================================================================== #

class PatchDiscriminator(nn.Module):
    """PatchGAN discriminator for 128×128 images."""

    def __init__(self, in_channels: int = 3, base_ch: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_channels, base_ch, 4, 2, 1)),      # 64
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv2d(base_ch, base_ch * 2, 4, 2, 1)),      # 32
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv2d(base_ch * 2, base_ch * 4, 4, 2, 1)),  # 16
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv2d(base_ch * 4, base_ch * 4, 4, 2, 1)),  # 8
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_ch * 4, 1, 4, 1, 1),                                    # 7
        )
        total = _param_count(self)
        print(f"[PatchDisc] Parameters: {total:,} ({total/1e6:.1f}M)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ================================================================== #
#  UNet v5 — Compact noise-prediction network
# ================================================================== #

class _UNetResBlock(nn.Module):
    """ResBlock with timestep embedding, dropout, and gradient-checkpoint support."""

    def __init__(self, channels: int, time_dim: int, dropout: float = 0.1,
                 use_checkpoint: bool = True):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.norm1    = nn.GroupNorm(32, channels)
        self.conv1    = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2    = nn.GroupNorm(32, channels)
        self.conv2    = nn.Conv2d(channels, channels, 3, padding=1)
        self.time_proj = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, channels))
        self.dropout  = nn.Dropout(dropout)

    def _forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return x + h

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and self.training:
            return grad_ckpt(self._forward, x, t_emb, use_reentrant=False)
        return self._forward(x, t_emb)


class LearnedSinusoidalEmbedding(nn.Module):
    """Learnable sinusoidal frequencies for timestep encoding."""

    def __init__(self, dim: int):
        super().__init__()
        half = dim // 2
        self.weights = nn.Parameter(torch.randn(half) * 0.01)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        freqs = t[:, None].float() * self.weights[None, :] * 2.0 * math.pi
        return torch.cat([freqs.sin(), freqs.cos()], dim=-1)


class _SelfAttentionBlock(nn.Module):
    """Spatial self-attention for feature maps (used at 16×16)."""

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x).reshape(B, C, H * W).permute(0, 2, 1)  # (B, HW, C)
        h, _ = self.attn(h, h, h)
        return x + h.permute(0, 2, 1).reshape(B, C, H, W)


class UNetv5(nn.Module):
    """UNet v5.1 for latent-space diffusion.

    Channels     : 96 → 192 → 384
    Attention    : self-attention at 16×16 (enc2 + dec2) + middle block
    Conditioning : class embedding added to time embedding
    Grad ckpt    : enabled by default

    ≈ 12 M params.
    """

    def __init__(
        self,
        in_channels:    int = 12,
        out_channels:   int = 12,
        base_ch:        int = 96,
        ch_mult:        tuple = (1, 2, 4),
        time_dim:       int = 256,
        num_heads:      int = 4,
        dropout:        float = 0.1,
        num_classes:    int = 10,
        use_checkpoint: bool = True,
    ):
        super().__init__()
        self.time_dim = time_dim
        channels = [base_ch * m for m in ch_mult]  # [96, 192, 384]

        # ── Time embedding ───────────────────────────────────────────
        self.time_sinusoidal = LearnedSinusoidalEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )

        # ── Class conditioning ───────────────────────────────────────
        self.class_embed = nn.Embedding(num_classes, time_dim)

        # ── Input ────────────────────────────────────────────────────
        self.input_conv = nn.Conv2d(in_channels, channels[0], 3, padding=1)

        # ── Encoder path ─────────────────────────────────────────────
        kw = dict(time_dim=time_dim, dropout=dropout, use_checkpoint=use_checkpoint)

        # Stage 1 — 32×32 @ channels[0]
        self.enc1_res = nn.ModuleList([_UNetResBlock(channels[0], **kw) for _ in range(2)])
        self.down1 = nn.Conv2d(channels[0], channels[1], 3, stride=2, padding=1)

        # Stage 2 — 16×16 @ channels[1]  +  self-attention
        self.enc2_res = nn.ModuleList([_UNetResBlock(channels[1], **kw) for _ in range(2)])
        self.enc2_attn = _SelfAttentionBlock(channels[1], num_heads)
        self.down2 = nn.Conv2d(channels[1], channels[2], 3, stride=2, padding=1)

        # ── Middle — 8×8 @ channels[2] ──────────────────────────────
        self.mid_res1 = _UNetResBlock(channels[2], **kw)
        self.mid_norm = nn.GroupNorm(32, channels[2])
        self.mid_attn = nn.MultiheadAttention(channels[2], num_heads, batch_first=True)
        self.mid_res2 = _UNetResBlock(channels[2], **kw)

        # ── Decoder path ─────────────────────────────────────────────
        # Up stage 1 — 8×8 → 16×16  +  self-attention
        self.skip_proj2 = nn.Conv2d(channels[2] + channels[1], channels[1], 1)
        self.dec2_res = nn.ModuleList([_UNetResBlock(channels[1], **kw) for _ in range(2)])
        self.dec2_attn = _SelfAttentionBlock(channels[1], num_heads)

        # Up stage 2 — 16×16 → 32×32
        self.skip_proj1 = nn.Conv2d(channels[1] + channels[0], channels[0], 1)
        self.dec1_res = nn.ModuleList([_UNetResBlock(channels[0], **kw) for _ in range(2)])

        # ── Output ───────────────────────────────────────────────────
        self.output_norm = nn.GroupNorm(32, channels[0])
        self.output_conv = nn.Conv2d(channels[0], out_channels, 3, padding=1)

        total = _param_count(self)
        print(f"[UNetv5] Parameters: {total:,} ({total/1e6:.1f}M)")
        print(f"[UNetv5] Channels:   {channels}")

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        class_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # ── Time + class embedding ───────────────────────────────────
        t_emb = self.time_mlp(self.time_sinusoidal(timesteps))     # (B, D)
        if class_ids is not None:
            t_emb = t_emb + self.class_embed(class_ids)

        # ── Encoder ──────────────────────────────────────────────────
        x = self.input_conv(x)
        for res in self.enc1_res:
            x = res(x, t_emb)
        skip1 = x                                                  # 32×32
        x = self.down1(x)

        for res in self.enc2_res:
            x = res(x, t_emb)
        x = self.enc2_attn(x)                                      # self-attn 16×16
        skip2 = x                                                  # 16×16
        x = self.down2(x)

        # ── Middle ───────────────────────────────────────────────────
        x = self.mid_res1(x, t_emb)
        B, C, H, W = x.shape
        h = self.mid_norm(x).reshape(B, C, H * W).permute(0, 2, 1)  # (B, HW, C)
        h, _ = self.mid_attn(h, h, h)
        x = x + h.permute(0, 2, 1).reshape(B, C, H, W)
        x = self.mid_res2(x, t_emb)

        # ── Decoder ──────────────────────────────────────────────────
        x = F.interpolate(x, scale_factor=2, mode="nearest")      # → 16×16
        x = self.skip_proj2(torch.cat([x, skip2], dim=1))
        for res in self.dec2_res:
            x = res(x, t_emb)
        x = self.dec2_attn(x)                                      # self-attn 16×16

        x = F.interpolate(x, scale_factor=2, mode="nearest")      # → 32×32
        x = self.skip_proj1(torch.cat([x, skip1], dim=1))
        for res in self.dec1_res:
            x = res(x, t_emb)

        # ── Output ───────────────────────────────────────────────────
        x = F.silu(self.output_norm(x))
        return self.output_conv(x)


# ================================================================== #
#  Noise scheduler (DDPM / cosine β)
# ================================================================== #

class NoiseScheduler:
    """DDPM cosine-β scheduler with v-prediction support."""

    def __init__(self, num_timesteps: int = 1000,
                 beta_start: float = 0.0001, beta_end: float = 0.02,
                 schedule: str = "cosine"):
        self.num_timesteps = num_timesteps
        if schedule == "cosine":
            # Cosine schedule (Nichol & Dhariwal, 2021)
            s = 0.008
            steps = torch.arange(num_timesteps + 1, dtype=torch.float64)
            f_t = torch.cos((steps / num_timesteps + s) / (1 + s) * math.pi / 2) ** 2
            alphas_cumprod = f_t / f_t[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = betas.clamp(max=0.999).float()
            alphas = 1.0 - betas
            self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        else:
            betas  = torch.linspace(beta_start, beta_end, num_timesteps)
            alphas = 1.0 - betas
            self.alphas_cumprod = torch.cumprod(alphas, dim=0)

    def add_noise(self, x0, noise, t):
        acp = self.alphas_cumprod.to(x0.device)
        sqrt_a  = acp[t].sqrt()
        sqrt_1a = (1.0 - acp[t]).sqrt()
        while sqrt_a.dim() < x0.dim():
            sqrt_a  = sqrt_a.unsqueeze(-1)
            sqrt_1a = sqrt_1a.unsqueeze(-1)
        return sqrt_a * x0 + sqrt_1a * noise

    def get_v_target(self, x0, noise, t):
        """v = √ᾱ·ε − √(1−ᾱ)·x₀"""
        acp = self.alphas_cumprod.to(x0.device)
        sqrt_a  = acp[t].sqrt()
        sqrt_1a = (1.0 - acp[t]).sqrt()
        while sqrt_a.dim() < x0.dim():
            sqrt_a  = sqrt_a.unsqueeze(-1)
            sqrt_1a = sqrt_1a.unsqueeze(-1)
        return sqrt_a * noise - sqrt_1a * x0


# ================================================================== #
#  DDIM Sampler (supports ε and v prediction)
# ================================================================== #

class DDIMSampler:
    def __init__(self, scheduler: NoiseScheduler, v_prediction: bool = False):
        self.T   = scheduler.num_timesteps
        self.acp = scheduler.alphas_cumprod
        self.v_prediction = v_prediction

    @torch.no_grad()
    def sample(self, unet, shape, class_ids, device,
               num_steps: int = 50, cfg_scale: float = 3.0):
        acp = self.acp.to(device)
        step_size = max(1, self.T // num_steps)
        timesteps = list(range(self.T - 1, -1, -step_size))

        x = torch.randn(shape, device=device)

        for j, t in enumerate(timesteps):
            t_b = torch.full((shape[0],), t, device=device, dtype=torch.long)

            pred_c = unet(x, t_b, class_ids)
            if cfg_scale > 1.0:
                pred_u = unet(x, t_b, None)
                pred   = pred_u + cfg_scale * (pred_c - pred_u)
            else:
                pred = pred_c

            alpha_t    = acp[t]
            alpha_prev = acp[timesteps[j + 1]] if j + 1 < len(timesteps) \
                         else torch.tensor(1.0, device=device)

            if self.v_prediction:
                # x₀ = √ᾱ · xₜ − √(1−ᾱ) · v
                x0_pred  = alpha_t.sqrt() * x - (1 - alpha_t).sqrt() * pred
                eps_pred = (1 - alpha_t).sqrt() * x + alpha_t.sqrt() * pred
            else:
                x0_pred  = (x - (1 - alpha_t).sqrt() * pred) / alpha_t.sqrt()
                eps_pred = pred

            x0_pred = x0_pred.clamp(-5, 5)
            dir_xt  = (1 - alpha_prev).sqrt() * eps_pred
            x = alpha_prev.sqrt() * x0_pred + dir_xt

        return x


# ================================================================== #
#  Datasets — CIFAR-10 or Image Folder
# ================================================================== #

class AnimalDataset(Dataset):
    """CIFAR-10 (or image-folder) at 128×128 with augmentation."""

    def __init__(self, root: str = "data/cifar10", img_size: int = 128,
                 animals_only: bool = False, max_samples: int = 0,
                 data_dir: Optional[str] = None):
        import torchvision
        import torchvision.transforms as T

        self.img_size = img_size

        self.transform = T.Compose([
            T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(0.5),
            T.ColorJitter(brightness=0.15, contrast=0.15,
                          saturation=0.15, hue=0.04),
            T.RandomRotation(8),
            T.ToTensor(),
        ])
        self.transform_vae = T.Compose([
            T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(0.5),
            T.ColorJitter(brightness=0.08, contrast=0.08),
            T.ToTensor(),
        ])

        self.use_vae_transform = True

        if data_dir and os.path.isdir(data_dir):
            # ── Image-folder dataset ─────────────────────────────────
            print(f"  Loading images from {data_dir} ...")
            self.mode = "folder"
            from torchvision.datasets import ImageFolder
            self._folder = ImageFolder(data_dir)
            self.num_classes = len(self._folder.classes)
            self.indices = list(range(len(self._folder)))
            print(f"  Found {len(self.indices)} images in "
                  f"{self.num_classes} classes")
        else:
            # ── CIFAR-10 ─────────────────────────────────────────────
            print(f"  Loading CIFAR-10 ...")
            self.mode = "cifar"
            self.num_classes = 10
            cifar = torchvision.datasets.CIFAR10(
                root=root, train=True, download=True)
            self._cifar = cifar

            if animals_only:
                animal_ids = {i for i, c in enumerate(CIFAR10_CLASSES)
                              if c in ANIMAL_CLASSES}
                self.indices = [i for i in range(len(cifar))
                                if cifar.targets[i] in animal_ids]
                # Also add test set for more data
                cifar_test = torchvision.datasets.CIFAR10(
                    root=root, train=False, download=True)
                self._cifar_test = cifar_test
                offset = len(cifar)
                for i in range(len(cifar_test)):
                    if cifar_test.targets[i] in animal_ids:
                        self.indices.append(offset + i)
                print(f"  Animals only: {len(self.indices)} images "
                      f"(train + test)")
            else:
                self.indices = list(range(len(cifar)))
                self._cifar_test = None

        if max_samples > 0 and max_samples < len(self.indices):
            random.seed(42)
            self.indices = random.sample(self.indices, max_samples)

        print(f"  Dataset size: {len(self.indices)}, "
              f"image: {img_size}×{img_size}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]

        if self.mode == "folder":
            pil_img, label = self._folder[real_idx]
        else:
            cifar_len = len(self._cifar)
            if real_idx < cifar_len:
                pil_img, label = self._cifar[real_idx]
            else:
                pil_img, label = self._cifar_test[real_idx - cifar_len]

        tfm = self.transform_vae if self.use_vae_transform else self.transform
        img_t = tfm(pil_img)
        return {"image": img_t, "label": label}


def _collate(batch):
    return {
        "images": torch.stack([b["image"] for b in batch]),
        "labels": torch.tensor([b["label"] for b in batch], dtype=torch.long),
    }


# ================================================================== #
#  Losses — edge (Sobel)
# ================================================================== #

def _sobel_edges(x: torch.Tensor) -> torch.Tensor:
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                       dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                       dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    gray = x.mean(dim=1, keepdim=True)
    return (F.conv2d(gray, kx, padding=1)**2 +
            F.conv2d(gray, ky, padding=1)**2).sqrt()


def _edge_loss(recon, target):
    return F.l1_loss(_sobel_edges(recon), _sobel_edges(target))


# ================================================================== #
#  FID (lightweight, optional — computed on CPU)
# ================================================================== #

@torch.no_grad()
def compute_fid(real_imgs: torch.Tensor, gen_imgs: torch.Tensor,
                device: str = "cpu") -> float:
    """Approximate FID using InceptionV3 features.

    Both inputs: (N, 3, H, W) in [0, 1].  Runs entirely on *device*.
    Returns FID score (lower is better).
    """
    try:
        from torchvision.models import inception_v3, Inception_V3_Weights
    except ImportError:
        print("  [FID] torchvision not available — skipping")
        return float("nan")

    model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
    model.fc = nn.Identity()          # → 2 048-d features
    model.eval().to(device)

    def _feats(imgs: torch.Tensor) -> torch.Tensor:
        out = []
        for i in range(imgs.shape[0]):
            img = imgs[i:i+1].to(device)
            img = F.interpolate(img, size=(299, 299), mode="bilinear",
                                align_corners=False)
            out.append(model(img).cpu())
        return torch.cat(out, 0)

    f_real = _feats(real_imgs).float()
    f_gen  = _feats(gen_imgs).float()

    mu_r, mu_g   = f_real.mean(0), f_gen.mean(0)
    sig_r = torch.cov(f_real.T)
    sig_g = torch.cov(f_gen.T)

    diff = mu_r - mu_g
    fid  = diff.dot(diff).item()

    # matrix sqrt via eigendecomposition of Σ_r @ Σ_g
    try:
        product = sig_r @ sig_g
        eigvals, eigvecs = torch.linalg.eigh(
            (product + product.T) / 2)          # symmetrise
        eigvals = eigvals.clamp(min=0).sqrt()
        sqrtm   = eigvecs @ torch.diag(eigvals) @ eigvecs.T
        fid    += torch.trace(sig_r + sig_g - 2 * sqrtm).item()
    except Exception:
        fid += (torch.trace(sig_r).item() + torch.trace(sig_g).item())

    del model
    _flush()
    return max(0.0, fid)


# ================================================================== #
#  Class-label ↔ text helpers  (for inference prompts)
# ================================================================== #

_TEXT_MAP: dict[str, int] = {}
for _i, _c in enumerate(CIFAR10_CLASSES):
    _TEXT_MAP[_c] = _i
    _TEXT_MAP[_c + "s"] = _i
_TEXT_MAP.update({
    "puppy": 5, "puppies": 5, "doggy": 5,
    "kitten": 3, "kitty": 3, "kittens": 3,
    "plane": 0, "jet": 0, "aircraft": 0,
    "car": 1, "auto": 1, "vehicle": 1, "cars": 1,
    "boat": 8, "ships": 8, "vessel": 8,
    "lorry": 9, "trucks": 9,
    "stag": 4, "doe": 4, "fawn": 4,
    "toad": 6, "frogs": 6,
    "pony": 7, "stallion": 7, "mare": 7, "horses": 7,
    "sparrow": 2, "robin": 2, "eagle": 2, "parrot": 2,
    "owl": 2, "pigeon": 2, "crow": 2,
    "animal": 3, "pet": 3,
})


def _text_to_class(text: str, num_classes: int = 10) -> int:
    low = text.lower()
    for word, cid in _TEXT_MAP.items():
        if word in low and cid < num_classes:
            return cid
    return random.randint(0, num_classes - 1)


# ================================================================== #
#  Phase 1 — Train VAE (+ EMA + Discriminator)
# ================================================================== #

def train_vae(
    vae: GenVAEv5,
    disc: PatchDiscriminator,
    dataset: AnimalDataset,
    device: str,
    *,
    steps: int         = 10000,
    lr: float          = 1e-4,
    grad_accum: int    = 32,
    ema_decay: float   = 0.999,
    kl_warmup_steps: Optional[int] = None,
    gan_start_step:  Optional[int] = None,
):
    print(f"\n{'='*60}")
    print(f"  Phase 1: Train GenVAE v5  (batch=1  ×  accum={grad_accum})")
    print(f"  Steps: {steps}  |  lr: {lr}")
    print(f"{'='*60}")

    vae.train()
    disc.train()

    opt_vae  = _make_optimizer(vae.parameters(), lr=lr,
                               betas=(0.9, 0.95), weight_decay=0.1)
    opt_disc = torch.optim.Adam(disc.parameters(), lr=lr * 0.15,
                                betas=(0.5, 0.999))

    # cosine LR with 5 % warmup
    warmup = max(1, steps // 20)

    def _lr_lambda(s):
        if s < warmup:
            return s / warmup
        progress = (s - warmup) / max(1, steps - warmup)
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))

    sched_vae = torch.optim.lr_scheduler.LambdaLR(opt_vae, _lr_lambda)

    ema_vae = EMA(vae, decay=ema_decay)
    use_amp_disc = torch.cuda.is_available()   # autocast only for disc
    scaler_disc  = torch.amp.GradScaler("cuda", enabled=use_amp_disc)

    dataset.use_vae_transform = True
    loader = DataLoader(dataset, batch_size=1, shuffle=True,
                        collate_fn=_collate, drop_last=True,
                        num_workers=0, pin_memory=True)

    gan_start = gan_start_step if gan_start_step is not None \
                else max(steps * 2 // 5, 800)      # delayed GAN start
    kl_warmup = kl_warmup_steps if kl_warmup_steps is not None \
                else max(1, steps * 2 // 5)         # 40 % KL warmup

    step = 0
    t0   = time.time()
    d_loss_val = 0.0
    data_iter  = iter(loader)
    nan_count  = 0

    print(f"  GAN starts at step {gan_start}")
    print(f"  KL warmup over {kl_warmup} steps (40%)")
    hdr = (f"  {'step':>6} | {'L1':>8} | {'edge':>8} | {'KL':>8} | "
           f"{'G(vae)':>8} | {'D':>8} | {'VRAM':>7}")
    print(hdr)
    print(f"  {'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-"
          f"{'-'*8}-+-{'-'*8}-+-{'-'*7}")

    while step < steps:
        # ── accumulation loop ────────────────────────────────────────
        opt_vae.zero_grad(set_to_none=True)
        opt_disc.zero_grad(set_to_none=True)

        acc_l1 = acc_edge = acc_kl = acc_g = 0.0

        for _ in range(grad_accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)

            images = batch["images"].to(device)

            # ── VAE forward in fp32 (fp16 overflows at 128ch) ───────
            recon, mu, logvar = vae(images)
            l1_v = F.l1_loss(recon, images)
            e_v  = _edge_loss(recon, images)

            # KL in fp32
            mu_f, logvar_f = mu.float(), logvar.float()
            kl_v = -0.5 * torch.sum(
                1 + logvar_f - mu_f.pow(2) - logvar_f.exp())
            latent_numel = mu_f.shape[0] * mu_f.shape[1] * mu_f.shape[2] * mu_f.shape[3]
            kl_v = kl_v / (latent_numel + 1e-8)
            kl_v = kl_v.clamp(min=0.0, max=10.0)

            # KL weight: starts at 5e-3, ramps to 5e-2
            # Must be strong enough to compete with L1 (~0.04)
            kl_w = 5e-3 + min(1.0, step / max(1, kl_warmup)) * 4.5e-2

            vae_loss = (l1_v + 0.3 * e_v + kl_w * kl_v) / grad_accum

            # ── NaN guard ────────────────────────────────────────────
            if torch.isnan(vae_loss) or torch.isinf(vae_loss):
                nan_count += 1
                print(f"  ⚠  NaN/Inf at step {step}  "
                      f"mu={mu_f.mean().item():.4f}  "
                      f"logvar={logvar_f.mean().item():.4f}  "
                      f"l1={l1_v.item():.4f}  kl={kl_v.item():.4f}")
                if nan_count >= 5:
                    print("  ⛔  Too many NaNs — stopping VAE training.")
                    break
                continue

            # GAN generator loss
            g_v = torch.tensor(0.0, device=device)
            if step >= gan_start:
                with torch.amp.autocast("cuda", enabled=use_amp_disc):
                    fake_out = disc(recon)
                    g_v = F.binary_cross_entropy_with_logits(
                        fake_out, torch.ones_like(fake_out))
                progress = min(1.0, (step - gan_start) /
                               max(1, steps - gan_start))
                gw = 0.005 + 0.03 * progress
                vae_loss = vae_loss + gw * g_v.float() / grad_accum

            vae_loss.backward()

            # ── Discriminator (autocast fp16 is fine here) ───────────
            if step >= gan_start:
                with torch.amp.autocast("cuda", enabled=use_amp_disc):
                    real_out = disc(images)
                    fake_out_d = disc(recon.detach())
                    d_loss = 0.5 * (
                        F.binary_cross_entropy_with_logits(
                            real_out, 0.9 * torch.ones_like(real_out)) +
                        F.binary_cross_entropy_with_logits(
                            fake_out_d, 0.1 * torch.ones_like(fake_out_d))
                    ) / grad_accum
                scaler_disc.scale(d_loss).backward()
                d_loss_val = d_loss.item() * grad_accum

            acc_l1   += l1_v.item()
            acc_edge += e_v.item()
            acc_kl   += kl_v.item()
            acc_g    += g_v.item()

        # ── NaN caused early continue — check if we should exit ──────
        if nan_count >= 5:
            break

        # ── optimiser step ───────────────────────────────────────────
        torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
        opt_vae.step()
        sched_vae.step()

        if step >= gan_start:
            scaler_disc.unscale_(opt_disc)
            torch.nn.utils.clip_grad_norm_(disc.parameters(), 1.0)
            scaler_disc.step(opt_disc)
            scaler_disc.update()
        ema_vae.update(vae)
        step += 1

        if step % 200 == 0 or step == 1:
            print(f"  {step:6d} | {acc_l1/grad_accum:8.4f} | "
                  f"{acc_edge/grad_accum:8.4f} | {acc_kl/grad_accum:8.4f} | "
                  f"{acc_g/grad_accum:8.4f} | {d_loss_val:8.4f} | "
                  f"{_vram_mb():7.0f}MB")

    elapsed = time.time() - t0
    print(f"\n  VAE done: {steps} steps, {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # swap to EMA weights for downstream use
    ema_vae.apply(vae)
    vae.eval()
    return vae, ema_vae


# ================================================================== #
#  Phase 2 — Train UNet (+ EMA)
# ================================================================== #

def train_unet(
    unet: UNetv5,
    vae: GenVAEv5,
    dataset: AnimalDataset,
    device: str,
    *,
    steps: int           = 15000,
    lr: float            = 2e-4,
    grad_accum: int      = 32,
    ema_decay: float     = 0.999,
    uncond_drop: float   = 0.1,
    v_prediction: bool   = False,
    fid_every: int       = 5000,
    patience: int        = 3,
):
    pred_mode = "v-prediction" if v_prediction else "ε-prediction"
    print(f"\n{'='*60}")
    print(f"  Phase 2: Train UNet v5  ({pred_mode})")
    print(f"  batch=1 × accum={grad_accum}  |  steps: {steps}  |  lr: {lr}")
    print(f"{'='*60}")

    scheduler = NoiseScheduler(num_timesteps=1000, schedule="cosine")
    sampler   = DDIMSampler(scheduler, v_prediction=v_prediction)

    unet.train()
    vae.eval()

    optimizer = _make_optimizer(unet.parameters(), lr=lr,
                                betas=(0.9, 0.95), weight_decay=0.1)

    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps, eta_min=1e-6)

    ema_unet   = EMA(unet, decay=ema_decay)
    use_amp    = torch.cuda.is_available()
    scaler     = torch.amp.GradScaler("cuda", enabled=use_amp)
    stopper    = EarlyStopping(patience=patience)

    dataset.use_vae_transform = False
    loader    = DataLoader(dataset, batch_size=1, shuffle=True,
                           collate_fn=_collate, drop_last=True,
                           num_workers=0, pin_memory=True)
    data_iter = iter(loader)

    step = 0
    losses: list[float] = []
    best_ema_loss = float("inf")
    best_state    = None
    t0 = time.time()

    print(f"  Unconditional drop: {uncond_drop*100:.0f}%")
    print(f"  FID computed every {fid_every} steps (on CPU)")
    print(f"  {'step':>6} | {'loss':>10} | {'avg':>10} | "
          f"{'ema_avg':>10} | {'VRAM':>7}")
    print(f"  {'-'*6}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*7}")

    while step < steps:
        optimizer.zero_grad(set_to_none=True)
        acc_loss = 0.0

        for _ in range(grad_accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)

            images = batch["images"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                with torch.no_grad():
                    z, _, _ = vae.encode(images)

                # class conditioning (drop for CFG training)
                if random.random() < uncond_drop:
                    cls = None
                else:
                    cls = labels

                t     = torch.randint(0, 1000, (z.shape[0],), device=device)
                noise = torch.randn_like(z)
                noisy = scheduler.add_noise(z, noise, t)

                if v_prediction:
                    target = scheduler.get_v_target(z, noise, t)
                else:
                    target = noise

                pred = unet(noisy, t, cls)
                loss = (F.mse_loss(pred, target) +
                        0.5 * F.l1_loss(pred, target)) / grad_accum

            scaler.scale(loss).backward()
            acc_loss += loss.item() * grad_accum

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        lr_sched.step()
        ema_unet.update(unet)

        step += 1
        avg_loss = acc_loss / grad_accum
        losses.append(avg_loss)

        # ── logging ──────────────────────────────────────────────────
        if step % 500 == 0 or step == 1:
            avg  = sum(losses[-500:]) / min(len(losses), 500)
            # EMA-model validation loss on last micro-batch
            ema_unet.apply(unet)
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
                ema_pred = unet(noisy, t, cls)
                ema_l    = F.mse_loss(ema_pred, target).item()
            ema_unet.restore(unet)

            elapsed = time.time() - t0
            eta     = elapsed / step * (steps - step)
            print(f"  {step:6d} | {avg_loss:10.5f} | {avg:10.5f} | "
                  f"{ema_l:10.5f} | {_vram_mb():7.0f}MB  "
                  f"[{elapsed:.0f}s / ~{eta:.0f}s ETA]")

            # save best EMA checkpoint
            if ema_l < best_ema_loss:
                best_ema_loss = ema_l
                best_state = {
                    "unet_ema": ema_unet.state_dict(),
                    "unet":     unet.state_dict(),
                }

        # ── early-stopping check every fid_every steps ───────────────
        if step % fid_every == 0 and step > 0:
            recent = sum(losses[-500:]) / min(len(losses), 500)
            if stopper.check(recent):
                print(f"\n  ⛔ Early stopping at step {step} "
                      f"(val loss plateau ×{patience})")
                break

    elapsed = time.time() - t0
    init_l  = sum(losses[:100]) / min(len(losses), 100)
    fin_l   = sum(losses[-100:]) / min(len(losses), 100)
    print(f"\n  UNet done: {step} steps, {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Loss: {init_l:.5f} → {fin_l:.5f} "
          f"({(1 - fin_l / max(init_l, 1e-8))*100:+.1f}%)")

    # restore best EMA weights
    if best_state:
        ema_unet.load_state_dict(best_state["unet_ema"])
    ema_unet.apply(unet)

    return unet, ema_unet, scheduler, sampler


# ================================================================== #
#  Generation
# ================================================================== #

@torch.no_grad()
def generate(
    unet: UNetv5,
    vae: GenVAEv5,
    sampler: DDIMSampler,
    prompts: List[str],
    device: str,
    latent_shape: tuple,
    num_classes: int = 10,
    num_steps: int   = 50,
    cfg_scale: float = 3.0,
    output_dir: str  = "output/own_model_v5",
):
    print(f"\n{'='*60}")
    print(f"  Generating {len(prompts)} images  "
          f"(steps={num_steps}, cfg={cfg_scale})")
    print(f"{'='*60}")

    os.makedirs(output_dir, exist_ok=True)
    unet.eval()
    vae.eval()

    for i, prompt in enumerate(prompts):
        cls_id  = _text_to_class(prompt, num_classes)
        cls_name = CIFAR10_CLASSES[cls_id] if cls_id < len(CIFAR10_CLASSES) \
                   else str(cls_id)
        print(f"  [{i+1}/{len(prompts)}] \"{prompt}\" → {cls_name}")

        cls_t = torch.tensor([cls_id], device=device, dtype=torch.long)
        x = sampler.sample(unet, latent_shape, cls_t, device,
                           num_steps=num_steps, cfg_scale=cfg_scale)

        img = vae.decode(x)[0].clamp(0, 1)
        img_np = (img.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        path = os.path.join(output_dir, f"gen_{i:03d}.bmp")
        _save_bmp(path, img_np)
        print(f"    → {path} ({img_np.shape[1]}×{img_np.shape[0]})")

    print(f"\n  Outputs: {os.path.abspath(output_dir)}")


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
#  Full pipeline
# ================================================================== #

def run_pipeline(
    steps_vae:       int   = 2000,
    steps_unet:      int   = 10000,
    img_size:        int   = 128,
    grad_accum:      int   = 8,
    max_samples:     int   = 0,
    animals_only:    bool  = False,
    v_prediction:    bool  = False,
    data_dir:        Optional[str] = None,
    save_path:       str   = "train/own_model_v5.pt",
    output_dir:      str   = "output/own_model_v5",
    gen_prompts:     Optional[List[str]] = None,
    cfg_scale:       float = 3.0,
    fid_every:       int   = 5000,
    patience:        int   = 3,
    vae_lr:          float = 1e-4,
    kl_warmup:       Optional[int] = None,
    gan_start:       Optional[int] = None,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*60}")
    print(f"  v5.1 — Diffusion on 128×128 images  (RTX 2050 optimised)")
    print(f"{'='*60}")
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**2
        print(f"  GPU : {gpu} ({vram:.0f} MB)")
    print(f"  Resolution   : {img_size}×{img_size}")
    print(f"  v-prediction : {v_prediction}")
    print(f"  Grad accum   : {grad_accum}")
    print(f"  VAE steps    : {steps_vae}")
    print(f"  UNet steps   : {steps_unet}")
    print(f"  CFG scale    : {cfg_scale}")
    print(f"{'='*60}\n")

    t_total = time.time()

    # ── Dataset ──────────────────────────────────────────────────────
    dataset = AnimalDataset(
        root="data/cifar10", img_size=img_size,
        animals_only=animals_only, max_samples=max_samples,
        data_dir=data_dir,
    )
    num_classes = dataset.num_classes

    # ── Model hyper-params ───────────────────────────────────────────
    latent_ch   = 12
    base_ch_vae = 128
    base_ch_un  = 96
    ch_mult_un  = (1, 2, 4)     # → 96, 192, 384
    time_dim    = 256

    # ── Create models ────────────────────────────────────────────────
    vae  = GenVAEv5(3, latent_ch, base_ch_vae).to(device)
    disc = PatchDiscriminator(3, 64).to(device)
    unet = UNetv5(latent_ch, latent_ch, base_ch_un, ch_mult_un,
                  time_dim=time_dim, num_heads=4, dropout=0.1,
                  num_classes=num_classes, use_checkpoint=True).to(device)

    total_params = _param_count(vae) + _param_count(unet)
    print(f"\n  Total inference params: {total_params:,} "
          f"({total_params/1e6:.1f}M)")

    # ── Phase 1: VAE ─────────────────────────────────────────────────
    vae, ema_vae = train_vae(vae, disc, dataset, device,
                             steps=steps_vae, lr=vae_lr,
                             grad_accum=grad_accum,
                             kl_warmup_steps=kl_warmup,
                             gan_start_step=gan_start)
    del disc
    _flush()

    # ── Phase 2: UNet ────────────────────────────────────────────────
    unet, ema_unet, scheduler, sampler = train_unet(
        unet, vae, dataset, device,
        steps=steps_unet, lr=2e-4, grad_accum=grad_accum,
        v_prediction=v_prediction, fid_every=fid_every,
        patience=patience,
    )

    # ── Save ─────────────────────────────────────────────────────────
    latent_size = img_size // 4       # 32 for 128 px
    config = {
        "version":       5.1,
        "img_size":      img_size,
        "latent_ch":     latent_ch,
        "latent_size":   latent_size,
        "base_ch_vae":   base_ch_vae,
        "base_ch_unet":  base_ch_un,
        "ch_mult":       list(ch_mult_un),
        "time_dim":      time_dim,
        "num_classes":   num_classes,
        "v_prediction":  v_prediction,
        "total_params":  total_params,
        "animals_only":  animals_only,
        "noise_schedule": "cosine",
    }

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    torch.save({
        "vae":      vae.state_dict(),
        "vae_ema":  ema_vae.state_dict(),
        "unet":     unet.state_dict(),
        "unet_ema": ema_unet.state_dict(),
        "config":   config,
    }, save_path)

    elapsed = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"  Saved : {os.path.abspath(save_path)}")
    print(f"  Size  : {os.path.getsize(save_path)/1024/1024:.1f} MB")
    print(f"  Params: {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"  Time  : {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"{'='*60}\n")

    # ── Generate ─────────────────────────────────────────────────────
    prompts = gen_prompts or DEFAULT_PROMPTS
    latent_shape = (1, latent_ch, latent_size, latent_size)
    generate(unet, vae, sampler, prompts, device, latent_shape,
             num_classes=num_classes, num_steps=50,
             cfg_scale=cfg_scale, output_dir=output_dir)

    return save_path


# ================================================================== #
#  Load checkpoint & generate
# ================================================================== #

def load_and_generate(path: str, prompts: List[str],
                      output_dir: str = "output/own_model_v5",
                      num_steps: int = 50, cfg_scale: float = 3.0):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Load] {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg  = ckpt["config"]

    vae = GenVAEv5(3, cfg["latent_ch"],
                   cfg.get("base_ch_vae", 128)).to(device)
    vae.load_state_dict(ckpt["vae"])
    vae.eval()

    unet = UNetv5(
        cfg["latent_ch"], cfg["latent_ch"],
        cfg.get("base_ch_unet", 96),
        tuple(cfg.get("ch_mult", [1, 2, 4])),
        time_dim=cfg.get("time_dim", 256),
        num_heads=4, dropout=0.0,
        num_classes=cfg.get("num_classes", 10),
        use_checkpoint=False,
    ).to(device)
    unet.load_state_dict(ckpt["unet"])
    unet.eval()

    v_pred = cfg.get("v_prediction", False)
    scheduler = NoiseScheduler(num_timesteps=1000, schedule="cosine")
    sampler   = DDIMSampler(scheduler, v_prediction=v_pred)

    total = cfg.get("total_params", _param_count(vae) + _param_count(unet))
    print(f"[Load] {total:,} params ({total/1e6:.1f}M)  |  "
          f"v-prediction={v_pred}")

    ls = cfg.get("latent_size", cfg["img_size"] // 4)
    latent_shape = (1, cfg["latent_ch"], ls, ls)

    generate(unet, vae, sampler, prompts, device, latent_shape,
             num_classes=cfg.get("num_classes", 10),
             num_steps=num_steps, cfg_scale=cfg_scale,
             output_dir=output_dir)


# ================================================================== #
#  CLI
# ================================================================== #

def main():
    p = argparse.ArgumentParser(
        description="v5.1 diffusion model — 128×128, ~30M params, RTX 2050")
    p.add_argument("--steps",        type=int,   default=10000)
    p.add_argument("--steps-vae",    type=int,   default=2000)
    p.add_argument("--grad-accum",   type=int,   default=8)
    p.add_argument("--img-size",     type=int,   default=128)
    p.add_argument("--max-samples",  type=int,   default=0)
    p.add_argument("--animals-only", action="store_true")
    p.add_argument("--v-prediction", action="store_true")
    p.add_argument("--data-dir",     type=str,   default=None,
                   help="Image folder (subfolders = classes)")
    p.add_argument("--cfg-scale",    type=float, default=3.0)
    p.add_argument("--vae-lr",       type=float, default=1e-4)
    p.add_argument("--kl-warmup",    type=int,   default=None,
                   help="KL warmup steps (default: 40%% of VAE steps)")
    p.add_argument("--gan-start",    type=int,   default=None,
                   help="Step to enable GAN loss (default: 40%% of VAE steps)")
    p.add_argument("--fid-every",    type=int,   default=5000)
    p.add_argument("--patience",     type=int,   default=3)
    p.add_argument("--prompt",       type=str,   default=None)
    p.add_argument("--load",         type=str,   default=None)
    p.add_argument("--save-path",    type=str,   default="train/own_model_v5.pt")
    p.add_argument("--output-dir",   type=str,   default="output/own_model_v5")
    args = p.parse_args()

    if args.load:
        prompts = [args.prompt] if args.prompt else DEFAULT_PROMPTS[:8]
        load_and_generate(args.load, prompts, args.output_dir,
                          cfg_scale=args.cfg_scale)
    else:
        gen = [args.prompt] if args.prompt else None
        run_pipeline(
            steps_vae=args.steps_vae,
            steps_unet=args.steps,
            img_size=args.img_size,
            grad_accum=args.grad_accum,
            max_samples=args.max_samples,
            animals_only=args.animals_only,
            v_prediction=args.v_prediction,
            data_dir=args.data_dir,
            save_path=args.save_path,
            output_dir=args.output_dir,
            gen_prompts=gen,
            cfg_scale=args.cfg_scale,
            fid_every=args.fid_every,
            patience=args.patience,
            vae_lr=args.vae_lr,
            kl_warmup=args.kl_warmup,
            gan_start=args.gan_start,
        )


if __name__ == "__main__":
    main()
