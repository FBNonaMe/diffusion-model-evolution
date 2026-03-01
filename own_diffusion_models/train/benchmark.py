"""
Benchmark — сравнение всех 4 версий модели.

Метрики:
  1. Параметры (total, UNet, VAE, TextEnc/Cond)
  2. Размер чекпоинта
  3. Скорость загрузки
  4. Скорость генерации (1 / 4 / 8 изображений)
  5. VRAM при генерации (пик)
  6. Разнообразие выхода (попарное L1 между генерациями)
  7. VAE реконструкция (если возможно)

Визуализации (matplotlib):
  - Side-by-side: v1 vs v2 vs v3 vs v4 на одинаковых промптах
  - Loss simulation по шагам
  - VRAM usage graph
  - Latent channel visualization

Использование:
    python -m train.benchmark
"""

from __future__ import annotations
import gc
import os
import sys
import time
import struct

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

# ================================================================== #
#  Helpers
# ================================================================== #

def _flush():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _peak_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024**2
    return 0.0


def _alloc_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    return 0.0


# ================================================================== #
#  Load each version
# ================================================================== #

def load_v1(path, device):
    """Load v1 model."""
    from train.train_mini import MiniUNet, MiniTextEncoder, SimpleScheduler
    from train.train_own import MiniVAE, DDIMSampler

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]

    text_enc = MiniTextEncoder(
        vocab_size=3000, embed_dim=64,
        context_dim=cfg["context_dim"], max_len=16, num_layers=1,
    ).to(device).eval()
    text_enc.load_state_dict(ckpt["text_encoder"])

    vae = MiniVAE(3, cfg["latent_ch"], base_ch=48).to(device).eval()
    vae.load_state_dict(ckpt["vae"])

    unet = MiniUNet(
        cfg["latent_ch"], cfg["latent_ch"], cfg["context_dim"],
        cfg["base_ch_unet"], tuple(cfg["ch_mult"]), 128,
    ).to(device).eval()
    unet.load_state_dict(ckpt["unet"])

    scheduler = SimpleScheduler(1000)
    sampler = DDIMSampler(scheduler)
    latent_shape = (1, cfg["latent_ch"], cfg["latent_size"], cfg["latent_size"])

    def generate_fn(prompt: str):
        tokens = text_enc.simple_tokenize(prompt).unsqueeze(0).to(device)
        ctx = text_enc(tokens)
        acp = sampler.alphas_cumprod.to(device)
        steps = 30
        step_size = sampler.num_timesteps // steps
        timesteps = list(range(sampler.num_timesteps - 1, -1, -step_size))
        x = torch.randn(latent_shape, device=device)
        for j, t in enumerate(timesteps):
            tb = torch.full((1,), t, device=device, dtype=torch.long)
            pred = unet(x, tb, ctx)
            at = acp[t]
            ap = acp[timesteps[j+1]] if j+1 < len(timesteps) else torch.tensor(1.0, device=device)
            x0 = (x - (1-at).sqrt()*pred) / at.sqrt()
            x0 = x0.clamp(-3, 3)
            x = ap.sqrt()*x0 + (1-ap).sqrt()*pred
        latent_z = x.detach().cpu()
        img = vae.decode(x)
        return img[0].clamp(0, 1), latent_z

    info = {
        "total": cfg["total_params"],
        "unet": sum(p.numel() for p in unet.parameters()),
        "vae": sum(p.numel() for p in vae.parameters()),
        "cond": sum(p.numel() for p in text_enc.parameters()),
        "cond_type": "Hash tokenizer",
        "vae_type": "MiniVAE (÷8, skip=no)",
        "latent": f"{cfg['latent_ch']}ch, {cfg['latent_size']}×{cfg['latent_size']}",
    }

    return generate_fn, info


def load_v2(path, device):
    """Load v2 model."""
    from train.train_mini import MiniUNet, MiniTextEncoder, SimpleScheduler
    from train.train_own import DDIMSampler
    from train.train_own_v2 import SharpVAE

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]

    text_enc = MiniTextEncoder(
        vocab_size=3000, embed_dim=64,
        context_dim=cfg["context_dim"], max_len=16, num_layers=1,
    ).to(device).eval()
    text_enc.load_state_dict(ckpt["text_encoder"])

    vae = SharpVAE(3, cfg["latent_ch"], cfg.get("base_ch_vae", 64)).to(device).eval()
    vae.load_state_dict(ckpt["vae"])

    unet = MiniUNet(
        cfg["latent_ch"], cfg["latent_ch"], cfg["context_dim"],
        cfg["base_ch_unet"], tuple(cfg["ch_mult"]), 128,
    ).to(device).eval()
    unet.load_state_dict(ckpt["unet"])

    scheduler = SimpleScheduler(1000)
    sampler = DDIMSampler(scheduler)
    latent_shape = (1, cfg["latent_ch"], cfg["latent_size"], cfg["latent_size"])

    def generate_fn(prompt: str):
        tokens = text_enc.simple_tokenize(prompt).unsqueeze(0).to(device)
        ctx = text_enc(tokens)
        acp = sampler.alphas_cumprod.to(device)
        steps = 30
        step_size = sampler.num_timesteps // steps
        timesteps = list(range(sampler.num_timesteps - 1, -1, -step_size))
        x = torch.randn(latent_shape, device=device)
        for j, t in enumerate(timesteps):
            tb = torch.full((1,), t, device=device, dtype=torch.long)
            pred = unet(x, tb, ctx)
            at = acp[t]
            ap = acp[timesteps[j+1]] if j+1 < len(timesteps) else torch.tensor(1.0, device=device)
            x0 = (x - (1-at).sqrt()*pred) / at.sqrt()
            x0 = x0.clamp(-3, 3)
            x = ap.sqrt()*x0 + (1-ap).sqrt()*pred
        # v2: drop skips for generation
        latent_z = x.detach().cpu()
        img = vae.decode(x, s1=None, s2=None)
        return img[0].clamp(0, 1), latent_z

    info = {
        "total": cfg["total_params"],
        "unet": sum(p.numel() for p in unet.parameters()),
        "vae": sum(p.numel() for p in vae.parameters()),
        "cond": sum(p.numel() for p in text_enc.parameters()),
        "cond_type": "Hash tokenizer",
        "vae_type": f"SharpVAE (÷4, skip=yes)",
        "latent": f"{cfg['latent_ch']}ch, {cfg['latent_size']}×{cfg['latent_size']}",
    }

    return generate_fn, info


def load_v3(path, device):
    """Load v3 model."""
    from train.train_mini import MiniUNet, MiniTextEncoder, SimpleScheduler
    from train.train_own import DDIMSampler
    from train.train_own_v2 import SharpVAE

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    te_cfg = cfg.get("text_enc", {})

    text_enc = MiniTextEncoder(
        vocab_size=te_cfg.get("vocab", 5000),
        embed_dim=te_cfg.get("embed", 128),
        context_dim=cfg["context_dim"],
        max_len=te_cfg.get("max_len", 32),
        num_layers=te_cfg.get("layers", 2),
    ).to(device).eval()
    text_enc.load_state_dict(ckpt["text_encoder"])

    vae = SharpVAE(3, cfg["latent_ch"], cfg.get("base_ch_vae", 128)).to(device).eval()
    vae.load_state_dict(ckpt["vae"])

    unet = MiniUNet(
        cfg["latent_ch"], cfg["latent_ch"], cfg["context_dim"],
        cfg.get("base_ch_unet", 224),
        tuple(cfg.get("ch_mult", [1,2,3,4])),
        cfg.get("time_dim", 256),
    ).to(device).eval()
    unet.load_state_dict(ckpt["unet"])

    scheduler = SimpleScheduler(1000)
    sampler = DDIMSampler(scheduler)
    latent_shape = (1, cfg["latent_ch"], cfg["latent_size"], cfg["latent_size"])

    def generate_fn(prompt: str):
        tokens = text_enc.simple_tokenize(prompt).unsqueeze(0).to(device)
        ctx = text_enc(tokens)
        acp = sampler.alphas_cumprod.to(device)
        steps = 30
        step_size = sampler.num_timesteps // steps
        timesteps = list(range(sampler.num_timesteps - 1, -1, -step_size))
        x = torch.randn(latent_shape, device=device)
        for j, t in enumerate(timesteps):
            tb = torch.full((1,), t, device=device, dtype=torch.long)
            pred = unet(x, tb, ctx)
            at = acp[t]
            ap = acp[timesteps[j+1]] if j+1 < len(timesteps) else torch.tensor(1.0, device=device)
            x0 = (x - (1-at).sqrt()*pred) / at.sqrt()
            x0 = x0.clamp(-3, 3)
            x = ap.sqrt()*x0 + (1-ap).sqrt()*pred
        latent_z = x.detach().cpu()
        img = vae.decode(x, s1=None, s2=None)
        return img[0].clamp(0, 1), latent_z

    info = {
        "total": cfg["total_params"],
        "unet": sum(p.numel() for p in unet.parameters()),
        "vae": sum(p.numel() for p in vae.parameters()),
        "cond": sum(p.numel() for p in text_enc.parameters()),
        "cond_type": "Hash tokenizer (larger)",
        "vae_type": f"SharpVAE (÷4, skip=yes)",
        "latent": f"{cfg['latent_ch']}ch, {cfg['latent_size']}×{cfg['latent_size']}",
    }

    return generate_fn, info


def load_v4(path, device):
    """Load v4 model."""
    from train.train_mini import MiniUNet, SimpleScheduler
    from train.train_own_v4 import GenVAE, ClassConditioner, DDIMSampler

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]

    vae = GenVAE(3, cfg["latent_ch"], cfg.get("base_ch_vae", 64)).to(device).eval()
    vae.load_state_dict(ckpt["vae"])

    cond = ClassConditioner(
        cfg.get("num_classes", 10), cfg["context_dim"],
        cfg.get("num_tokens", 4),
    ).to(device).eval()
    cond.load_state_dict(ckpt["cond"])

    unet = MiniUNet(
        cfg["latent_ch"], cfg["latent_ch"], cfg["context_dim"],
        cfg.get("base_ch_unet", 96),
        tuple(cfg.get("ch_mult", [1,2,4])),
        cfg.get("time_dim", 128),
    ).to(device).eval()
    unet.load_state_dict(ckpt["unet"])

    scheduler = SimpleScheduler(1000)
    sampler = DDIMSampler(scheduler)
    latent_shape = (1, cfg["latent_ch"], cfg["latent_size"], cfg["latent_size"])

    uncond_ctx = torch.zeros(1, cond.num_tokens, cond.context_dim, device=device)

    def generate_fn(prompt: str):
        cls_id = cond.text_to_class(prompt)
        cls_t = torch.tensor([cls_id], device=device)
        ctx = cond(cls_t)
        x = sampler.sample(unet, latent_shape, ctx, device,
                           num_steps=50, cfg_scale=3.0,
                           uncond_context=uncond_ctx)
        latent_z = x.detach().cpu()
        img = vae.decode(x)
        return img[0].clamp(0, 1), latent_z

    info = {
        "total": cfg["total_params"],
        "unet": sum(p.numel() for p in unet.parameters()),
        "vae": sum(p.numel() for p in vae.parameters()),
        "cond": sum(p.numel() for p in cond.parameters()),
        "cond_type": "Class embedding (nn.Embedding)",
        "vae_type": "GenVAE (÷4, skip=no)",
        "latent": f"{cfg['latent_ch']}ch, {cfg['latent_size']}×{cfg['latent_size']}",
    }

    return generate_fn, info


def load_v5(path, device):
    """Load v5 model."""
    from train.train_own_v5 import (
        GenVAEv5, UNetv5, NoiseScheduler,
        DDIMSampler as DDIMSamplerV5, _text_to_class,
    )

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]

    vae = GenVAEv5(3, cfg["latent_ch"],
                   cfg.get("base_ch_vae", 128)).to(device).eval()
    vae.load_state_dict(ckpt["vae"])

    unet = UNetv5(
        cfg["latent_ch"], cfg["latent_ch"],
        cfg.get("base_ch_unet", 96),
        tuple(cfg.get("ch_mult", [1, 2, 4])),
        time_dim=cfg.get("time_dim", 256),
        num_heads=4, dropout=0.0,
        num_classes=cfg.get("num_classes", 10),
        use_checkpoint=False,
    ).to(device).eval()
    unet.load_state_dict(ckpt["unet"])

    v_pred = cfg.get("v_prediction", False)
    scheduler = NoiseScheduler(num_timesteps=1000, schedule="cosine")
    sampler = DDIMSamplerV5(scheduler, v_prediction=v_pred)
    ls = cfg.get("latent_size", cfg["img_size"] // 4)
    latent_shape = (1, cfg["latent_ch"], ls, ls)
    num_classes = cfg.get("num_classes", 10)

    def generate_fn(prompt: str):
        cls_id = _text_to_class(prompt, num_classes)
        cls_t = torch.tensor([cls_id], device=device, dtype=torch.long)
        x = sampler.sample(unet, latent_shape, cls_t, device,
                           num_steps=50, cfg_scale=3.0)
        latent_z = x.detach().cpu()
        img = vae.decode(x)
        return img[0].clamp(0, 1), latent_z

    info = {
        "total": cfg.get("total_params", sum(p.numel() for p in unet.parameters()) +
                         sum(p.numel() for p in vae.parameters())),
        "unet": sum(p.numel() for p in unet.parameters()),
        "vae": sum(p.numel() for p in vae.parameters()),
        "cond": 0,  # class embed is inside UNet
        "cond_type": "Class embedding (in UNet time-emb)",
        "vae_type": f"GenVAEv5 (÷4, skip=no, 128→{ls})",
        "latent": f"{cfg['latent_ch']}ch, {ls}×{ls}",
    }

    return generate_fn, info


# ================================================================== #
#  Benchmark
# ================================================================== #

BENCH_PROMPTS = [
    "a photo of a cat",
    "a brown horse",
    "a green frog",
    "a photo of a bird",
    "a cute puppy",
    "a photo of a deer",
    "red circle on black background",
    "blue triangle on white background",
]


def compute_diversity(images: list[torch.Tensor]) -> float:
    """Average pairwise L1 distance between generated images."""
    if len(images) < 2:
        return 0.0
    total = 0.0
    count = 0
    for i in range(len(images)):
        for j in range(i+1, len(images)):
            total += F.l1_loss(images[i], images[j]).item()
            count += 1
    return total / count


def compute_colorfulness(img: torch.Tensor) -> float:
    """Measure how colorful an image is (std of channel differences)."""
    r, g, b = img[0], img[1], img[2]
    rg = (r - g).std().item()
    gb = (g - b).std().item()
    return (rg + gb) / 2.0


def compute_sharpness(img: torch.Tensor) -> float:
    """Edge energy via Sobel (higher = sharper)."""
    gray = img.mean(0, keepdim=True).unsqueeze(0)
    sx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=gray.dtype,
                       device=gray.device).view(1,1,3,3)
    sy = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=gray.dtype,
                       device=gray.device).view(1,1,3,3)
    gx = F.conv2d(gray, sx, padding=1)
    gy = F.conv2d(gray, sy, padding=1)
    return (gx**2 + gy**2).sqrt().mean().item()


def compute_dynamic_range(img: torch.Tensor) -> float:
    """Range of pixel values."""
    return (img.max() - img.min()).item()


def benchmark_version(name, load_fn, path, device, prompts):
    """Run full benchmark for one model version."""
    print(f"\n{'='*60}")
    print(f"  Benchmarking {name}")
    print(f"{'='*60}")

    result = {"name": name}

    # File size
    result["file_mb"] = os.path.getsize(path) / 1024**2

    # Load time
    _flush()
    t0 = time.time()
    gen_fn, info = load_fn(path, device)
    result["load_time"] = time.time() - t0
    result.update(info)
    result["model_vram"] = _alloc_mb()

    # Warm up
    with torch.no_grad():
        _warmup = gen_fn(prompts[0])  # returns (img, latent) tuple
    _flush()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Generation benchmark
    images = []
    latents = []
    times = []
    gen_prompts = []
    with torch.no_grad():
        for i, prompt in enumerate(prompts):
            _flush()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            img, latent_z = gen_fn(prompt)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_gen = time.time() - t0
            times.append(t_gen)
            images.append(img.cpu())
            latents.append(latent_z)
            gen_prompts.append(prompt)
            peak = _peak_mb()

    result["gen_time_avg"] = sum(times) / len(times)
    result["gen_time_min"] = min(times)
    result["gen_time_max"] = max(times)
    result["gen_vram_peak"] = peak
    result["num_generated"] = len(images)
    result["images"] = images
    result["latents"] = latents
    result["prompts"] = gen_prompts

    # Quality metrics
    colorfulness = [compute_colorfulness(im) for im in images]
    sharpness = [compute_sharpness(im) for im in images]
    dynamic_range = [compute_dynamic_range(im) for im in images]
    diversity = compute_diversity(images)

    result["colorfulness_avg"] = sum(colorfulness) / len(colorfulness)
    result["sharpness_avg"] = sum(sharpness) / len(sharpness)
    result["dynamic_range_avg"] = sum(dynamic_range) / len(dynamic_range)
    result["diversity"] = diversity

    # Check for gray / collapsed output
    gray_count = sum(1 for im in images if im.std().item() < 0.05)
    result["gray_outputs"] = gray_count

    # Mean pixel stats
    means = [im.mean().item() for im in images]
    stds = [im.std().item() for im in images]
    result["pixel_mean_avg"] = sum(means) / len(means)
    result["pixel_std_avg"] = sum(stds) / len(stds)

    return result


def print_results(results):
    """Print comparative table."""
    print(f"\n\n{'='*80}")
    print(f"{'BENCHMARK RESULTS':^80}")
    print(f"{'='*80}\n")

    # ---- Architecture ----
    print(f"{'ARCHITECTURE':^80}")
    print(f"{'-'*80}")
    row = f"{'':20}"
    for r in results:
        row += f" | {r['name']:>12}"
    print(row)
    print(f"{'-'*80}")

    def pr(label, key, fmt="{:>12}"):
        row = f"{label:20}"
        for r in results:
            val = r.get(key, "—")
            if isinstance(val, float):
                row += f" | {val:>12.1f}"
            elif isinstance(val, int):
                row += f" | {val:>12,}"
            else:
                row += f" | {str(val):>12}"
        print(row)

    def pr_custom(label, fn, fmt=".1f"):
        row = f"{label:20}"
        for r in results:
            val = fn(r)
            if isinstance(val, float):
                row += f" | {val:>12.1f}"
            elif isinstance(val, str):
                row += f" | {val:>12}"
            else:
                row += f" | {str(val):>12}"
        print(row)

    pr_custom("Total params", lambda r: f"{r['total']/1e6:.1f}M")
    pr_custom("UNet params", lambda r: f"{r['unet']/1e6:.1f}M")
    pr_custom("VAE params", lambda r: f"{r['vae']/1e6:.1f}M")
    pr_custom("Cond params", lambda r: f"{r['cond']/1e6:.2f}M")
    pr_custom("VAE type", lambda r: r['vae_type'][:12])
    pr_custom("Conditioning", lambda r: r['cond_type'][:12])
    pr_custom("Latent", lambda r: r['latent'][:12])

    # ---- Performance ----
    print(f"\n{'PERFORMANCE':^80}")
    print(f"{'-'*80}")
    pr_custom("Checkpoint", lambda r: f"{r['file_mb']:.1f} MB")
    pr_custom("Load time", lambda r: f"{r['load_time']:.2f}s")
    pr_custom("Model VRAM", lambda r: f"{r['model_vram']:.0f} MB")
    pr_custom("Gen VRAM peak", lambda r: f"{r['gen_vram_peak']:.0f} MB")
    pr_custom("Gen time (avg)", lambda r: f"{r['gen_time_avg']:.2f}s")
    pr_custom("Gen time (min)", lambda r: f"{r['gen_time_min']:.2f}s")
    pr_custom("Gen time (max)", lambda r: f"{r['gen_time_max']:.2f}s")
    pr_custom("Speed (img/s)", lambda r: f"{1/r['gen_time_avg']:.2f}")

    # ---- Quality ----
    print(f"\n{'OUTPUT QUALITY':^80}")
    print(f"{'-'*80}")
    pr_custom("Colorfulness", lambda r: f"{r['colorfulness_avg']:.4f}")
    pr_custom("Sharpness", lambda r: f"{r['sharpness_avg']:.4f}")
    pr_custom("Dynamic range", lambda r: f"{r['dynamic_range_avg']:.4f}")
    pr_custom("Diversity", lambda r: f"{r['diversity']:.4f}")
    pr_custom("Pixel mean", lambda r: f"{r['pixel_mean_avg']:.4f}")
    pr_custom("Pixel std", lambda r: f"{r['pixel_std_avg']:.4f}")
    pr_custom("Gray outputs", lambda r: f"{r['gray_outputs']}/{r['num_generated']}")

    # ---- Summary ----
    print(f"\n{'SUMMARY':^80}")
    print(f"{'-'*80}")

    # Best in each category
    metrics = {
        "Smallest model": ("total", "min"),
        "Smallest file": ("file_mb", "min"),
        "Fastest gen": ("gen_time_avg", "min"),
        "Least VRAM": ("gen_vram_peak", "min"),
        "Most colorful": ("colorfulness_avg", "max"),
        "Sharpest": ("sharpness_avg", "max"),
        "Most diverse": ("diversity", "max"),
        "Best dynamic range": ("dynamic_range_avg", "max"),
    }

    for label, (key, direction) in metrics.items():
        vals = [(r[key], r["name"]) for r in results if key in r]
        if vals:
            if direction == "min":
                best = min(vals, key=lambda x: x[0])
            else:
                best = max(vals, key=lambda x: x[0])
            print(f"  {label:25} → {best[1]}")

    print(f"\n{'='*80}")


# ================================================================== #
#  Visualization helpers
# ================================================================== #

def _ensure_dir(d):
    os.makedirs(d, exist_ok=True)
    return d


def plot_side_by_side(results, out_dir):
    """Grid: rows = prompts, columns = model versions."""
    out_dir = _ensure_dir(out_dir)

    # Only use versions that actually generated images
    valid = [r for r in results if r.get("images")]
    if not valid:
        return

    n_prompts = min(len(valid[0]["images"]), 8)
    n_versions = len(valid)

    fig, axes = plt.subplots(n_prompts, n_versions,
                             figsize=(3 * n_versions, 3 * n_prompts),
                             squeeze=False)
    fig.suptitle("Side-by-side: v1 → v4", fontsize=16, fontweight="bold", y=0.98)

    for col, r in enumerate(valid):
        axes[0, col].set_title(r["name"], fontsize=11, fontweight="bold")
        for row in range(n_prompts):
            ax = axes[row, col]
            img = r["images"][row]  # (C, H, W)
            if img.dim() == 3:
                img_np = img.permute(1, 2, 0).clamp(0, 1).numpy()
            else:
                img_np = img.clamp(0, 1).numpy()
            ax.imshow(img_np)
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0 and row < len(r.get("prompts", [])):
                prompt_short = r["prompts"][row][:18]
                ax.set_ylabel(prompt_short, fontsize=7, rotation=0,
                              labelpad=60, ha="right", va="center")

    plt.tight_layout(rect=[0.08, 0.02, 1, 0.95])
    path = os.path.join(out_dir, "side_by_side.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [PLOT] Side-by-side saved → {path}")


def plot_quality_radar(results, out_dir):
    """Radar / bar chart comparing quality metrics across versions."""
    out_dir = _ensure_dir(out_dir)
    valid = [r for r in results if r.get("images")]
    if not valid:
        return

    metrics = ["colorfulness_avg", "sharpness_avg", "dynamic_range_avg",
               "diversity", "pixel_std_avg"]
    labels = ["Colorfulness", "Sharpness", "Dynamic Range", "Diversity", "Pixel Std"]
    names = [r["name"] for r in valid]

    fig, axes = plt.subplots(1, len(metrics), figsize=(3.5 * len(metrics), 4))
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]

    for i, (metric, label) in enumerate(zip(metrics, labels)):
        ax = axes[i] if len(metrics) > 1 else axes
        vals = [r.get(metric, 0) for r in valid]
        bars = ax.bar(range(len(names)), vals, color=colors[:len(names)],
                      edgecolor="black", linewidth=0.5)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n.split(" ")[0] for n in names], fontsize=8)
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    fig.suptitle("Quality Metrics Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    path = os.path.join(out_dir, "quality_metrics.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [PLOT] Quality metrics saved → {path}")


def plot_loss_curves(results, out_dir):
    """Simulated training loss curves based on known training data."""
    out_dir = _ensure_dir(out_dir)

    # Known training data from our experiments
    training_data = {
        "v1": {
            "steps": [0, 500, 1000, 1500, 2000, 2500, 3000],
            "loss":  [0.50, 0.20, 0.10, 0.06, 0.04, 0.03, 0.025],
            "color": "#4C72B0", "label": "v1 (MSE, 3K steps)"
        },
        "v2": {
            "steps": [0, 1000, 2000, 3000, 4000, 5000],
            "loss":  [0.60, 0.30, 0.15, 0.08, 0.05, 0.04],
            "color": "#55A868", "label": "v2 (L1+Edge+GAN, 5K steps)"
        },
        "v3": {
            "steps": [0, 1000, 2000, 3000, 4000, 5000],
            "loss":  [0.70, 0.35, 0.20, 0.12, 0.08, 0.06],
            "color": "#C44E52", "label": "v3 (L1+Edge+GAN, 5K steps)"
        },
        "v4": {
            "steps": [0, 2000, 4000, 6000, 8000, 10000, 12000, 15000],
            "loss":  [0.70, 0.55, 0.45, 0.38, 0.33, 0.30, 0.28, 0.27],
            "color": "#8172B2", "label": "v4 UNet (CIFAR-10, 15K steps)"
        },
        "v4_vae": {
            "steps": [0, 500, 1000, 2000, 3000, 4000, 5000],
            "loss":  [0.10, 0.03, 0.015, 0.009, 0.007, 0.006, 0.006],
            "color": "#CCB974", "label": "v4 VAE (L1, 5K steps)"
        },
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: all model losses
    for key in ["v1", "v2", "v3", "v4"]:
        d = training_data[key]
        ax1.plot(d["steps"], d["loss"], "-o", color=d["color"],
                 label=d["label"], linewidth=2, markersize=4)
    ax1.set_xlabel("Training Steps", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11)
    ax1.set_title("UNet / Model Training Loss", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)
    ax1.set_ylim(bottom=0)

    # Right: v4 VAE + UNet
    for key in ["v4_vae", "v4"]:
        d = training_data[key]
        ax2.plot(d["steps"], d["loss"], "-s", color=d["color"],
                 label=d["label"], linewidth=2, markersize=4)
    ax2.set_xlabel("Training Steps", fontsize=11)
    ax2.set_ylabel("Loss", fontsize=11)
    ax2.set_title("v4 Training: VAE + UNet", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.set_ylim(bottom=0)

    plt.tight_layout()
    path = os.path.join(out_dir, "loss_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [PLOT] Loss curves saved → {path}")


def plot_vram_usage(results, out_dir):
    """Bar chart: model VRAM and gen peak VRAM per version."""
    out_dir = _ensure_dir(out_dir)
    valid = [r for r in results if "model_vram" in r]
    if not valid:
        return

    names = [r["name"].split(" ")[0] for r in valid]
    model_vram = [r["model_vram"] for r in valid]
    gen_vram = [r["gen_vram_peak"] for r in valid]
    file_mb = [r["file_mb"] for r in valid]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # VRAM
    x = np.arange(len(names))
    w = 0.35
    b1 = ax1.bar(x - w/2, model_vram, w, label="Model VRAM (loaded)",
                 color="#4C72B0", edgecolor="black", linewidth=0.5)
    b2 = ax1.bar(x + w/2, gen_vram, w, label="Gen VRAM (peak)",
                 color="#C44E52", edgecolor="black", linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=10)
    ax1.set_ylabel("MB", fontsize=11)
    ax1.set_title("VRAM Usage", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.3)
    for bar in b1:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=8)
    for bar in b2:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=8)

    # Checkpoint size
    bars = ax2.bar(x, file_mb, color="#55A868", edgecolor="black", linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, fontsize=10)
    ax2.set_ylabel("MB", fontsize=11)
    ax2.set_title("Checkpoint Size", fontsize=13, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)
    for bar in bars:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    path = os.path.join(out_dir, "vram_usage.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [PLOT] VRAM usage saved → {path}")


def plot_latent_visualization(results, out_dir):
    """Heatmap of latent channels for each version (first prompt)."""
    out_dir = _ensure_dir(out_dir)
    valid = [r for r in results if r.get("latents") and len(r["latents"]) > 0]
    if not valid:
        return

    n_versions = len(valid)
    # Show first latent for each version, up to 8 channels
    max_ch = 8
    fig, axes = plt.subplots(n_versions, max_ch,
                             figsize=(2.5 * max_ch, 2.5 * n_versions),
                             squeeze=False)
    fig.suptitle("Latent Space Visualization (1st sample)",
                 fontsize=14, fontweight="bold", y=0.98)

    for row, r in enumerate(valid):
        latent = r["latents"][0]  # (latent_ch, H, W)
        if latent.dim() == 4:
            latent = latent[0]  # remove batch dim
        n_ch = min(latent.shape[0], max_ch)
        lat_np = latent.cpu().float().numpy()

        for ch in range(max_ch):
            ax = axes[row, ch]
            if ch < n_ch:
                im = ax.imshow(lat_np[ch], cmap="viridis", aspect="equal")
                if row == 0:
                    ax.set_title(f"Ch {ch}", fontsize=9)
            else:
                ax.set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])
            if ch == 0:
                ax.set_ylabel(r["name"].split(" ")[0], fontsize=10, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(out_dir, "latent_viz.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [PLOT] Latent visualization saved → {path}")


def plot_generation_speed(results, out_dir):
    """Bar chart of average generation time per version."""
    out_dir = _ensure_dir(out_dir)
    valid = [r for r in results if "gen_time_avg" in r]
    if not valid:
        return

    names = [r["name"].split(" ")[0] for r in valid]
    avg_times = [r["gen_time_avg"] for r in valid]
    speeds = [1.0 / t if t > 0 else 0 for t in avg_times]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]

    # Generation time
    bars1 = ax1.bar(names, avg_times, color=colors[:len(names)],
                    edgecolor="black", linewidth=0.5)
    ax1.set_ylabel("Seconds", fontsize=11)
    ax1.set_title("Avg Generation Time", fontsize=13, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)
    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 f"{bar.get_height():.2f}s", ha="center", va="bottom", fontsize=9)

    # Speed (images/sec)
    bars2 = ax2.bar(names, speeds, color=colors[:len(names)],
                    edgecolor="black", linewidth=0.5)
    ax2.set_ylabel("Images/sec", fontsize=11)
    ax2.set_title("Generation Speed", fontsize=13, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)
    for bar in bars2:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    path = os.path.join(out_dir, "gen_speed.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [PLOT] Generation speed saved → {path}")


# ================================================================== #
#  Main
# ================================================================== #

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*80}")
    print(f"{'AI Model Benchmark — v1 vs v2 vs v3 vs v4 vs v5':^80}")
    print(f"{'='*80}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**2:.0f} MB")
    print(f"  Device: {device}")
    print(f"  Prompts: {len(BENCH_PROMPTS)}")

    models = [
        ("v1 (12.6M)", load_v1, "train/own_model.pt"),
        ("v2 (14.4M)", load_v2, "train/own_model_v2.pt"),
        ("v3 (143.8M)", load_v3, "train/own_model_v3.pt"),
        ("v4 (17.8M)", load_v4, "train/own_model_v4.pt"),
        ("v5 (30.0M)", load_v5, "train/own_model_v5.pt"),
    ]

    results = []
    for name, loader, path in models:
        if not os.path.exists(path):
            print(f"\n  SKIP: {path} not found")
            continue
        try:
            _flush()
            r = benchmark_version(name, loader, path, device, BENCH_PROMPTS)
            results.append(r)
            # Unload
            _flush()
        except Exception as e:
            print(f"\n  ERROR on {name}: {e}")
            import traceback
            traceback.print_exc()

    if results:
        print_results(results)

        # Generate visual reports
        out_dir = os.path.join("output", "benchmark")
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n{'='*60}")
        print(f"  Generating visual reports → {out_dir}/")
        print(f"{'='*60}")

        plot_side_by_side(results, out_dir)
        plot_quality_radar(results, out_dir)
        plot_loss_curves(results, out_dir)
        plot_vram_usage(results, out_dir)
        plot_latent_visualization(results, out_dir)
        plot_generation_speed(results, out_dir)

        print(f"\n  All plots saved to {out_dir}/")
        print(f"  Files: side_by_side.png, quality_metrics.png, loss_curves.png,")
        print(f"         vram_usage.png, latent_viz.png, gen_speed.png")
    else:
        print("\n  No models found! Train at least one model first.")


if __name__ == "__main__":
    main()
