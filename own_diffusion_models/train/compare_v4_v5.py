"""
Compare v4 vs v5 models:
  1. Compute FID for both models against real CIFAR-10 animal images
  2. Generate a side-by-side HTML comparison page

Usage::
    python -m train.compare_v4_v5
"""
from __future__ import annotations

import base64
import gc
import io
import os
import struct
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── imports from v4 and v5 ──────────────────────────────────────────
from train.train_own_v4 import (
    GenVAE as GenVAEv4,
    ClassConditioner,
    DDIMSampler as DDIMSamplerV4,
    CIFAR10_CLASSES,
    ANIMAL_CLASSES,
    DEFAULT_PROMPTS,
)
from train.train_mini import MiniUNet, SimpleScheduler

from train.train_own_v5 import (
    GenVAEv5,
    UNetv5,
    NoiseScheduler,
    DDIMSampler as DDIMSamplerV5,
    _text_to_class,
    _param_count,
)


def _flush():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ================================================================== #
#  Load real images (CIFAR-10 animals, resized)
# ================================================================== #

def load_real_images(n: int = 64, img_size: int = 128) -> torch.Tensor:
    """Return (N, 3, img_size, img_size) tensor in [0,1]."""
    import torchvision
    import torchvision.transforms as T
    from PIL import Image

    tfm = T.Compose([
        T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
    ])
    cifar = torchvision.datasets.CIFAR10(
        root="data/cifar10", train=True, download=True)

    animal_ids = {i for i, c in enumerate(CIFAR10_CLASSES) if c in ANIMAL_CLASSES}
    imgs = []
    for i in range(len(cifar)):
        if cifar.targets[i] in animal_ids:
            pil = Image.fromarray(cifar.data[i])
            imgs.append(tfm(pil))
            if len(imgs) >= n:
                break
    return torch.stack(imgs[:n])


# ================================================================== #
#  FID computation (lightweight, CPU)
# ================================================================== #

@torch.no_grad()
def compute_fid(real: torch.Tensor, gen: torch.Tensor) -> float:
    """Approx FID via InceptionV3. Both inputs (N, 3, H, W) in [0,1]."""
    try:
        from torchvision.models import inception_v3, Inception_V3_Weights
    except ImportError:
        print("  torchvision not found — skipping FID")
        return float("nan")

    device = "cpu"
    model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
    model.fc = nn.Identity()
    model.eval().to(device)

    def feats(imgs):
        out = []
        for i in range(imgs.shape[0]):
            x = imgs[i:i+1].to(device)
            x = F.interpolate(x, size=(299, 299), mode="bilinear",
                              align_corners=False)
            out.append(model(x).cpu())
        return torch.cat(out, 0).float()

    f_r = feats(real)
    f_g = feats(gen)

    mu_r, mu_g = f_r.mean(0), f_g.mean(0)
    sig_r = torch.cov(f_r.T)
    sig_g = torch.cov(f_g.T)

    diff = mu_r - mu_g
    fid = diff.dot(diff).item()

    try:
        product = sig_r @ sig_g
        product = (product + product.T) / 2
        eigvals, eigvecs = torch.linalg.eigh(product)
        eigvals = eigvals.clamp(min=0).sqrt()
        sqrtm = eigvecs @ torch.diag(eigvals) @ eigvecs.T
        fid += torch.trace(sig_r + sig_g - 2 * sqrtm).item()
    except Exception:
        fid += (torch.trace(sig_r).item() + torch.trace(sig_g).item())

    del model
    _flush()
    return max(0.0, fid)


# ================================================================== #
#  Load and generate from v4
# ================================================================== #

@torch.no_grad()
def generate_v4(ckpt_path: str, prompts: list[str],
                device: str) -> list[torch.Tensor]:
    """Return list of (3, H, W) tensors in [0,1]."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]

    vae = GenVAEv4(3, cfg["latent_ch"], cfg.get("base_ch_vae", 64)).to(device)
    vae.load_state_dict(ckpt["vae"])
    vae.eval()

    cond = ClassConditioner(
        cfg.get("num_classes", 10), cfg["context_dim"],
        cfg.get("num_tokens", 4)).to(device)
    cond.load_state_dict(ckpt["cond"])
    cond.eval()

    unet = MiniUNet(
        cfg["latent_ch"], cfg["latent_ch"], cfg["context_dim"],
        cfg.get("base_ch_unet", 96),
        tuple(cfg.get("ch_mult", [1, 2, 4])),
        cfg.get("time_dim", 128)).to(device)
    unet.load_state_dict(ckpt["unet"])
    unet.eval()

    scheduler = SimpleScheduler(num_timesteps=1000)
    sampler = DDIMSamplerV4(scheduler)
    ls = cfg.get("latent_size", cfg["img_size"] // 4)
    latent_shape = (1, cfg["latent_ch"], ls, ls)
    uncond_ctx = torch.zeros(1, cond.num_tokens, cond.context_dim, device=device)

    results = []
    for prompt in prompts:
        cls_id = cond.text_to_class(prompt)
        cls_t = torch.tensor([cls_id], device=device)
        ctx = cond(cls_t)
        x = sampler.sample(unet, latent_shape, ctx, device,
                           num_steps=50, cfg_scale=3.0,
                           uncond_context=uncond_ctx)
        img = vae.decode(x)[0].clamp(0, 1).cpu()
        results.append(img)

    del vae, unet, cond
    _flush()
    return results


# ================================================================== #
#  Load and generate from v5
# ================================================================== #

@torch.no_grad()
def generate_v5(ckpt_path: str, prompts: list[str],
                device: str) -> list[torch.Tensor]:
    """Return list of (3, H, W) tensors in [0,1]."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]

    vae = GenVAEv5(3, cfg["latent_ch"],
                   cfg.get("base_ch_vae", 128)).to(device)
    vae.load_state_dict(ckpt["vae"])
    vae.eval()

    unet = UNetv5(
        cfg["latent_ch"], cfg["latent_ch"],
        cfg.get("base_ch_unet", 64),
        tuple(cfg.get("ch_mult", [1, 2, 4])),
        time_dim=cfg.get("time_dim", 256),
        num_heads=4, dropout=0.0,
        num_classes=cfg.get("num_classes", 10),
        use_checkpoint=False).to(device)
    unet.load_state_dict(ckpt["unet"])
    unet.eval()

    v_pred = cfg.get("v_prediction", False)
    scheduler = NoiseScheduler(num_timesteps=1000)
    sampler = DDIMSamplerV5(scheduler, v_prediction=v_pred)
    ls = cfg.get("latent_size", cfg["img_size"] // 4)
    latent_shape = (1, cfg["latent_ch"], ls, ls)

    num_classes = cfg.get("num_classes", 10)
    results = []
    for prompt in prompts:
        cls_id = _text_to_class(prompt, num_classes)
        cls_t = torch.tensor([cls_id], device=device, dtype=torch.long)
        x = sampler.sample(unet, latent_shape, cls_t, device,
                           num_steps=50, cfg_scale=3.0)
        img = vae.decode(x)[0].clamp(0, 1).cpu()
        results.append(img)

    del vae, unet
    _flush()
    return results


# ================================================================== #
#  BMP read helper (for existing output files)
# ================================================================== #

def read_bmp_as_tensor(path: str) -> torch.Tensor:
    """Read BMP file → (3, H, W) tensor in [0,1]."""
    from PIL import Image
    img = Image.open(path).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


# ================================================================== #
#  HTML generation
# ================================================================== #

def tensor_to_png_b64(t: torch.Tensor, size: int = 256) -> str:
    """(3,H,W) [0,1] → base64 PNG for embedding in HTML."""
    from PIL import Image
    img = (t.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    pil = Image.fromarray(img).resize((size, size), Image.NEAREST)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def build_html(
    prompts: list[str],
    v4_imgs: list[torch.Tensor],
    v5_imgs: list[torch.Tensor],
    real_imgs: torch.Tensor,
    fid_v4: float,
    fid_v5: float,
    v4_params: int,
    v5_params: int,
    out_path: str,
):
    rows = ""
    for i, prompt in enumerate(prompts):
        v4_b64 = tensor_to_png_b64(v4_imgs[i]) if i < len(v4_imgs) else ""
        v5_b64 = tensor_to_png_b64(v5_imgs[i]) if i < len(v5_imgs) else ""
        rows += f"""
        <tr>
          <td class="prompt">{prompt}</td>
          <td><img src="data:image/png;base64,{v4_b64}"></td>
          <td><img src="data:image/png;base64,{v5_b64}"></td>
        </tr>"""

    # A few real images for reference
    real_row = ""
    for i in range(min(8, real_imgs.shape[0])):
        b64 = tensor_to_png_b64(real_imgs[i], size=128)
        real_row += f'<img src="data:image/png;base64,{b64}" style="margin:4px;">'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>v4 vs v5 Comparison</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #0d1117;
         color: #c9d1d9; max-width: 1100px; margin: 0 auto; padding: 24px; }}
  h1 {{ color: #58a6ff; text-align: center; }}
  h2 {{ color: #8b949e; border-bottom: 1px solid #30363d; padding-bottom: 8px; }}
  .stats {{ display: flex; gap: 32px; justify-content: center; margin: 24px 0; }}
  .stat-card {{ background: #161b22; border: 1px solid #30363d; border-radius: 12px;
                padding: 20px 32px; text-align: center; min-width: 200px; }}
  .stat-card h3 {{ margin: 0 0 8px; color: #58a6ff; }}
  .stat-card .value {{ font-size: 28px; font-weight: bold; }}
  .stat-card .label {{ font-size: 13px; color: #8b949e; margin-top: 4px; }}
  .better {{ color: #3fb950; }}
  .worse {{ color: #f85149; }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 16px; }}
  th {{ background: #161b22; color: #58a6ff; padding: 12px; text-align: center;
       border-bottom: 2px solid #30363d; }}
  td {{ padding: 8px; text-align: center; border-bottom: 1px solid #21262d;
       vertical-align: middle; }}
  td.prompt {{ text-align: left; font-style: italic; color: #8b949e;
              max-width: 200px; font-size: 14px; }}
  img {{ width: 192px; height: 192px; border-radius: 8px;
        border: 1px solid #30363d; image-rendering: auto; }}
  .real-imgs {{ text-align: center; margin: 16px 0; }}
  .real-imgs img {{ width: 96px; height: 96px; border-radius: 6px; }}
  .footer {{ text-align: center; color: #484f58; font-size: 12px; margin-top: 32px; }}
</style>
</head>
<body>
<h1>v4 vs v5 — Side-by-Side Comparison</h1>

<div class="stats">
  <div class="stat-card">
    <h3>v4</h3>
    <div class="value {'worse' if fid_v4 > fid_v5 else 'better'}">{fid_v4:.1f}</div>
    <div class="label">FID ↓ better</div>
    <div class="label" style="margin-top:8px">{v4_params/1e6:.1f}M params</div>
  </div>
  <div class="stat-card">
    <h3>v5</h3>
    <div class="value {'better' if fid_v5 < fid_v4 else 'worse'}">{fid_v5:.1f}</div>
    <div class="label">FID ↓ better</div>
    <div class="label" style="margin-top:8px">{v5_params/1e6:.1f}M params</div>
  </div>
</div>

<h2>Real Reference Images (CIFAR-10 Animals, upscaled)</h2>
<div class="real-imgs">{real_row}</div>

<h2>Generated Samples</h2>
<table>
  <tr><th>Prompt</th><th>v4 (64×64)</th><th>v5 (128×128)</th></tr>
  {rows}
</table>

<div class="footer">Generated by compare_v4_v5.py</div>
</body>
</html>"""

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  HTML saved: {os.path.abspath(out_path)}")


# ================================================================== #
#  Main
# ================================================================== #

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    v4_path = "train/own_model_v4.pt"
    v5_path = "train/own_model_v5.pt"

    prompts = [
        "a photo of a cat",
        "a cute puppy",
        "a photo of a bird",
        "a green frog",
        "a brown horse",
        "a deer in nature",
        "a fluffy dog",
        "a small kitten",
    ]

    print("=" * 60)
    print("  v4 vs v5 Comparison")
    print("=" * 60)

    # ── Load real images for FID ─────────────────────────────────────
    print("\n[1/5] Loading real animal images ...")
    real = load_real_images(n=64, img_size=128)
    print(f"  Real images: {real.shape}")

    # ── Generate v4 ──────────────────────────────────────────────────
    print("\n[2/5] Generating v4 images ...")
    t0 = time.time()
    v4_imgs = generate_v4(v4_path, prompts, device)
    print(f"  v4: {len(v4_imgs)} images in {time.time()-t0:.1f}s")
    _flush()

    # ── Generate v5 ──────────────────────────────────────────────────
    print("\n[3/5] Generating v5 images ...")
    t0 = time.time()
    v5_imgs = generate_v5(v5_path, prompts, device)
    print(f"  v5: {len(v5_imgs)} images in {time.time()-t0:.1f}s")
    _flush()

    # ── Compute FID ──────────────────────────────────────────────────
    print("\n[4/5] Computing FID (this may take a few minutes on CPU) ...")

    # Stack generated images and resize to 128×128 for fair comparison
    v4_stack = torch.stack(v4_imgs)
    if v4_stack.shape[-1] != 128:
        v4_stack = F.interpolate(v4_stack, size=(128, 128), mode="bilinear",
                                 align_corners=False)
    v5_stack = torch.stack(v5_imgs)
    if v5_stack.shape[-1] != 128:
        v5_stack = F.interpolate(v5_stack, size=(128, 128), mode="bilinear",
                                 align_corners=False)

    # Use more real images for FID reference
    real_fid = real[:64]

    print("  Computing FID for v4 ...")
    fid_v4 = compute_fid(real_fid, v4_stack)
    print(f"  FID v4 = {fid_v4:.1f}")

    print("  Computing FID for v5 ...")
    fid_v5 = compute_fid(real_fid, v5_stack)
    print(f"  FID v5 = {fid_v5:.1f}")

    # ── Param counts ─────────────────────────────────────────────────
    ckpt_v4 = torch.load(v4_path, map_location="cpu", weights_only=False)
    v4_params = ckpt_v4["config"].get("total_params", 0)
    del ckpt_v4

    ckpt_v5 = torch.load(v5_path, map_location="cpu", weights_only=False)
    v5_params = ckpt_v5["config"].get("total_params", 0)
    del ckpt_v5

    # ── Build HTML ───────────────────────────────────────────────────
    print("\n[5/5] Building HTML comparison ...")
    build_html(
        prompts, v4_imgs, v5_imgs, real,
        fid_v4, fid_v5, v4_params, v5_params,
        out_path="output/compare_v4_v5.html",
    )

    print(f"\n{'='*60}")
    print(f"  FID v4: {fid_v4:.1f}")
    print(f"  FID v5: {fid_v5:.1f}")
    winner = "v5" if fid_v5 < fid_v4 else "v4"
    print(f"  Winner: {winner}")
    print(f"  HTML:   output/compare_v4_v5.html")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
