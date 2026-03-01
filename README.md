# Own Diffusion Models — v1 → v5.1

Training pipeline for custom generative diffusion image models in PyTorch.  
From synthetic shapes (v1) to real 128×128 photographs (v5.1).

**Trained and tested on NVIDIA RTX 2050 (4 GB VRAM).**

---

## Results

### Generated Images (v5.1, 128×128)

Prompts: `"a photo of a cat"`, `"a cute puppy"`, `"a photo of a bird"`, `"a green frog"`,
`"a brown horse"`, `"a deer in nature"`, `"a fluffy dog"`, `"a small kitten"`, and others.

Output files: `output/own_model_v5/gen_*.bmp` (16 images).

### Benchmark — All 5 Versions

#### Architecture

| | **v1 (12.6M)** | **v2 (14.4M)** | **v3 (143.8M)** | **v4 (17.8M)** | **v5.1 (30.0M)** |
|---|---|---|---|---|---|
| Total params | 12.6M | 14.4M | 143.8M | 17.8M | **30.0M** |
| UNet params | 10.7M | 10.7M | 136.6M | 13.9M | 11.5M |
| VAE params | 1.6M | 3.5M | 6.2M | 3.9M | **18.4M** |
| VAE type | MiniVAE (÷8) | SharpVAE (÷4) | SharpVAE (÷4) | GenVAE (÷4) | **GenVAEv5 (÷4)** |
| Conditioning | Hash tokenizer | Hash tokenizer | Hash tokenizer | Class embed | **Class embed** |
| Latent space | 4ch, 8×8 | 4ch, 16×16 | 4ch, 16×16 | 8ch, 16×16 | **12ch, 32×32** |

#### Performance (RTX 2050, 4 GB VRAM)

| | **v1** | **v2** | **v3** | **v4** | **v5.1** |
|---|---|---|---|---|---|
| Checkpoint | 48.1 MB | 55.2 MB | 548.5 MB | 68.2 MB | **229.0 MB** |
| Model VRAM | 49 MB | 65 MB | 563 MB | 77 MB | 124 MB |
| Gen VRAM peak | 62 MB | 75 MB | 625 MB | 83 MB | **165 MB** |
| Gen time (avg) | **0.22 s** | 0.35 s | 0.53 s | 0.57 s | 0.61 s |
| Speed (img/s) | **4.58** | 2.88 | 1.88 | 1.75 | 1.63 |

#### Generation Quality

| Metric | **v1** | **v2** | **v3** | **v4** | **v5.1** |
|---|---|---|---|---|---|
| Colorfulness | 0.034 | 0.000 | 0.000 | **0.060** | 0.043 |
| Sharpness | 0.151 | 0.123 | 0.122 | **0.374** | 0.204 |
| Dynamic range | 0.399 | 0.014 | 0.003 | **0.799** | 0.721 |
| **Diversity** | 0.054 | 0.000 | 0.000 | 0.218 | **0.390 ✅** |
| Pixel std | 0.092 | 0.006 | 0.001 | 0.162 | 0.156 |
| Gray outputs | 0/8 | 8/8 ❌ | 8/8 ❌ | 0/8 ✅ | 0/8 ✅ |

#### Summary

| Category | Best Model |
|---|---|
| 🏆 Smallest size | **v1** (12.6M, 48 MB) |
| 🏆 Fastest | **v1** (0.22 s, 4.58 img/s) |
| 🏆 Lowest VRAM | **v1** (62 MB peak) |
| 🏆 Most colorful | **v4** (colorfulness 0.060) |
| 🏆 Sharpest | **v4** (sharpness 0.374) |
| 🏆 Best dynamic range | **v4** (dynamic range 0.799) |
| 🏆 **Most diverse** | **v5.1** (diversity 0.390) |

Visual reports: `output/benchmark/*.png`

---

## Model Evolution

### v1 — MiniVAE + MiniUNet (12.6M)

**First version.** Synthetic dataset (circles, squares, triangles).
MiniVAE with ÷8 downsampling (128→8×8 latent, 4 channels). Hash tokenizer.

- Script: `train/train_own.py` (653 lines)
- Result: blurry colored shapes, 64×64
- Time: 27 s

### v2 — SharpVAE + GAN (14.4M)

**Sharpness.** L1 + Edge (Sobel) + GAN instead of MSE.
SharpVAE with ÷4, skip connections (16×16 latent).

- Script: `train/train_own_v2.py` (630 lines)
- Result: sharp shapes, but mode collapse (gray outputs 8/8)
- Time: 200 s

### v3 — Scaling Up (143.8M)

**Scale.** UNet 136.6M (channels [224, 448, 672, 896]).
12 shapes, 18 colors, 11 backgrounds.

- Script: `train/train_own_v3.py` (620 lines)
- Result: mode collapse persisted (8/8 gray)
- Time: 486 s (8 min), 2.2 GB VRAM

### v4 — Real Photos (17.8M)

**Transition to real data.** CIFAR-10 (30K animal photos).
GenVAE without skip connections. Class embedding + CFG.

- Script: `train/train_own_v4.py` (764 lines)
- Result: **first realistic images**, vivid colors
- VAE L1: 0.006 (28× better than v3), UNet loss: −61%
- Time: 2,062 s (34 min)

### v5.1 — High-res + Attention (30.0M) ★

**Current best model.** 128×128. GenVAEv5 (18.4M) — 12ch latent 32×32.
UNet (11.5M) with self-attention at 16×16. Cosine β, CosineAnnealingLR, EMA.

- Script: `train/train_own_v5.py` (1,210 lines)
- Result: **128×128, best diversity (0.39)**
- UNet loss: 1.07→0.71 (−34%)
- Time: 17,012 s (284 min), 578 MB peak VRAM
- Checkpoint: 229 MB

---

## v5.1 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              Own Model v5.1  —  30.0M params                │
│                   128×128 RGB → 128×128 RGB                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  GenVAEv5  (18.4M)                                          │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Encoder: Conv→128ch──3×ResBlk──↓2──256ch──         │    │
│  │           3×ResBlk──↓2──256ch──3×ResBlk             │    │
│  │  Latent:  μ + σ → z ∈ ℝ^{12×32×32}                 │    │
│  │           logvar ∈ [−6, 6]                          │    │
│  │  Decoder: Conv←128ch──3×ResBlk──↑2──256ch──         │    │
│  │           3×ResBlk──↑2──256ch──3×ResBlk──Sigmoid    │    │
│  │  Loss: L1 + 0.3·Edge + KL(5e-3→5e-2) + GAN         │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
│  UNet v5.1  (11.5M)                                         │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Enc Stage 1:  96ch (32×32) — 2×ResBlk              │    │
│  │  Enc Stage 2: 192ch (16×16) — 2×ResBlk + Self-Attn  │    │
│  │  Middle:      384ch  (8×8)  — ResBlk + MHA + ResBlk │    │
│  │  Dec Stage 2: 192ch (16×16) — 2×ResBlk + Self-Attn  │    │
│  │  Dec Stage 1:  96ch (32×32) — 2×ResBlk              │    │
│  │                                                     │    │
│  │  Time:  LearnedSinusoidal → MLP(256→1024→256)       │    │
│  │  Class: nn.Embedding(10, 256) → add to time emb     │    │
│  │  Skip:  concat + 1×1 conv projection                │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
│  Training:                                                  │
│  ├── Noise: cosine β schedule (Nichol & Dhariwal, 2021)     │
│  ├── Sampler: DDIM, 50 steps                                │
│  ├── LR: 2e-4, CosineAnnealingLR (eta_min=1e-6)            │
│  ├── Optimizer: AdamW (β₁=0.9, β₂=0.95, wd=0.1)           │
│  ├── EMA: decay=0.999, generation from EMA weights          │
│  ├── CFG: classifier-free guidance (scale=3.0)              │
│  ├── Grad accumulation: 8                                   │
│  └── Precision: fp32 (VAE) + fp16 autocast (UNet)           │
└─────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Requirements

- Python 3.10+
- PyTorch 2.0+ with CUDA
- NVIDIA GPU ≥ 4 GB VRAM

### Installation

```bash
git clone https://github.com/FBNonaMe/diffusion-model-evolution.git
cd diffusion-model-evolution

python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### Training

```bash
# v5.1 — full training (~284 min on RTX 2050)
python -m train.train_own_v5 --animals-only

# v5.1 — quick test (~50 min)
python -m train.train_own_v5 --animals-only --steps-vae 2000 --steps 3000 \
    --grad-accum 8 --vae-lr 1e-4 --kl-warmup 600 --gan-start 800

# v5.1 — full run (10K UNet steps)
python -m train.train_own_v5 --animals-only --steps 10000 --steps-vae 2000 \
    --grad-accum 8 --cfg-scale 3.0

# v4 — previous version (~34 min)
python -m train.train_own_v4 --animals-only

# v1/v2/v3 — synthetic data
python -m train.train_own        # v1, ~27 s
python -m train.train_own_v2     # v2, ~3 min
python -m train.train_own_v3     # v3, ~8 min
```

### Generate from Trained Model

```bash
python -m train.train_own_v5 --load train/own_model_v5.pt
python -m train.train_own_v5 --load train/own_model_v5.pt --prompt "a fluffy cat"
python -m train.train_own_v5 --load train/own_model_v5.pt --cfg-scale 5.0
```

### Benchmark

```bash
python -m train.benchmark
# → output/benchmark/*.png (6 charts)
```

### Compare v4 vs v5 (FID)

```bash
python -m train.compare_v4_v5
# → output/compare_v4_v5.html
```

---

## CLI Parameters (v5.1)

| Parameter | Default | Description |
|---|---|---|
| `--steps` | 10000 | UNet training steps |
| `--steps-vae` | 2000 | VAE training steps |
| `--grad-accum` | 8 | Gradient accumulation |
| `--img-size` | 128 | Image resolution |
| `--animals-only` | false | Use only animals from CIFAR-10 |
| `--v-prediction` | false | v-prediction instead of ε-prediction |
| `--cfg-scale` | 3.0 | CFG scale |
| `--vae-lr` | 1e-4 | VAE learning rate |
| `--kl-warmup` | 40% steps | KL warmup steps |
| `--gan-start` | 40% steps | Step to enable GAN loss |
| `--fid-every` | 5000 | FID check interval |
| `--patience` | 3 | Early-stopping patience |
| `--data-dir` | — | Image data directory |
| `--load` | — | Load checkpoint |
| `--prompt` | — | Text prompt |
| `--save-path` | `train/own_model_v5.pt` | Checkpoint save path |
| `--output-dir` | `output/own_model_v5` | Output image directory |

---

## Project Structure

```
cifar10-diffusion-from-scratch/
├── README.md                     # This file
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git exclusions
│
├── train/                        # Training scripts
│   ├── __init__.py
│   ├── train_own.py              # v1: 12.6M, synthetic (653 lines)
│   ├── train_own_v2.py           # v2: 14.4M, GAN (630 lines)
│   ├── train_own_v3.py           # v3: 143.8M, scaled up (620 lines)
│   ├── train_own_v4.py           # v4: 17.8M, CIFAR-10 (764 lines)
│   ├── train_own_v5.py           # v5.1: 30.0M, 128×128, attention ★
│   ├── benchmark.py              # Benchmark v1–v5 (799 lines)
│   ├── compare_v4_v5.py          # FID comparison v4 vs v5
│   ├── own_model.pt              # Checkpoint v1 (48 MB)
│   ├── own_model_v2.pt           # Checkpoint v2 (55 MB)
│   ├── own_model_v3.pt           # Checkpoint v3 (549 MB)
│   ├── own_model_v4.pt           # Checkpoint v4 (68 MB)
│   └── own_model_v5.pt           # Checkpoint v5.1 (229 MB) ★
│
├── output/                       # Results
│   ├── benchmark/                # Benchmark charts
│   │   ├── side_by_side.png
│   │   ├── quality_metrics.png
│   │   ├── vram_usage.png
│   │   ├── gen_speed.png
│   │   ├── latent_viz.png
│   │   └── loss_curves.png
│   ├── compare_v4_v5.html
│   ├── own_model/                # v1 outputs
│   ├── own_model_v2/             # v2 outputs
│   ├── own_model_v3/             # v3 outputs
│   ├── own_model_v4/             # v4 outputs (64×64)
│   └── own_model_v5/             # v5.1 outputs (128×128) ★
│       ├── gen_000.bmp ... gen_015.bmp
│
└── data/                         # Dataset (auto-download)
    └── cifar10/                  # CIFAR-10 (~170 MB)
```

**Total:** ~5,000 lines of Python across 7 scripts.

---

## Key Design Decisions

### NaN in VAE

GenVAEv5 with `base_ch=128` at 128×128 produces activations beyond fp16 range.  
**Solution:** fp32 for VAE, fp16 autocast only for UNet/discriminator.

### KL Explosion

KL divergence grows from 0.15→6.5 by step 200 with low KL weight.  
**Solution:** KL weight 5e-3→5e-2 with 40% warmup.

### Mode Collapse (v2/v3)

VAE with skip connections produces gray outputs when generating from random z.  
**Solution:** VAE without skip connections (GenVAE, v4+).

### Cosine β Schedule (v5.1)

Linear β concentrates SNR on small timesteps. Cosine schedule
provides uniform distribution → better diversity (0.39 vs 0.22).

---

## Hardware

| Component | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 2050 |
| VRAM | 4096 MB |
| RAM | 16 GB |
| OS | Windows 10 |
| Python | 3.13 |
| PyTorch | 2.6.0+cu124 |

Peak VRAM during v5.1 training: **578 MB** (out of 4096 MB).

---

## License

MIT
