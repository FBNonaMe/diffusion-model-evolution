# diffusion-model-evolution
Lightweight diffusion models trained from scratch on CIFAR-10 (PyTorch). Evolution from 1.2M to 30M parameters, optimized for low-VRAM GPUs (4 GB).

Пайплайн обучения собственных генеративных диффузионных моделей изображений на PyTorch.  
От синтетических фигур (v1) до реальных фотографий 128×128 (v5.1).

**Обучено и протестировано на NVIDIA RTX 2050 (4 GB VRAM).**

---

## Результаты

### Сгенерированные изображения (v5.1, 128×128)

Промпты: `"a photo of a cat"`, `"a cute puppy"`, `"a photo of a bird"`, `"a green frog"`,
`"a brown horse"`, `"a deer in nature"`, `"a fluffy dog"`, `"a small kitten"` и другие.

Выходные файлы: `output/own_model_v5/gen_*.bmp` (16 изображений).

### Benchmark — все 5 версий

#### Архитектура

| | **v1 (12.6M)** | **v2 (14.4M)** | **v3 (143.8M)** | **v4 (17.8M)** | **v5.1 (30.0M)** |
|---|---|---|---|---|---|
| Total params | 12.6M | 14.4M | 143.8M | 17.8M | **30.0M** |
| UNet params | 10.7M | 10.7M | 136.6M | 13.9M | 11.5M |
| VAE params | 1.6M | 3.5M | 6.2M | 3.9M | **18.4M** |
| VAE type | MiniVAE (÷8) | SharpVAE (÷4) | SharpVAE (÷4) | GenVAE (÷4) | **GenVAEv5 (÷4)** |
| Conditioning | Hash tokenizer | Hash tokenizer | Hash tokenizer | Class embed | **Class embed** |
| Latent space | 4ch, 8×8 | 4ch, 16×16 | 4ch, 16×16 | 8ch, 16×16 | **12ch, 32×32** |

#### Производительность (RTX 2050, 4 GB VRAM)

| | **v1** | **v2** | **v3** | **v4** | **v5.1** |
|---|---|---|---|---|---|
| Checkpoint | 48.1 MB | 55.2 MB | 548.5 MB | 68.2 MB | **229.0 MB** |
| Model VRAM | 49 MB | 65 MB | 563 MB | 77 MB | 124 MB |
| Gen VRAM peak | 62 MB | 75 MB | 625 MB | 83 MB | **165 MB** |
| Gen time (avg) | **0.22 s** | 0.35 s | 0.53 s | 0.57 s | 0.61 s |
| Speed (img/s) | **4.58** | 2.88 | 1.88 | 1.75 | 1.63 |

#### Качество генерации

| Метрика | **v1** | **v2** | **v3** | **v4** | **v5.1** |
|---|---|---|---|---|---|
| Colorfulness | 0.034 | 0.000 | 0.000 | **0.060** | 0.043 |
| Sharpness | 0.151 | 0.123 | 0.122 | **0.374** | 0.204 |
| Dynamic range | 0.399 | 0.014 | 0.003 | **0.799** | 0.721 |
| **Diversity** | 0.054 | 0.000 | 0.000 | 0.218 | **0.390 ✅** |
| Pixel std | 0.092 | 0.006 | 0.001 | 0.162 | 0.156 |
| Gray outputs | 0/8 | 8/8 ❌ | 8/8 ❌ | 0/8 ✅ | 0/8 ✅ |

#### Итоги

| Категория | Лучшая модель |
|---|---|
| 🏆 Наименьший размер | **v1** (12.6M, 48 MB) |
| 🏆 Самая быстрая | **v1** (0.22 с, 4.58 img/s) |
| 🏆 Минимум VRAM | **v1** (62 MB peak) |
| 🏆 Наиболее цветная | **v4** (colorfulness 0.060) |
| 🏆 Наибольшая чёткость | **v4** (sharpness 0.374) |
| 🏆 Лучший дин. диапазон | **v4** (dynamic range 0.799) |
| 🏆 **Наибольшее разнообразие** | **v5.1** (diversity 0.390) |

Визуальные отчёты: `output/benchmark/*.png`

---

## Эволюция моделей

### v1 — MiniVAE + MiniUNet (12.6M)

**Первая версия.** Синтетический датасет (круги, квадраты, треугольники).
MiniVAE с ÷8 downsampling (128→8×8 латент, 4 канала). Hash-токенизатор.

- Скрипт: `train/train_own.py` (653 строки)
- Результат: размытые цветные фигуры, 64×64
- Время: 27 с

### v2 — SharpVAE + GAN (14.4M)

**Чёткость.** L1 + Edge (Sobel) + GAN вместо MSE.
SharpVAE с ÷4, skip-соединениями (16×16 латент).

- Скрипт: `train/train_own_v2.py` (630 строк)
- Результат: чёткие фигуры, но mode collapse (серые выходы 8/8)
- Время: 200 с

### v3 — Масштабирование (143.8M)

**Масштаб.** UNet 136.6M (каналы [224, 448, 672, 896]).
12 фигур, 18 цветов, 11 фонов.

- Скрипт: `train/train_own_v3.py` (620 строк)
- Результат: mode collapse сохранился (8/8 серых)
- Время: 486 с (8 мин), 2.2 GB VRAM

### v4 — Реальные фото (17.8M)

**Переход на реальные данные.** CIFAR-10 (30K фото животных).
GenVAE без skip-соединений. Class embedding + CFG.

- Скрипт: `train/train_own_v4.py` (764 строки)
- Результат: **первые реалистичные изображения**, яркие цвета
- VAE L1: 0.006 (в 28× лучше v3), UNet loss: −61%
- Время: 2 062 с (34 мин)

### v5.1 — High-res + Attention (30.0M) ★

**Текущая лучшая модель.** 128×128. GenVAEv5 (18.4M) — 12ch латент 32×32.
UNet (11.5M) с self-attention на 16×16. Cosine β, CosineAnnealingLR, EMA.

- Скрипт: `train/train_own_v5.py` (1 210 строк)
- Результат: **128×128, лучшее diversity (0.39)**
- UNet loss: 1.07→0.71 (−34%)
- Время: 17 012 с (284 мин), 578 MB peak VRAM
- Чекпоинт: 229 MB

---

## Архитектура v5.1

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

## Быстрый старт

### Требования

- Python 3.10+
- PyTorch 2.0+ с CUDA
- NVIDIA GPU ≥ 4 GB VRAM

### Установка

```bash
git clone https://github.com/<your-username>/own-diffusion-models.git
cd own-diffusion-models

python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### Обучение

```bash
# v5.1 — полное обучение (~284 мин на RTX 2050)
python -m train.train_own_v5 --animals-only

# v5.1 — быстрый тест (~50 мин)
python -m train.train_own_v5 --animals-only --steps-vae 2000 --steps 3000 \
    --grad-accum 8 --vae-lr 1e-4 --kl-warmup 600 --gan-start 800

# v5.1 — полный прогон (10K UNet шагов)
python -m train.train_own_v5 --animals-only --steps 10000 --steps-vae 2000 \
    --grad-accum 8 --cfg-scale 3.0

# v4 — предыдущая версия (~34 мин)
python -m train.train_own_v4 --animals-only

# v1/v2/v3 — синтетика
python -m train.train_own        # v1, ~27 с
python -m train.train_own_v2     # v2, ~3 мин
python -m train.train_own_v3     # v3, ~8 мин
```

### Генерация из обученной модели

```bash
python -m train.train_own_v5 --load train/own_model_v5.pt
python -m train.train_own_v5 --load train/own_model_v5.pt --prompt "a fluffy cat"
python -m train.train_own_v5 --load train/own_model_v5.pt --cfg-scale 5.0
```

### Бенчмарк

```bash
python -m train.benchmark
# → output/benchmark/*.png (6 графиков)
```

### Сравнение v4 vs v5 (FID)

```bash
python -m train.compare_v4_v5
# → output/compare_v4_v5.html
```

---

## CLI параметры v5.1

| Параметр | По умолчанию | Описание |
|---|---|---|
| `--steps` | 10000 | Шаги обучения UNet |
| `--steps-vae` | 2000 | Шаги обучения VAE |
| `--grad-accum` | 8 | Gradient accumulation |
| `--img-size` | 128 | Разрешение изображений |
| `--animals-only` | false | Только животные из CIFAR-10 |
| `--v-prediction` | false | v-prediction вместо ε-prediction |
| `--cfg-scale` | 3.0 | CFG scale |
| `--vae-lr` | 1e-4 | Learning rate для VAE |
| `--kl-warmup` | 40% steps | KL warmup шаги |
| `--gan-start` | 40% steps | Шаг включения GAN loss |
| `--fid-every` | 5000 | Интервал FID-проверки |
| `--patience` | 3 | Early-stopping patience |
| `--data-dir` | — | Папка с изображениями |
| `--load` | — | Загрузить чекпоинт |
| `--prompt` | — | Текстовый промпт |
| `--save-path` | `train/own_model_v5.pt` | Путь сохранения |
| `--output-dir` | `output/own_model_v5` | Папка выходных изображений |

---

## Структура проекта

```
own_diffusion_models/
├── README.md                     # Этот файл
├── requirements.txt              # Зависимости Python
├── .gitignore                    # Исключения для Git
│
├── train/                        # Скрипты обучения
│   ├── __init__.py
│   ├── train_own.py              # v1: 12.6M, синтетика (653 строки)
│   ├── train_own_v2.py           # v2: 14.4M, GAN (630 строк)
│   ├── train_own_v3.py           # v3: 143.8M, масштаб (620 строк)
│   ├── train_own_v4.py           # v4: 17.8M, CIFAR-10 (764 строки)
│   ├── train_own_v5.py           # v5.1: 30.0M, 128×128, attention ★
│   ├── benchmark.py              # Бенчмарк v1–v5 (799 строк)
│   ├── compare_v4_v5.py          # FID сравнение v4 vs v5
│   ├── own_model.pt              # Чекпоинт v1 (48 MB)
│   ├── own_model_v2.pt           # Чекпоинт v2 (55 MB)
│   ├── own_model_v3.pt           # Чекпоинт v3 (549 MB)
│   ├── own_model_v4.pt           # Чекпоинт v4 (68 MB)
│   └── own_model_v5.pt           # Чекпоинт v5.1 (229 MB) ★
│
├── output/                       # Результаты
│   ├── benchmark/                # Графики бенчмарка
│   │   ├── side_by_side.png
│   │   ├── quality_metrics.png
│   │   ├── vram_usage.png
│   │   ├── gen_speed.png
│   │   ├── latent_viz.png
│   │   └── loss_curves.png
│   ├── compare_v4_v5.html
│   ├── own_model/                # Выходы v1
│   ├── own_model_v2/             # Выходы v2
│   ├── own_model_v3/             # Выходы v3
│   ├── own_model_v4/             # Выходы v4 (64×64)
│   └── own_model_v5/             # Выходы v5.1 (128×128) ★
│       ├── gen_000.bmp ... gen_015.bmp
│
└── data/                         # Датасет (auto-download)
    └── cifar10/                  # CIFAR-10 (~170 MB)
```

**Всего:** ~5 000 строк Python в 7 скриптах.

---

## Ключевые решения

### NaN в VAE

GenVAEv5 с `base_ch=128` при 128×128 порождает активации за пределами fp16.  
**Решение:** fp32 для VAE, fp16 autocast только для UNet/дискриминатора.

### KL explosion

KL divergence растёт с 0.15→6.5 к шагу 200 при малом KL weight.  
**Решение:** KL weight 5e-3→5e-2 с warmup 40%.

### Mode collapse (v2/v3)

VAE со skip-соединениями даёт серый выход при генерации из random z.  
**Решение:** VAE без skip (GenVAE, v4+).

### Cosine β schedule (v5.1)

Линейный β концентрирует SNR на малых timesteps. Cosine schedule
даёт равномерное распределение → лучшее diversity (0.39 vs 0.22).

---

## Hardware

| Компонент | Значение |
|---|---|
| GPU | NVIDIA GeForce RTX 2050 |
| VRAM | 4096 MB |
| RAM | 16 GB |
| ОС | Windows 10 |
| Python | 3.13 |
| PyTorch | 2.6.0+cu124 |

Пиковое VRAM при обучении v5.1: **578 MB** (из 4096 MB).

---

## License

MIT
