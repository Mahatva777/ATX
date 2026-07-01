# BAH 2026 — PS10: IR Image Colorization & Enhancement

**Team:** ATX  
**Problem Statement:** PS10 — Infrared Image Colorization and Enhancement for Improved Object Interpretation  
**Track:** AI/ML · Bhartiya Antriksh Hackathon 2026

---

## 📦 Submission Artefacts

| Artefact | Location |
|----------|----------|
| **This codebase** | [github.com/Mahatva777/ATX](https://github.com/Mahatva777/ATX) |
| **Model weights** (`sr_best.pth`, `colorize_G_best.pth`, `colorize_D_last.pth`) | [Google Drive](https://drive.google.com/drive/folders/1TN0A_Vp4rMtlfKeF_baBdQPfBR0_GS4m?usp=sharing) |
| **Technical presentation** | [`BAH2026_PS10_ATX_Submission.pdf`](BAH2026_PS10_ATX_Submission.pdf) (this repo) |
| **Kaggle training notebook** | [`arm2026.ipynb`](arm2026.ipynb) (this repo) |
| **Sample outputs** | Inside model weights Drive folder → `output/model_outputs/` |

---

## 🗂 Repository Layout

```
ps10_project/
│
├── arm2026.ipynb                  ← Kaggle T4 GPU training notebook (run this)
├── BAH2026_PS10_ATX_Submission.pdf ← Technical presentation
├── README.md                      ← This file
├── README-2.md                    ← Original ISRO baseline README (problem spec)
│
├── models/
│   ├── sr_model.py                ← TIRSuperResolutionNet (SRResNet-lite, 785K params)
│   ├── colorization_model.py      ← UNetGenerator (9-level Pix2Pix, 54.4M params)
│   │                                 + PatchGANDiscriminator (2.8M params)
│   └── semantic_constraint.py     ← BandRatioConsistencyLoss
│
├── datasets/
│   └── patch_dataset.py           ← SRPatchDataset, ColorizationPatchDataset
│                                     (recursive dir-walk pairing, spatial-dim shape checks)
│
├── train_sr.py                    ← SR training loop (L1 loss, Adam)
├── train_colorize.py              ← GAN training loop (L1 + adversarial + semantic)
├── inference.py                   ← End-to-end: raw 200m TIR → SR → colorized GeoTIFF
├── evaluate.py                    ← PSNR / SSIM / FID per tile
├── bonus_physics_informed.py      ← RadiativeMonotonicityLoss (optional bonus)
├── requirements.txt
└── tests/
    └── smoke_test.py              ← Torch-free unit tests
```

---

## 🏗 Architecture

### Pipeline Overview

```
Raw TIR B10 @200m
      │
      ▼  pad to 256-multiple (reflect)
┌─────────────────────────┐
│  TIRSuperResolutionNet  │  Stage 1 — Super Resolution
│  SRResNet-lite          │
│  785,921 params         │
│  Best val PSNR: 12.44dB │
└─────────────────────────┘
      │  (1, H×2, W×2) @100m   also saves → tir_superresolved_100m/<id>.tif
      ▼
┌─────────────────────────┐
│  UNetGenerator          │  Stage 2 — Colorization
│  9-level Pix2Pix GAN    │
│  54,402,627 params      │
│  Best val L1: 0.0751    │
└─────────────────────────┘
      │  crop to exact 2H×2W, denormalize
      ▼
  output/model_outputs/colorized_tir_100m/<id>.tif  (Blue·Green·Red)
```

### Stage 1 — TIRSuperResolutionNet (`models/sr_model.py`)

| Property | Value |
|----------|-------|
| Architecture | SRResNet-lite — 8 residual blocks + global skip connection |
| Input → Output | `(1, 256, 256)` TIR @200m → `(1, 512, 512)` TIR @100m |
| Parameters | 785,921 |
| Upsampling | Single-stage 2× PixelShuffle (sub-pixel convolution) |
| Loss | L1 pixel loss |
| Best val PSNR | **12.44 dB** |
| Training time | ~0.7 min on Kaggle T4 |

### Stage 2 — UNetGenerator + PatchGANDiscriminator (`models/colorization_model.py`)

| Property | Value |
|----------|-------|
| Architecture | 9-level U-Net encoder-decoder with skip connections |
| Input → Output | `(1, 512, 512)` TIR @100m → `(3, 512, 512)` RGB @100m |
| Generator params | 54,402,627 |
| Discriminator | PatchGAN (70×70 receptive field), 2,765,377 params |
| Loss | L1 (λ=100) + GAN adversarial + BandRatioConsistencyLoss |
| Best val L1 | **0.0751** (↓ 89% from epoch 1) |
| Training time | ~2.9 min on Kaggle T4 |

> **Why 9 levels?** driver.py outputs 512×512 patches. A 9-level U-Net collapses the
> bottleneck to exactly 1×1, giving correct information compression for 512px input.

---

## 🗄 Data Pipeline

The ISRO baseline repo `driver.py` generates all training patches from Landsat 9 bands:

```
input/<product_id>/
    demo_B2.tif, demo_B3.tif, demo_B4.tif   ← RGB bands @30m (resampled by USGS)
    demo_B10.tif                              ← TIR band @30m (resampled by USGS)
          │
          ▼  python driver.py
output/patches/<product_id>/sample_NNN/
    rgb_100m_512.npy    (3, 512, 512) uint16  ← Colorization target
    tir_100m_512.npy    (1, 512, 512) uint16  ← SR target / colorization input
    tir_200m.npy        (1, 256, 256) uint16  ← SR input
    *.png                                      ← Visualization only — do NOT train on these
```

**Rescaling factors:**
- RGB 30m → 100m: ×3.33 downscale  
- TIR 30m → 100m: ×3.33 downscale  
- TIR 30m → 200m: ×6.67 downscale  

**Co-registration guarantee:** 1 pixel @200m = exactly 2×2 block @100m.

---

## 🚀 Quickstart (Kaggle)

### 1. Open the Notebook

Open `arm2026.ipynb` on Kaggle with a **T4 GPU** accelerator.
Upload `ps10_project/` as a Kaggle Dataset.

| Cell | Purpose | Est. Time |
|------|---------|-----------|
| Cell 1 | Environment setup | ~1 min |
| Cell 2 | Install deps (`pip install -r requirements.txt`) | ~2 min |
| Cell 3 | Clone ISRO repo + extract project zip | ~1 min |
| Cell 4 (GEE) | Download Landsat 9 tiles via Google Earth Engine | ~5 min |
| Cell 5 | Run `driver.py` to generate patches | ~5 min |
| **Bulletproof Dataset Cell** | Build `aug_sr` + `aug_color` (16 effective patches each) | ~5 sec |
| **SR Training Cell** | Train `TIRSuperResolutionNet`, 20 epochs | ~1 min |
| **Color Training Cell** | Train `UNetGenerator` GAN, 40 epochs | ~3 min |
| **Inference Cell** | Run end-to-end on real B10 tile | ~30 sec |
| **Zip Cell** | Package outputs for download | ~5 sec |

### 2. Local Setup

```bash
git clone https://github.com/Mahatva777/ATX
cd ATX/ps10_project
pip install -r requirements.txt
```

### 3. Run Inference with Pre-trained Weights

Download weights from [Google Drive](https://drive.google.com/drive/folders/1TN0A_Vp4rMtlfKeF_baBdQPfBR0_GS4m?usp=sharing) into `checkpoints/`:

```bash
python inference.py \
    --input_tir   path/to/PRODUCT_ID_B10.tif \
    --sr_ckpt     checkpoints/sr_best.pth \
    --colorize_ckpt checkpoints/colorize_G_best.pth \
    --output_root output/model_outputs
```

---

## 📋 Mandatory Output Format

```
output/
└── model_outputs/
    ├── tir_superresolved_100m/
    │   └── <product_id>.tif       ← uint8, 1 band (TIR @100m)
    └── colorized_tir_100m/
        └── <product_id>.tif       ← uint8, 3 bands
```

> ⚠️ **Band order is strictly Band 1 = Blue, Band 2 = Green, Band 3 = Red**  
> `product_id` must exactly match the original input file stem.  
> Both outputs must be valid GeoTIFFs with correct CRS and affine transform.

---

## 📊 Training Results

| Metric | SR Model | Colorization Model |
|--------|----------|--------------------|
| Best val PSNR | **12.44 dB** | — |
| Best val L1 | — | **0.0751** |
| Epochs | 20 | 40 |
| Batch size | 8 | 8 |
| Learning rate | 2e-4 | 2e-4 |
| Optimizer | Adam (β=0.9, 0.999) | Adam (β=0.5, 0.999) |
| Training time | ~0.7 min (T4) | ~2.9 min (T4) |
| Real patches | 2 | 2 |
| Effective patches | 16 (8× aug) | 16 (8× aug) |

**Augmentation:** 4-way rotation × horizontal flip = 8× multiplier.

---

## 🔑 Key Engineering Decisions

1. **9-level U-Net** — `driver.py` outputs 512×512 patches (not 256×256 per baseline README). 9 levels → 1×1 bottleneck.

2. **Directory-based patch pairing** — `patch_dataset.py` groups files by parent directory, fixing the digit-suffix mismatch between `tir_200m.npy` and `tir_100m_512.npy`.

3. **Native 512px SR→colorize chain** — No resize between stages; colorization trained natively at 512×512 resolution.

4. **Pad-infer-crop** — Real tiles (e.g. 836×939 px) are reflection-padded to the nearest 256-multiple before U-Net inference, then cropped back to exact 2× input dimensions.

5. **CPU inference for large scenes** — Avoids T4 CUDA OOM on full satellite tiles (~30s on CPU for an 836×939 scene).

6. **Physics-informed bonus loss** — `bonus_physics_informed.py` contains `RadiativeMonotonicityLoss`. Wire into `train_colorize.py` with weight 1.0–3.0 for bonus credit.

---

## 🔬 Evaluate

```bash
python evaluate.py \
    --pred_dir output/model_outputs/colorized_tir_100m \
    --gt_dir   path/to/ground_truth_rgb_100m
```

Outputs per-tile PSNR, SSIM, and FID (FID requires `pytorch-fid`).

---

## 📁 Model Weights — Drive Contents

| File | Description |
|------|-------------|
| `sr_best.pth` | TIRSuperResolutionNet — best val PSNR checkpoint |
| `sr_last.pth` | TIRSuperResolutionNet — final epoch checkpoint |
| `colorize_G_best.pth` | UNetGenerator — best val L1 checkpoint |
| `colorize_G_last.pth` | UNetGenerator — final epoch checkpoint |
| `colorize_D_last.pth` | PatchGANDiscriminator — final epoch checkpoint |

All weights trained on real Landsat 9 data (Delhi/Rajasthan region, 2023).

---

## ⚙️ Requirements

```
torch>=2.0
torchvision
rasterio
numpy
scipy
tifffile
scikit-image
geemap
earthengine-api
```

Install: `pip install -r requirements.txt`
