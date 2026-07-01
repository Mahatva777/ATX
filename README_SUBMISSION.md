# PS10 — IR Colorization & Enhancement: Training/Inference Scaffold

Built against ISRO BAH2026 PS10 (`jugal-sac/IR-colorization-BAH2026`).
Everything in this folder was written and **smoke-tested with synthetic
data** in a sandboxed environment with no GPU and no internet access — see
`TEST_REPORT.md` for exactly what was and wasn't verified, and what you
must re-check the moment you have real data + a GPU.

## 0. Known open item before you start (read this first)

The README's dataset spec says:
```
SR pair:            256x256 (200m TIR)  ->  512x512 (100m TIR)
Colorization pair:  256x256 (100m TIR)   ->  256x256 (100m RGB)
```
Both the SR-stage *target* and the colorization-stage *input* are "100m
TIR" — but at **different patch sizes** (512x512 vs 256x256). That means
you cannot feed the SR model's raw output straight into the colorization
model without a resize/crop step. `inference.py` has this flagged
explicitly at the `color_in = sr_out` line. Decide early whether you'll
downsample the SR output 512->256 before colorizing, or retrain
colorization at 512x512, and update that line accordingly — this is the
single most important design decision left open in this scaffold.

## 1. Setup (run in Colab / Kaggle, NOT here)

```bash
pip install -r requirements.txt
```

## 2. Generate patches

Run the original repo's `driver.py` (fork of `jugal-sac/IR-colorization-BAH2026`)
against real Landsat 9 tiles downloaded per its README. Once you can see
the actual output filenames in `output/patches/`, **split them into two
directories** before training (see `datasets/patch_dataset.py` docstring
for why):

```
output/patches/sr/         <- only 200m/100m TIR pairs (SR patches)
output/patches/colorize/   <- only 100m TIR (256) + RGB patches (colorization patches)
```

Then sanity-check indexing before burning GPU time:
```bash
python datasets/patch_dataset.py output/patches/sr output/patches/colorize
```
If it reports 0 pairs or a shape-mismatch error, fix the glob patterns /
directory split in `patch_dataset.py` — don't proceed to training until
this prints the expected pair counts and shapes.

## 3. Train

```bash
python train_sr.py --patches_dir output/patches/sr --epochs 30 --batch_size 8
python train_colorize.py --patches_dir output/patches/colorize --epochs 60 --batch_size 8
```
Checkpoints land in `checkpoints/{sr,colorize_G,colorize_D}_{last,best}.pth`.
Adjust `--epochs`/`--batch_size` down if you hit time or VRAM limits — see
the "JUDGMENT CALL" comments in both scripts.

## 4. Run inference (produces the mandatory submission structure)

```bash
python inference.py --input_tir path/to/PRODUCT_ID_B10_200m.tif \
    --sr_ckpt checkpoints/sr_best.pth \
    --colorize_ckpt checkpoints/colorize_G_best.pth \
    --output_root output/model_outputs
```
Writes:
```
output/model_outputs/
  tir_superresolved_100m/<product_id>.tif
  colorized_tir_100m/<product_id>.tif      <- band order forced to Blue, Green, Red
```

## 5. Evaluate

```bash
python evaluate.py --pred_dir output/model_outputs/colorized_tir_100m \
    --gt_dir path/to/ground_truth_rgb_100m
```
Prints per-tile PSNR/SSIM/FID plus means. FID needs `pytorch-fid` + torch
(so it only runs in the Colab/Kaggle environment, not in a torch-less
sandbox).

## 6. Bonus: physics-informed loss

`bonus_physics_informed.py` has a documented, ready-to-wire-in
`RadiativeMonotonicityLoss`. Add it into `train_colorize.py`'s generator
loss with a small weight (1.0–3.0) if you have time left — it's the exact
thing the mentor transcript calls out for bonus points.

## Repo layout
```
models/
  sr_model.py              -- TIRSuperResolutionNet (compact residual SR net)
  colorization_model.py    -- UNetGenerator + PatchGANDiscriminator (Pix2Pix)
  semantic_constraint.py   -- BandRatioConsistencyLoss (default) / LandCoverClassifierLoss (optional)
datasets/
  patch_dataset.py         -- SRPatchDataset, ColorizationPatchDataset
train_sr.py                 -- SR training loop
train_colorize.py           -- Colorization GAN training loop
inference.py                 -- End-to-end raw TIR -> mandatory output structure
evaluate.py                  -- PSNR / SSIM / FID / timing
bonus_physics_informed.py    -- Optional radiative-consistency loss
tests/smoke_test.py          -- Torch-free tests, see TEST_REPORT.md
requirements.txt
```
