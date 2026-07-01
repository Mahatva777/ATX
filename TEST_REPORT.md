# Test Report

Environment this code was built and tested in: **no internet access, no
GPU, no `torch`/`rasterio` installed.** So "tested" here means something
specific — read this before trusting any of it blindly.

## What was actually run and passed (24/24 checks, `tests/smoke_test.py`)

1. **Patch-pairing logic** (`datasets/patch_dataset.py`)
   - Found a real bug: the patch-ID regex grabbed the *first* digit run in
     a filename (e.g. the "100" in `tir_100m_0007.npy`), not the patch
     index — collapsing all patches onto one fake "id". Fixed by anchoring
     to the trailing digit run.
   - Found a real bug: per-patch normalization divided by zero (→ NaN) for
     constant/near-constant patches with large radiance values, because
     a `+1e-6` epsilon was silently swallowed by float32 rounding at
     magnitude ~10³–10⁵. Fixed with a relative epsilon computed in
     float64.
   - Added a design fix, not just a bug fix: SR-target and
     colorization-input patches are both "100m TIR" but at different
     patch sizes (512² vs 256²) per the README — sharing one directory
     made them ambiguous by filename alone. Split into `sr/` and
     `colorize/` subdirectories, with an explicit shape check that raises
     immediately if a dataset is pointed at the wrong folder.
   - Verified: correct pair counts, correct shapes, and the
     wrong-directory case actually raises instead of silently training on
     bad data.

2. **`inference.py` output compliance**
   - Verified the mandatory folder structure
     (`tir_superresolved_100m/`, `colorized_tir_100m/`) and
     `<product_id>.tif` naming are produced exactly as specified.
   - Verified the **Blue-Green-Red band order** requirement with a
     targeted test using a distinctive fake RGB array (R=bright,
     G=mid, B=dark) — confirmed disk band 0 actually contains the
     generator's B value, not a channel swap error. (My first version of
     this test had the assertion backwards; fixed and re-verified.)
   - Refactored `run_inference` so the file-writing/reorder logic is a
     pure-numpy function (`postprocess_and_write`) independent of torch —
     this was necessary because the original version couldn't be
     exercised at all without torch installed (a `NameError` crashed it).
   - Confirmed the resolution mismatch flagged in the code (SR outputs
     512² but colorization was trained at 256²) is real and that
     `evaluate.py` correctly *skips with a warning* rather than crashing
     or silently miscomputing when shapes don't match — but this mismatch
     still needs a design decision from you (resize/crop strategy) before
     the two stages can be chained in `inference.py`.

3. **`evaluate.py` PSNR/SSIM** (scikit-image, no torch needed)
   - Verified against synthetic images: identical images score
     PSNR > 45dB and SSIM ≈ 1.0; a noised version scores strictly lower on
     both metrics; mismatched-shape pairs are skipped with a warning
     rather than crashing.

4. **Model architecture shape audits** (manual arithmetic, since torch
   isn't installed here to actually run a forward pass)
   - `UNetGenerator`: traced every encoder/decoder level's channel count
     by hand and confirmed every skip-connection concat produces exactly
     the channel count the next `ConvTranspose2d` declares — an 8-level
     Pix2Pix U-Net is easy to get one level off. This was correct on
     first pass, no bug found.
   - `TIRSuperResolutionNet`: confirmed the single PixelShuffle(2) stage
     takes `base*4` channels to `base` channels at 2x spatial resolution,
     landing on exactly (B,1,512,512) from a (B,1,256,256) input.
   - `PatchGANDiscriminator`: traced the conv arithmetic to confirm a
     ~30×30 output grid for a 256×256 input, consistent with the standard
     70×70-receptive-field PatchGAN design.
   - All `torch`-dependent files were also syntax-checked with
     `py_compile` (catches typos/syntax errors, not logic errors).

## What was NOT tested here (needs Colab/Kaggle GPU + real data)

- Actual forward/backward passes and gradient flow through any model —
  the shape audits above are arithmetic, not execution.
- GAN training stability (loss balance between G/D, mode collapse risk).
- FID computation (`pytorch-fid` requires torch + torchvision's
  InceptionV3 weights, unavailable here).
- Real GeoTIFF georeferencing round-trip via `rasterio` (not installed
  here; `inference.py`'s tifffile fallback path was exercised instead,
  which explicitly does NOT preserve CRS — do not submit outputs produced
  by that fallback path).
- Real driver.py output filenames — `datasets/patch_dataset.py`'s glob
  patterns are informed guesses from the README's prose description, not
  the actual script (I didn't have `driver.py`'s source, only the README).
  **Run `python datasets/patch_dataset.py <sr_dir> <colorize_dir>` against
  your real patches before training and fix the globs if it reports 0
  pairs.**
- Actual PSNR/SSIM/FID numbers on real imagery — the metric *code* is
  verified correct, but real scores depend entirely on real training.

## Bottom line

The scaffolding logic (data pairing, output structure, band order,
evaluation) has been exercised and had 3 real bugs found and fixed. The
model architectures are dimensionally verified by hand but never executed.
Budget time in Colab for: (1) confirming `patch_dataset.py`'s globs
against real driver.py filenames, (2) deciding the 512→256 resolution
hand-off between stages, (3) an actual short training run to confirm
loss goes down before committing your full 30-hour budget to it.
