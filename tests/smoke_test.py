"""
Smoke tests for the parts of the pipeline that do NOT require torch/GPU,
run here with synthetic data standing in for real Landsat 9 patches (this
sandbox has no internet access to fetch real data and no torch installed).

What this validates:
  1. Patch dataset indexing/pairing logic (datasets/patch_dataset.py)
  2. Per-patch normalization behaves sanely
  3. inference.py's file-writing path: correct folder structure, correct
     <product_id>.tif naming, and CRITICALLY the Blue-Green-Red band order
     required by the submission spec
  4. evaluate.py's PSNR/SSIM computation against known synthetic images
     (including a sanity check: identical images -> PSNR = inf, SSIM = 1)

What this does NOT validate (needs torch + real data, run in Colab):
  - Actual model forward/backward passes (sr_model.py, colorization_model.py)
  - GAN training dynamics
  - FID (needs InceptionV3 via torch/torchvision)
  - Real georeferencing round-trip (needs rasterio; this sandbox lacks it,
    so inference.py's tifffile fallback path is exercised instead)
"""
import os
import shutil
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

FAIL = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail and not cond else ""))
    if not cond:
        FAIL.append(name)


# ---------------------------------------------------------------------
# 1. Generate synthetic patches mimicking driver.py's output convention,
#    split into separate sr/ and colorize/ subdirectories (see
#    datasets/patch_dataset.py docstring for why this separation matters:
#    SR-target and colorization-input are both "100m TIR" but at
#    DIFFERENT patch sizes, so they can't safely share one directory).
# ---------------------------------------------------------------------
SYN_ROOT = os.path.join(PROJECT_ROOT, "tests", "_synthetic_patches")
SR_DIR = os.path.join(SYN_ROOT, "sr")
COLOR_DIR = os.path.join(SYN_ROOT, "colorize")
if os.path.exists(SYN_ROOT):
    shutil.rmtree(SYN_ROOT)
os.makedirs(SR_DIR)
os.makedirs(COLOR_DIR)

N_PATCHES = 12
rng = np.random.default_rng(42)
for i in range(N_PATCHES):
    pid = f"{i:04d}"
    tir_200 = (rng.random((256, 256)) * 4000).astype(np.float32)      # SR input
    tir_100_sr_target = (rng.random((512, 512)) * 4000).astype(np.float32)  # SR target
    np.save(os.path.join(SR_DIR, f"tir_200m_{pid}.npy"), tir_200)
    np.save(os.path.join(SR_DIR, f"tir_100m_{pid}.npy"), tir_100_sr_target)

    tir_100_color_input = (rng.random((256, 256)) * 4000).astype(np.float32)  # colorization input
    rgb_100 = (rng.random((256, 256, 3)) * 255).astype(np.float32)    # colorization target
    np.save(os.path.join(COLOR_DIR, f"tir_100m_{pid}.npy"), tir_100_color_input)
    np.save(os.path.join(COLOR_DIR, f"rgb_100m_{pid}.npy"), rgb_100)

print("=" * 70)
print("TEST GROUP 1: dataset indexing/normalization (datasets/patch_dataset.py)")
print("=" * 70)

from datasets.patch_dataset import _normalize, _BasePatchIndex, SRPatchDataset, ColorizationPatchDataset

# normalization sanity
const_patch = np.full((32, 32), 500.0, dtype=np.float32)
norm_const = _normalize(const_patch)
check("normalize() handles constant patch without NaN/inf",
      np.isfinite(norm_const).all(), f"got range [{norm_const.min()}, {norm_const.max()}]")

# also check a LARGE-magnitude constant patch, since that's exactly what
# tripped the original float32-epsilon bug (raw DN radiance in the thousands)
const_patch_large = np.full((32, 32), 50000.0, dtype=np.float32)
norm_const_large = _normalize(const_patch_large)
check("normalize() handles large-magnitude constant patch without NaN/inf",
      np.isfinite(norm_const_large).all(), f"got range [{norm_const_large.min()}, {norm_const_large.max()}]")

varied_patch = rng.random((64, 64)).astype(np.float32) * 1000
norm_varied = _normalize(varied_patch)
check("normalize() maps varied patch into [-1, 1]",
      norm_varied.min() >= -1.001 and norm_varied.max() <= 1.001,
      f"got range [{norm_varied.min():.3f}, {norm_varied.max():.3f}]")

# indexing: colorization pairing (tir_100m_XXXX <-> rgb_100m_XXXX) in COLOR_DIR
idx = _BasePatchIndex(COLOR_DIR)
color_pairs = idx._index("tir_100m_*.npy", "rgb_100m_*.npy")
check("colorization pair indexing finds all synthetic patches",
      len(color_pairs) == N_PATCHES, f"found {len(color_pairs)}, expected {N_PATCHES}")

if color_pairs:
    tir_path, rgb_path = color_pairs[0]
    tir_arr = np.load(tir_path)
    rgb_arr = np.load(rgb_path)
    check("colorization pair shapes match spec (256x256 TIR, 256x256x3 RGB)",
          tir_arr.shape == (256, 256) and rgb_arr.shape == (256, 256, 3),
          f"got TIR {tir_arr.shape}, RGB {rgb_arr.shape}")

# indexing: SR pairing (tir_200m_XXXX <-> tir_100m_XXXX) in SR_DIR
sr_idx = _BasePatchIndex(SR_DIR)
sr_pairs = sr_idx._index("*200m*.npy", "*100m*.npy")
check("SR pair indexing finds all synthetic patches",
      len(sr_pairs) == N_PATCHES, f"found {len(sr_pairs)}, expected {N_PATCHES}")
if sr_pairs:
    low_path, high_path = sr_pairs[0]
    low_arr, high_arr = np.load(low_path), np.load(high_path)
    check("SR pair shapes match spec (256x256 input, 512x512 target)",
          low_arr.shape == (256, 256) and high_arr.shape == (512, 512),
          f"got low {low_arr.shape}, high {high_arr.shape}")

# End-to-end dataset class test (this is what train_sr.py / train_colorize.py
# actually instantiate) -- note: __getitem__ needs torch, so we only test
# __init__/indexing here, which is exactly what works without torch installed.
try:
    sr_ds = SRPatchDataset(SR_DIR)
    check("SRPatchDataset() constructs successfully against synthetic sr/ dir",
          len(sr_ds) == N_PATCHES, f"len={len(sr_ds)}")
except Exception as e:
    check("SRPatchDataset() constructs successfully against synthetic sr/ dir", False, str(e))

try:
    color_ds = ColorizationPatchDataset(COLOR_DIR)
    check("ColorizationPatchDataset() constructs successfully against synthetic colorize/ dir",
          len(color_ds) == N_PATCHES, f"len={len(color_ds)}")
except Exception as e:
    check("ColorizationPatchDataset() constructs successfully against synthetic colorize/ dir", False, str(e))

# Cross-check: pointing SRPatchDataset at the COLORIZATION directory should
# fail LOUDLY (shape mismatch) rather than silently train on wrong data.
try:
    SRPatchDataset(COLOR_DIR)
    check("SRPatchDataset raises when pointed at wrong (colorization) directory", False,
          "did not raise -- this would silently corrupt training")
except RuntimeError:
    check("SRPatchDataset raises when pointed at wrong (colorization) directory", True)

# ---------------------------------------------------------------------
# 2. inference.py file-writing / folder-structure / band-order compliance
# ---------------------------------------------------------------------
print()
print("=" * 70)
print("TEST GROUP 2: inference.py output structure + band order (no-torch fallback)")
print("=" * 70)

# create a synthetic single-band "raw TIR" tif input using tifffile
import tifffile as tiff_mod

INPUT_TIR_PATH = os.path.join(SYN_ROOT, "LC09_TEST_PRODUCT_001.tif")
raw_tir = (rng.random((256, 256)) * 4000).astype(np.float32)
tiff_mod.imwrite(INPUT_TIR_PATH, raw_tir)

OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "tests", "_synthetic_output", "model_outputs")
if os.path.exists(os.path.dirname(OUTPUT_ROOT)):
    shutil.rmtree(os.path.dirname(OUTPUT_ROOT))

import inference as inference_mod

sr_path, color_path = inference_mod.run_inference(
    INPUT_TIR_PATH, sr_model=None, colorize_model=None, device=None,
    product_id="LC09_TEST_PRODUCT_001", output_root=OUTPUT_ROOT,
)

check("SR output written to mandatory tir_superresolved_100m/ folder",
      os.path.normpath(sr_path) == os.path.normpath(
          os.path.join(OUTPUT_ROOT, "tir_superresolved_100m", "LC09_TEST_PRODUCT_001.tif")))
check("Colorized output written to mandatory colorized_tir_100m/ folder",
      os.path.normpath(color_path) == os.path.normpath(
          os.path.join(OUTPUT_ROOT, "colorized_tir_100m", "LC09_TEST_PRODUCT_001.tif")))
check("Output filename matches input product_id exactly",
      os.path.basename(sr_path) == "LC09_TEST_PRODUCT_001.tif")
check("SR output file exists on disk", os.path.exists(sr_path))
check("Colorized output file exists on disk", os.path.exists(color_path))

# ---- verify band order is Blue, Green, Red as required ----
color_arr = tiff_mod.imread(color_path)  # written as (H, W, C) by the fallback path
check("Colorized output has 3 bands", color_arr.ndim == 3 and color_arr.shape[-1] == 3,
      f"got shape {color_arr.shape}")

# Because the no-torch fallback feeds the SAME normalized TIR into all 3
# "RGB" channels before the B,G,R reorder is applied, we instead verify
# band order by calling postprocess_and_write() directly with a distinctive
# fake generator output -- this is the pure-numpy function that owns the
# reorder, so it's testable with no torch dependency at all.
print("\n  Calling postprocess_and_write() directly with a distinctive fake color output...")

# distinctive RGB generator output: R channel = all 0.9, G = all 0.5, B = all -0.9 (tanh range)
fake_rgb = np.stack([
    np.full((256, 256), 0.9, dtype=np.float32),   # R
    np.full((256, 256), 0.5, dtype=np.float32),   # G
    np.full((256, 256), -0.9, dtype=np.float32),  # B
], axis=0)
fake_sr = np.zeros((512, 512), dtype=np.float32)

sr_path2, color_path2 = inference_mod.postprocess_and_write(
    fake_sr, fake_rgb, product_id="LC09_BANDORDER_CHECK", output_root=OUTPUT_ROOT,
)

reordered = tiff_mod.imread(color_path2)  # (H, W, 3) via fallback writer
# fallback writer does np.moveaxis(array_chw, 0, -1) on the (3,H,W) BGR stack.
# The reorder REPOSITIONS channels, it doesn't swap values -- so disk band 0
# (the "Blue" slot) must contain the generator's actual B-channel VALUE
# (-0.9, dark), disk band 1 ("Green" slot) the G-channel value (0.5, mid),
# and disk band 2 ("Red" slot) the R-channel value (0.9, bright).
disk_band0, disk_band1, disk_band2 = reordered[..., 0], reordered[..., 1], reordered[..., 2]
check("Band-order reorder: disk band 0 (Blue slot) holds generator's B=-0.9 value (dark)",
      disk_band0.mean() < 50, f"mean={disk_band0.mean():.1f}")
check("Band-order reorder: disk band 1 (Green slot) holds generator's G=0.5 value (mid)",
      100 < disk_band1.mean() < 220, f"mean={disk_band1.mean():.1f}")
check("Band-order reorder: disk band 2 (Red slot) holds generator's R=0.9 value (bright)",
      disk_band2.mean() > 200, f"mean={disk_band2.mean():.1f}")

# ---------------------------------------------------------------------
# 3. evaluate.py PSNR/SSIM correctness
# ---------------------------------------------------------------------
print()
print("=" * 70)
print("TEST GROUP 3: evaluate.py PSNR/SSIM (skimage, no torch needed)")
print("=" * 70)

EVAL_PRED_DIR = os.path.join(PROJECT_ROOT, "tests", "_eval_pred")
EVAL_GT_DIR = os.path.join(PROJECT_ROOT, "tests", "_eval_gt")
for d in (EVAL_PRED_DIR, EVAL_GT_DIR):
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d)

# case A: identical images -> PSNR should be very high (~inf, skimage caps it), SSIM ~1
identical_img = (rng.random((128, 128, 3)) * 255).astype(np.uint8)
tiff_mod.imwrite(os.path.join(EVAL_PRED_DIR, "TILE_A.tif"), identical_img)
tiff_mod.imwrite(os.path.join(EVAL_GT_DIR, "TILE_A.tif"), identical_img)

# case B: noisy version -> PSNR/SSIM should be lower than case A but finite
noisy_img = np.clip(identical_img.astype(np.int16) + rng.integers(-40, 40, identical_img.shape), 0, 255).astype(np.uint8)
tiff_mod.imwrite(os.path.join(EVAL_PRED_DIR, "TILE_B.tif"), noisy_img)
tiff_mod.imwrite(os.path.join(EVAL_GT_DIR, "TILE_B.tif"), identical_img)

import evaluate as evaluate_mod

results = evaluate_mod.compute_psnr_ssim(EVAL_PRED_DIR, EVAL_GT_DIR)
by_id = {r["product_id"]: r for r in results}

check("evaluate.py finds both matched pred/gt pairs", len(results) == 2, f"got {len(results)}")
check("Identical images score near-maximum PSNR (>45dB, capped by skimage near-inf)",
      by_id.get("TILE_A", {}).get("psnr", 0) > 45,
      f"got {by_id.get('TILE_A', {}).get('psnr')}")
check("Identical images score SSIM ~1.0", by_id.get("TILE_A", {}).get("ssim", 0) > 0.99,
      f"got {by_id.get('TILE_A', {}).get('ssim')}")
check("Noisy image scores strictly lower PSNR than identical image",
      by_id.get("TILE_B", {}).get("psnr", 999) < by_id.get("TILE_A", {}).get("psnr", 0))
check("Noisy image scores strictly lower SSIM than identical image",
      by_id.get("TILE_B", {}).get("ssim", 999) < by_id.get("TILE_A", {}).get("ssim", 0))

# ---------------------------------------------------------------------
# 4. Syntax-check torch-dependent files (can't execute without torch, but
#    can confirm they at least parse cleanly)
# ---------------------------------------------------------------------
print()
print("=" * 70)
print("TEST GROUP 4: syntax check of torch-dependent files (py_compile)")
print("=" * 70)
import py_compile

torch_files = [
    "models/sr_model.py", "models/colorization_model.py",
    "models/semantic_constraint.py", "train_sr.py", "train_colorize.py",
    "bonus_physics_informed.py",
]
for f in torch_files:
    path = os.path.join(PROJECT_ROOT, f)
    try:
        py_compile.compile(path, doraise=True)
        check(f"{f} compiles (syntax valid)", True)
    except py_compile.PyCompileError as e:
        check(f"{f} compiles (syntax valid)", False, str(e))

# ---------------------------------------------------------------------
print()
print("=" * 70)
if FAIL:
    print(f"RESULT: {len(FAIL)} test(s) FAILED: {FAIL}")
    sys.exit(1)
else:
    print("RESULT: ALL SMOKE TESTS PASSED")
