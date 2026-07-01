"""
Evaluation: PSNR, SSIM, FID between model outputs and ground-truth RGB,
plus per-tile inference wall-clock time.

Usage:
    python evaluate.py --pred_dir output/model_outputs/colorized_tir_100m \
        --gt_dir path/to/ground_truth_rgb_100m --timing_log timing.csv

PSNR/SSIM use scikit-image (no torch dependency -- works in any environment).
FID uses pytorch-fid and therefore needs torch + torchvision (Inception v3
weights); it is skipped with a clear message if unavailable so the rest of
the evaluation still runs.
"""
import argparse
import glob
import os
import time

import numpy as np
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim

try:
    import tifffile
except ImportError:
    tifffile = None


def load_image(path):
    if tifffile is not None and path.lower().endswith((".tif", ".tiff")):
        arr = tifffile.imread(path)
    else:
        from PIL import Image
        arr = np.array(Image.open(path))
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[0] < arr.shape[-1]:
        arr = np.moveaxis(arr, 0, -1)  # CHW -> HWC
    return arr


def compute_psnr_ssim(pred_dir: str, gt_dir: str):
    pred_files = sorted(glob.glob(os.path.join(pred_dir, "*.tif")) +
                         glob.glob(os.path.join(pred_dir, "*.tiff")))
    results = []
    for pred_path in pred_files:
        product_id = os.path.splitext(os.path.basename(pred_path))[0]
        gt_candidates = glob.glob(os.path.join(gt_dir, f"{product_id}.*"))
        if not gt_candidates:
            print(f"  [skip] no ground truth found for {product_id}")
            continue
        gt_path = gt_candidates[0]

        pred = load_image(pred_path).astype(np.float64)
        gt = load_image(gt_path).astype(np.float64)
        if pred.shape != gt.shape:
            print(f"  [warn] shape mismatch for {product_id}: pred {pred.shape} vs gt {gt.shape} -- skipping")
            continue

        data_range = 255.0 if pred.max() > 1.5 else 1.0
        psnr_val = sk_psnr(gt, pred, data_range=data_range)
        channel_axis = 2 if pred.ndim == 3 else None
        ssim_val = sk_ssim(gt, pred, data_range=data_range, channel_axis=channel_axis)
        results.append({"product_id": product_id, "psnr": psnr_val, "ssim": ssim_val})
    return results


def compute_fid(pred_dir: str, gt_dir: str):
    try:
        from pytorch_fid.fid_score import calculate_fid_given_paths
    except ImportError:
        print("  [skip] pytorch-fid / torch not installed here -- run this "
              "part in Colab where torch is available.")
        return None
    fid_value = calculate_fid_given_paths([pred_dir, gt_dir], batch_size=8,
                                           device="cuda", dims=2048)
    return fid_value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--timing_log", default=None,
                     help="Optional CSV to append a per-tile inference-time summary to.")
    args = ap.parse_args()

    print("Computing PSNR / SSIM ...")
    results = compute_psnr_ssim(args.pred_dir, args.gt_dir)
    if results:
        mean_psnr = float(np.mean([r["psnr"] for r in results]))
        mean_ssim = float(np.mean([r["ssim"] for r in results]))
        print(f"\n{'product_id':30s} {'PSNR (dB)':>10s} {'SSIM':>8s}")
        for r in results:
            print(f"{r['product_id']:30s} {r['psnr']:10.2f} {r['ssim']:8.4f}")
        print(f"\nMean PSNR: {mean_psnr:.2f} dB   Mean SSIM: {mean_ssim:.4f}   (n={len(results)})")
    else:
        print("No matched pred/gt pairs found -- check --pred_dir / --gt_dir naming.")

    print("\nComputing FID ...")
    fid_val = compute_fid(args.pred_dir, args.gt_dir)
    if fid_val is not None:
        print(f"FID: {fid_val:.3f}")


if __name__ == "__main__":
    main()
