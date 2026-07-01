"""
End-to-end inference: raw 200m TIR GeoTIFF -> SR model -> colorization
model -> writes outputs to the MANDATORY submission folder structure:

    output/
      model_outputs/
        tir_superresolved_100m/<product_id>.tif
        colorized_tir_100m/<product_id>.tif

Band order requirement (per README, checked literally by judges):
    colorized TIFF channel order = Blue, Green, Red  (NOT the usual RGB order)

Usage:
    python inference.py --input_tir path/to/PRODUCT_ID_B10_200m.tif \
        --sr_ckpt checkpoints/sr_best.pth --colorize_ckpt checkpoints/colorize_G_best.pth \
        --output_root output/model_outputs

Georeferencing: uses rasterio to read the input's CRS/transform and writes
it back onto both outputs (required so outputs are usable GIS layers, not
just images). Falls back to tifffile (no georeferencing) ONLY if rasterio
isn't installed, purely so this script's file-writing/band-order logic can
be smoke-tested in environments without GDAL -- do not submit outputs
produced by the fallback path, since they will lack a CRS.
"""
import argparse
import os
import torch.nn.functional as F

import numpy as np

try:
    import rasterio
    HAVE_RASTERIO = True
except ImportError:
    HAVE_RASTERIO = False
    import tifffile

try:
    import torch
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


def _normalize(arr: np.ndarray):
    arr = arr.astype(np.float32)
    lo, hi = np.percentile(arr, 1), np.percentile(arr, 99)
    if hi - lo < 1e-6:
        hi = lo + 1e-6
    norm = np.clip((arr - lo) / (hi - lo), 0, 1)
    return norm, lo, hi


def _denormalize01(x01: np.ndarray) -> np.ndarray:
    """Map a [0,1] float array to uint8 for GeoTIFF output."""
    return np.clip(x01 * 255.0, 0, 255).astype(np.uint8)


def _run_models(tir_tanh: np.ndarray, sr_model, colorize_model, device):
    """Torch-dependent step: runs the two models. Returns numpy arrays.
    sr_np: (H,W) tanh-range float32.  rgb_np: (3,H,W) tanh-range float32, RGB order.
    """
    with torch.no_grad():
        x = torch.from_numpy(tir_tanh).float().unsqueeze(0).unsqueeze(0).to(device)
        sr_out = sr_model(x)  # (1,1,512,512), tanh output
        sr_np = sr_out.squeeze().cpu().numpy()

        # Feed the super-resolved 512x512 TIR directly into the colorization model.
        # UNetGenerator is now a 9-level U-Net trained natively at 512x512
        # (matching the actual driver.py patch size), so no resize is needed.
        rgb_out = colorize_model(sr_out)  # (1,3,512,512), tanh output
        rgb_np = rgb_out.squeeze().cpu().numpy()  # (3,H,W) in RGB order
    return sr_np, rgb_np



def postprocess_and_write(sr_np: np.ndarray, rgb_np: np.ndarray, product_id: str,
                           output_root: str, profile=None, transform=None, crs=None):
    """
    Pure-numpy step (no torch dependency): denormalizes model outputs,
    reorders the colorized output to the REQUIRED Blue-Green-Red band
    order, and writes both GeoTIFFs to the mandatory folder structure.

    sr_np:  (H, W) tanh-range [-1, 1] float array.
    rgb_np: (3, H, W) tanh-range [-1, 1] float array, channel order R, G, B
            (i.e. the generator's natural output order -- the reorder to
            B, G, R happens inside this function, exactly once, so it
            can't be silently duplicated or skipped elsewhere).
    """
    sr_dir = os.path.join(output_root, "tir_superresolved_100m")
    color_dir = os.path.join(output_root, "colorized_tir_100m")
    os.makedirs(sr_dir, exist_ok=True)
    os.makedirs(color_dir, exist_ok=True)

    sr_uint8 = _denormalize01((sr_np + 1) / 2)
    sr_out_path = os.path.join(sr_dir, f"{product_id}.tif")
    _write_geotiff(sr_out_path, sr_uint8[None, ...], profile, transform, crs, band_names=["TIR"])

    rgb_uint8 = _denormalize01((rgb_np + 1) / 2)  # (3,H,W), currently R,G,B
    r, g, b = rgb_uint8[0], rgb_uint8[1], rgb_uint8[2]
    bgr_stack = np.stack([b, g, r], axis=0)  # REQUIRED band order: Blue, Green, Red
    color_out_path = os.path.join(color_dir, f"{product_id}.tif")
    _write_geotiff(color_out_path, bgr_stack, profile, transform, crs,
                    band_names=["Blue", "Green", "Red"])

    return sr_out_path, color_out_path


def run_inference(input_tir_path: str, sr_model, colorize_model, device, product_id: str,
                   output_root: str):
    if HAVE_RASTERIO:
        with rasterio.open(input_tir_path) as src:
            tir = src.read(1)
            profile = src.profile.copy()
            transform = src.transform
            crs = src.crs
    else:
        tir = tifffile.imread(input_tir_path)
        if tir.ndim == 3:
            tir = tir[..., 0]
        profile, transform, crs = None, None, None

    tir_norm01, lo, hi = _normalize(tir)
    tir_tanh = tir_norm01 * 2 - 1  # match dataset's [-1,1] normalization

    if HAVE_TORCH and sr_model is not None and colorize_model is not None:
        sr_np, rgb_np = _run_models(tir_tanh, sr_model, colorize_model, device)
    else:
        # No torch / no models available: identity stand-in so the
        # file-writing / band-order / folder-structure logic can still be
        # exercised end-to-end. NEVER use this path for a real submission.
        sr_np = tir_tanh
        rgb_np = np.stack([tir_tanh, tir_tanh, tir_tanh], axis=0)

    return postprocess_and_write(sr_np, rgb_np, product_id, output_root, profile, transform, crs)


def _write_geotiff(path, array_chw, profile, transform, crs, band_names):
    """array_chw: (C, H, W) uint8."""
    c, h, w = array_chw.shape
    if HAVE_RASTERIO and profile is not None:
        out_profile = profile.copy()
        out_profile.update(dtype="uint8", count=c, height=h, width=w,
                            transform=transform, crs=crs, driver="GTiff")
        with rasterio.open(path, "w", **out_profile) as dst:
            for i in range(c):
                dst.write(array_chw[i], i + 1)
                dst.set_band_description(i + 1, band_names[i])
    else:
        # tifffile fallback: (H, W, C), no georeferencing -- TEST ONLY
        tifffile.imwrite(path, np.moveaxis(array_chw, 0, -1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_tir", required=True)
    ap.add_argument("--sr_ckpt", default="checkpoints/sr_best.pth")
    ap.add_argument("--colorize_ckpt", default="checkpoints/colorize_G_best.pth")
    ap.add_argument("--output_root", default="output/model_outputs")
    ap.add_argument("--product_id", default=None,
                     help="Defaults to the input filename stem, matched to the required <product_id>.tif naming.")
    ap.add_argument("--device", default="cuda" if HAVE_TORCH and torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    product_id = args.product_id or os.path.splitext(os.path.basename(args.input_tir))[0]

    sr_model, colorize_model, device = None, None, args.device
    if HAVE_TORCH:
        from models.sr_model import TIRSuperResolutionNet
        from models.colorization_model import UNetGenerator
        device = torch.device(args.device)
        sr_model = TIRSuperResolutionNet().to(device)
        sr_model.load_state_dict(torch.load(args.sr_ckpt, map_location=device))
        sr_model.eval()
        colorize_model = UNetGenerator().to(device)
        colorize_model.load_state_dict(torch.load(args.colorize_ckpt, map_location=device))
        colorize_model.eval()

    sr_path, color_path = run_inference(args.input_tir, sr_model, colorize_model, device,
                                         product_id, args.output_root)
    print("Wrote:", sr_path)
    print("Wrote:", color_path)


if __name__ == "__main__":
    main()
