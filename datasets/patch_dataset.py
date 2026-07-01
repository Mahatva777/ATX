"""
Dataset classes reading the .npy patches produced by the original repo's
driver.py (see output/patches/ per the README).

IMPORTANT: the README describes the *shapes and pairing* precisely:
    SR pair:            256x256 (200m TIR)  ->  512x512 (100m TIR)
    Colorization pair:  256x256 (100m TIR)   ->  256x256 (100m RGB)
but does not give exact filenames -- driver.py wasn't in the files I had
access to. This loader assumes a reasonably conventional naming scheme
(one .npy per patch, per band/resolution, sharing a common patch ID) and
is written so the glob patterns in `_index()` are the ONE place you need
to edit once you've actually run driver.py and can see real filenames in
output/patches/.

ASSUMPTION THAT NEEDS VERIFYING: SR-target patches and colorization-input
patches are BOTH "100m TIR" but at DIFFERENT patch sizes (512x512 vs
256x256) per the README. If driver.py saves them into the same flat
directory with similar names, a naive glob can't tell them apart by name
alone. This loader therefore (a) expects SR patches and colorization
patches to live in separate subdirectories -- pass --sr_patches_dir and
--colorize_patches_dir separately once you've run driver.py and can see
its real output layout -- and (b) validates array shape against the
expected size on load and raises immediately if it doesn't match, so a
mis-pointed directory fails loudly instead of silently training on the
wrong pairs.

Run `python datasets/patch_dataset.py <sr_dir> <colorize_dir>` after
generating real patches to sanity-check indexing before training.
"""
import glob
import os
import re
from typing import List, Tuple

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # allows the indexing/shape logic to be smoke-tested without torch
    torch = None
    Dataset = object


def _normalize(arr: np.ndarray) -> np.ndarray:
    """Min-max normalize a patch to [-1, 1]. Radiometric range is scene-
    dependent, so per-patch normalization is used (documented tradeoff:
    this discards absolute radiance information the physics-informed loss
    could otherwise use -- see bonus_physics_informed.py for a variant
    that keeps raw radiance alongside the normalized tensor)."""
    arr = arr.astype(np.float64)  # float64 avoids float32-precision loss in the
    # near-constant-patch epsilon guard below (a small additive epsilon can be
    # silently swallowed by float32 rounding when lo/hi are large, e.g. raw
    # digital-number radiance values in the thousands)
    lo, hi = np.percentile(arr, 1), np.percentile(arr, 99)
    span = hi - lo
    if span < 1e-6:
        # relative epsilon: safe even when lo/hi are large radiance values
        eps = max(1e-6, abs(lo) * 1e-6, 1.0)
        hi = lo + eps
    arr = np.clip(arr, lo, hi)
    arr = 2 * (arr - lo) / (hi - lo) - 1
    return arr.astype(np.float32)


class _BasePatchIndex:
    """Shared file-discovery logic. EDIT THE GLOB PATTERNS to match your
    actual driver.py output once you've run it."""

    def __init__(self, patches_dir: str):
        self.patches_dir = patches_dir

    def _index(self, low_glob: str, high_glob: str) -> List[Tuple[str, str]]:
        # Recursive search: finds files in both the root dir and any
        # subdirectories (e.g. output/patches/tile_SW/, tile_NW/, ...).
        low_files = sorted(set(
            glob.glob(os.path.join(self.patches_dir, low_glob)) +
            glob.glob(os.path.join(self.patches_dir, "**", low_glob), recursive=True)
        ))
        high_files = sorted(set(
            glob.glob(os.path.join(self.patches_dir, high_glob)) +
            glob.glob(os.path.join(self.patches_dir, "**", high_glob), recursive=True)
        ))

        def digit_id(path: str) -> str:
            """Trailing digit run of the filename stem, or '' if none exists."""
            stem = os.path.splitext(os.path.basename(path))[0]
            m = re.search(r"(\d+)$", stem)
            return m.group(1) if m else ""

        # Group by parent directory: driver.py puts each tile's patch set
        # into one subdirectory, so same-dir files are spatially matched.
        from collections import defaultdict
        low_by_dir:  dict = defaultdict(list)
        high_by_dir: dict = defaultdict(list)
        for f in low_files:
            low_by_dir[os.path.dirname(os.path.abspath(f))].append(f)
        for f in high_files:
            high_by_dir[os.path.dirname(os.path.abspath(f))].append(f)

        pairs: List[Tuple[str, str]] = []
        for d in sorted(set(low_by_dir) & set(high_by_dir)):
            lows  = low_by_dir[d]
            highs = high_by_dir[d]

            if len(lows) == 1 and len(highs) == 1:
                # Exactly one of each in this directory (driver.py per-tile
                # output: tir_200m.npy + tir_100m_512.npy). Pair directly
                # regardless of digit suffix — they're the only match possible.
                pairs.append((lows[0], highs[0]))
            else:
                # Multiple files per directory (flat numbered layout e.g.
                # tir_200m_0000.npy, tir_200m_0001.npy, ...).
                # Match by trailing digit run as original logic.
                low_by_id  = {digit_id(f): f for f in lows}
                high_by_id = {digit_id(f): f for f in highs}
                common = sorted(set(low_by_id) & set(high_by_id),
                                key=lambda x: (len(x), x))
                pairs.extend([(low_by_id[k], high_by_id[k]) for k in common])

        return pairs


class SRPatchDataset(_BasePatchIndex, Dataset):
    """
    Yields (tir_200m, tir_100m) pairs, shapes (1,256,256) and (1,512,512),
    both normalized to [-1, 1].

    Actual driver.py filenames: tir_200m.npy (256x256) + tir_100m_512.npy (512x512).
    The glob patterns below are written to match this naming exactly.
    """

    def __init__(self, patches_dir: str,
                 low_glob: str = "*tir*200m*.npy",
                 high_glob: str = "*tir*100m*.npy",
                 expected_low_shape=(256, 256), expected_high_shape=(512, 512)):
        _BasePatchIndex.__init__(self, patches_dir)
        self.pairs = self._index(low_glob, high_glob)
        if len(self.pairs) == 0:
            raise RuntimeError(
                f"No SR pairs found in {patches_dir} with patterns "
                f"'{low_glob}' / '{high_glob}'. Check driver.py's actual "
                f"output filenames and adjust the glob patterns."
            )
        # Fail loudly if this directory holds the wrong patch type.
        # Use [-2:] to compare only spatial dims (H, W) — ignores the
        # channel dim so (1,256,256) and (256,256) both pass for (256,256).
        low0  = np.load(self.pairs[0][0]).shape
        high0 = np.load(self.pairs[0][1]).shape
        if tuple(low0[-2:]) != expected_low_shape or tuple(high0[-2:]) != expected_high_shape:
            raise RuntimeError(
                f"SRPatchDataset: expected spatial shapes {expected_low_shape} -> "
                f"{expected_high_shape} but found {low0} -> {high0} in "
                f"{patches_dir}. Are sr_patches_dir and "
                f"colorize_patches_dir pointed at the right folders?"
            )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        low_path, high_path = self.pairs[idx]
        low = _normalize(np.load(low_path))
        high = _normalize(np.load(high_path))
        low_t = torch.from_numpy(low).unsqueeze(0) if low.ndim == 2 else torch.from_numpy(low)
        high_t = torch.from_numpy(high).unsqueeze(0) if high.ndim == 2 else torch.from_numpy(high)
        return low_t, high_t


class ColorizationPatchDataset(_BasePatchIndex, Dataset):
    """
    Yields (tir_100m, rgb_100m) pairs.

    Actual driver.py output: tir_100m_512.npy (1,512,512) and
    rgb_100m_512.npy (3,512,512). Despite the README describing 256x256
    colorization patches, the real driver produces 512x512 for both TIR
    and RGB at 100m. The colorization model (UNetGenerator) is trained at
    this native 512x512 size accordingly.
    """

    def __init__(self, patches_dir: str,
                 tir_glob: str = "*tir*100m*.npy",
                 rgb_glob: str = "*rgb*100m*.npy",
                 expected_tir_shape=(512, 512)):
        _BasePatchIndex.__init__(self, patches_dir)
        self.pairs = self._index(tir_glob, rgb_glob)
        if len(self.pairs) == 0:
            raise RuntimeError(
                f"No colorization pairs found in {patches_dir} with patterns "
                f"'{tir_glob}' / '{rgb_glob}'. Check driver.py's actual "
                f"output filenames and adjust the glob patterns."
            )
        tir0 = np.load(self.pairs[0][0]).shape
        # Compare only spatial dims [-2:] so (1,512,512) passes for (512,512).
        tir_spatial = tuple(tir0[-2:])
        valid_shapes = {expected_tir_shape, (256, 256), (512, 512)}
        if tir_spatial not in valid_shapes:
            raise RuntimeError(
                f"ColorizationPatchDataset: unexpected TIR spatial shape {tir_spatial} "
                f"(full shape {tir0}) in {patches_dir}. "
                f"Expected one of {valid_shapes}. "
                f"Check --colorize_patches_dir is pointed at the correct folder."
            )
        self.tir_hw = tir_spatial  # store actual HxW for model config

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        tir_path, rgb_path = self.pairs[idx]
        tir = _normalize(np.load(tir_path))
        rgb = np.load(rgb_path).astype(np.float32)
        # rgb patches are expected as (H, W, 3) from driver.py; normalize per-channel
        if rgb.ndim == 3 and rgb.shape[-1] == 3:
            rgb = np.stack([_normalize(rgb[..., c]) for c in range(3)], axis=0)  # -> (3,H,W)
        elif rgb.ndim == 3 and rgb.shape[0] == 3:
            rgb = np.stack([_normalize(rgb[c]) for c in range(3)], axis=0)
        else:
            raise ValueError(f"Unexpected RGB patch shape {rgb.shape} in {rgb_path}")

        tir_t = torch.from_numpy(tir).unsqueeze(0) if tir.ndim == 2 else torch.from_numpy(tir)
        rgb_t = torch.from_numpy(rgb)
        return tir_t, rgb_t


if __name__ == "__main__":
    import sys
    sr_dir = sys.argv[1] if len(sys.argv) > 1 else "output/patches/sr"
    color_dir = sys.argv[2] if len(sys.argv) > 2 else "output/patches/colorize"
    try:
        ds = SRPatchDataset(sr_dir)
        print(f"SR dataset: {len(ds)} pairs found in {sr_dir}")
    except RuntimeError as e:
        print(f"SR dataset: {e}")
    try:
        ds = ColorizationPatchDataset(color_dir)
        print(f"Colorization dataset: {len(ds)} pairs found in {color_dir}")
    except RuntimeError as e:
        print(f"Colorization dataset: {e}")
