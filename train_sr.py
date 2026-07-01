"""
Train the super-resolution model.

Usage:
    python train_sr.py --patches_dir output/patches --epochs 30 --batch_size 8

Judgment calls flagged inline with EPOCHS/BATCH_SIZE defaults -- tune down
if you're on a small free-tier GPU or the patch count is small.
"""
import argparse
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from models.sr_model import TIRSuperResolutionNet
from datasets.patch_dataset import SRPatchDataset


def psnr_from_mse(mse: torch.Tensor) -> torch.Tensor:
    # data range is 2.0 since tensors are normalized to [-1, 1]
    return 10 * torch.log10(4.0 / (mse + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patches_dir", default="output/patches/sr",
                     help="Directory holding ONLY SR pairs (200m TIR input, 512x512 100m TIR target). "
                          "Must be separate from colorization patches -- see datasets/patch_dataset.py docstring.")
    ap.add_argument("--epochs", type=int, default=30)  # JUDGMENT CALL: raise if val loss still falling
    ap.add_argument("--batch_size", type=int, default=8)  # JUDGMENT CALL: lower to 4 if you hit OOM on T4
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--val_split", type=float, default=0.1)
    ap.add_argument("--ckpt_dir", default="checkpoints")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.makedirs(args.ckpt_dir, exist_ok=True)
    device = torch.device(args.device)

    full_ds = SRPatchDataset(args.patches_dir)
    n_val = max(1, int(len(full_ds) * args.val_split))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = TIRSuperResolutionNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    l1 = nn.L1Loss()

    best_val_psnr = -float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        train_loss = 0.0
        for low, high in train_loader:
            low, high = low.to(device), high.to(device)
            opt.zero_grad()
            pred = model(low)
            loss = l1(pred, high)
            loss.backward()
            opt.step()
            train_loss += loss.item() * low.size(0)
        train_loss /= len(train_ds)

        model.eval()
        val_mse, n = 0.0, 0
        with torch.no_grad():
            for low, high in val_loader:
                low, high = low.to(device), high.to(device)
                pred = model(low)
                mse = torch.mean((pred - high) ** 2)
                val_mse += mse.item() * low.size(0)
                n += low.size(0)
        val_mse /= n
        val_psnr = psnr_from_mse(torch.tensor(val_mse)).item()

        print(f"[SR] epoch {epoch}/{args.epochs}  train_L1={train_loss:.4f}  "
              f"val_MSE={val_mse:.4f}  val_PSNR={val_psnr:.2f}dB  ({time.time()-t0:.1f}s)")

        torch.save(model.state_dict(), os.path.join(args.ckpt_dir, "sr_last.pth"))
        if val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            torch.save(model.state_dict(), os.path.join(args.ckpt_dir, "sr_best.pth"))

    print(f"Done. Best val PSNR: {best_val_psnr:.2f}dB. "
          f"Checkpoints in {args.ckpt_dir}/sr_best.pth and sr_last.pth")


if __name__ == "__main__":
    main()
