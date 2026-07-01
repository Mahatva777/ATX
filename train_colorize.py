"""
Train the colorization model (conditional GAN, Pix2Pix-style).

Usage:
    python train_colorize.py --patches_dir output/patches --epochs 60 --batch_size 8 \
        --semantic_loss_weight 5.0

Loss = lambda_l1 * L1(fake, real) + adversarial + lambda_sem * semantic_constraint
Defaults follow the original Pix2Pix paper's lambda_l1=100 balance, scaled
down slightly since we're adding a third loss term.
"""
import argparse
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from models.colorization_model import UNetGenerator, PatchGANDiscriminator
from models.semantic_constraint import BandRatioConsistencyLoss
from datasets.patch_dataset import ColorizationPatchDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patches_dir", default="output/patches/colorize",
                     help="Directory holding ONLY colorization pairs (256x256 100m TIR input, 256x256 RGB target). "
                          "Must be separate from SR patches -- see datasets/patch_dataset.py docstring.")
    ap.add_argument("--epochs", type=int, default=60)  # JUDGMENT CALL: GANs need more epochs than plain SR
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lambda_l1", type=float, default=80.0)
    ap.add_argument("--semantic_loss_weight", type=float, default=5.0)  # JUDGMENT CALL: start low, raise if hallucinations persist
    ap.add_argument("--val_split", type=float, default=0.1)
    ap.add_argument("--ckpt_dir", default="checkpoints")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.makedirs(args.ckpt_dir, exist_ok=True)
    device = torch.device(args.device)

    full_ds = ColorizationPatchDataset(args.patches_dir)
    n_val = max(1, int(len(full_ds) * args.val_split))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    G = UNetGenerator().to(device)
    D = PatchGANDiscriminator().to(device)
    opt_g = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))

    bce = nn.BCEWithLogitsLoss()
    l1 = nn.L1Loss()
    semantic_loss_fn = BandRatioConsistencyLoss()

    best_val_l1 = float("inf")
    for epoch in range(1, args.epochs + 1):
        G.train(); D.train()
        t0 = time.time()
        running = {"g": 0.0, "d": 0.0, "l1": 0.0, "sem": 0.0}
        for tir, rgb_real in train_loader:
            tir, rgb_real = tir.to(device), rgb_real.to(device)
            bsz = tir.size(0)

            # ---- Discriminator step ----
            with torch.no_grad():
                rgb_fake = G(tir)
            opt_d.zero_grad()
            pred_real = D(tir, rgb_real)
            pred_fake = D(tir, rgb_fake.detach())
            loss_d_real = bce(pred_real, torch.ones_like(pred_real))
            loss_d_fake = bce(pred_fake, torch.zeros_like(pred_fake))
            loss_d = 0.5 * (loss_d_real + loss_d_fake)
            loss_d.backward()
            opt_d.step()

            # ---- Generator step ----
            opt_g.zero_grad()
            rgb_fake = G(tir)
            pred_fake_for_g = D(tir, rgb_fake)
            loss_g_adv = bce(pred_fake_for_g, torch.ones_like(pred_fake_for_g))
            loss_g_l1 = l1(rgb_fake, rgb_real) * args.lambda_l1
            loss_g_sem = semantic_loss_fn(tir, rgb_fake) * args.semantic_loss_weight
            loss_g = loss_g_adv + loss_g_l1 + loss_g_sem
            loss_g.backward()
            opt_g.step()

            running["g"] += loss_g.item() * bsz
            running["d"] += loss_d.item() * bsz
            running["l1"] += loss_g_l1.item() * bsz
            running["sem"] += loss_g_sem.item() * bsz

        n = len(train_ds)
        print(f"[Colorize] epoch {epoch}/{args.epochs}  "
              f"G={running['g']/n:.3f}  D={running['d']/n:.3f}  "
              f"L1={running['l1']/n:.3f}  Sem={running['sem']/n:.3f}  "
              f"({time.time()-t0:.1f}s)")

        # ---- validation (L1 only, cheap proxy for checkpoint selection) ----
        G.eval()
        val_l1, nval = 0.0, 0
        with torch.no_grad():
            for tir, rgb_real in val_loader:
                tir, rgb_real = tir.to(device), rgb_real.to(device)
                rgb_fake = G(tir)
                val_l1 += l1(rgb_fake, rgb_real).item() * tir.size(0)
                nval += tir.size(0)
        val_l1 /= nval
        print(f"           val_L1={val_l1:.4f}")

        torch.save(G.state_dict(), os.path.join(args.ckpt_dir, "colorize_G_last.pth"))
        torch.save(D.state_dict(), os.path.join(args.ckpt_dir, "colorize_D_last.pth"))
        if val_l1 < best_val_l1:
            best_val_l1 = val_l1
            torch.save(G.state_dict(), os.path.join(args.ckpt_dir, "colorize_G_best.pth"))

    print(f"Done. Best val L1: {best_val_l1:.4f}. "
          f"Checkpoints in {args.ckpt_dir}/colorize_G_best.pth")


if __name__ == "__main__":
    main()
