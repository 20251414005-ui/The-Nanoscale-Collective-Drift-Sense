"""
train_model.py — trains the Siamese localizer on synthetic Drift-Sense pairs.

Reads a manifest.csv (produced by dataset_generator.py), builds a Gaussian
heatmap target centered at each pair's ground-truth location, and trains
the model to regress that heatmap (a standard approach for keypoint/center
localization — same idea used in CenterNet-style detectors).

Usage:
    python train_model.py --manifest data/train/manifest.csv --epochs 30 --output model_weights/siamese_localizer.pt
"""

import argparse
import csv
import os
import random

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from model import SiameseLocalizer
from localize_dl import heatmap_to_prediction


class DriftSenseDataset(Dataset):
    def __init__(self, rows, heatmap_stride=8):
        self.rows = rows
        self.heatmap_stride = heatmap_stride

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]

        ref_img = Image.open(row["ref_path"]).convert("L")
        # downsample reference 10x to match search-image scale, same as
        # the classical baseline's downsample_reference() in localize.py
        w, h = ref_img.size
        template_img = ref_img.resize((w // 10, h // 10), Image.LANCZOS)
        template = np.array(template_img, dtype=np.float32) / 255.0

        search_img = Image.open(row["search_path"]).convert("L")
        search = np.array(search_img, dtype=np.float32) / 255.0

        gt_x, gt_y = float(row["gt_x"]), float(row["gt_y"])

        template_t = torch.from_numpy(template).unsqueeze(0)   # (1, H, W)
        search_t = torch.from_numpy(search).unsqueeze(0)       # (1, H, W)

        return template_t, search_t, torch.tensor([gt_x, gt_y], dtype=torch.float32)


def make_gaussian_heatmap(size, center_xy, sigma=3.0):
    """size: (h, w) of the heatmap. center_xy: (x, y) in heatmap coordinates."""
    h, w = size
    yy, xx = torch.meshgrid(torch.arange(h, dtype=torch.float32),
                             torch.arange(w, dtype=torch.float32), indexing="ij")
    cx, cy = center_xy
    heat = torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
    return heat


def weighted_heatmap_loss(pred, target, pos_weight=200.0):
    """
    Plain MSE over a mostly-zero heatmap lets the network minimize loss by
    predicting near-zero everywhere, without learning to localize anything
    (this was empirically confirmed: loss dropped to 0.001 but accuracy on
    the TRAINING set itself was 0%, meaning the low loss was fake progress).
    Fix: weight errors near the true peak (target > 0.1) much more heavily
    than background errors, so getting the peak location right actually
    matters to the loss.
    """
    weights = torch.ones_like(target)
    weights[target > 0.1] = pos_weight
    return (weights * (pred - target) ** 2).mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", type=str, default="model_weights/siamese_localizer.pt")
    parser.add_argument("--heatmap_stride", type=int, default=8)
    parser.add_argument("--val_fraction", type=float, default=0.15,
                         help="Fraction of data held out for validation (never trained on)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    with open(args.manifest, "r") as f:
        all_rows = list(csv.DictReader(f))

    random.seed(args.seed)
    shuffled = all_rows[:]
    random.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * args.val_fraction))
    val_rows = shuffled[:n_val]
    train_rows = shuffled[n_val:]
    print(f"Train pairs: {len(train_rows)}   Validation pairs (held out, never trained on): {len(val_rows)}")

    train_dataset = DriftSenseDataset(train_rows, heatmap_stride=args.heatmap_stride)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    val_dataset = DriftSenseDataset(val_rows, heatmap_stride=args.heatmap_stride)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    model = SiameseLocalizer().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for template, search, gt_xy in train_loader:
            template, search, gt_xy = template.to(device), search.to(device), gt_xy.to(device)

            optimizer.zero_grad()
            heatmap_pred = model(template, search)  # (B, 1, ho, wo)
            B, _, ho, wo = heatmap_pred.shape

            # The cross-correlation output at heatmap index (i,j) corresponds
            # to the template's CENTER landing at search-feature-map position
            # (i + ht/2, j + wt/2), where (ht, wt) is the template's own
            # feature-map size. Get that directly from the encoder so the
            # offset is always correct even if the architecture changes.
            with torch.no_grad():
                template_feat = model.encoder(template)
            _, _, ht, wt = template_feat.shape

            targets = []
            for b in range(B):
                gx = gt_xy[b, 0].item() / args.heatmap_stride - wt / 2
                gy = gt_xy[b, 1].item() / args.heatmap_stride - ht / 2
                targets.append(make_gaussian_heatmap((ho, wo), (gx, gy)))
            target_heatmap = torch.stack(targets).unsqueeze(1).to(device)

            loss = weighted_heatmap_loss(heatmap_pred, target_heatmap)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Validation: real pixel-error on data the model has NEVER trained
        # on, computed every few epochs. This is what actually tells us
        # whether the model is learning to generalize or just memorizing
        # the training set (a real failure mode we hit and confirmed
        # empirically in an earlier run: 19.4% train accuracy but 0% on a
        # freshly generated held-out set).
        val_msg = ""
        if (epoch + 1) % 5 == 0 or (epoch + 1) == args.epochs:
            model.eval()
            errors = []
            with torch.no_grad():
                for template, search, gt_xy in val_loader:
                    template, search = template.to(device), search.to(device)
                    template_feat = model.encoder(template)
                    heatmap = model(template, search)[0, 0]
                    _, _, ht, wt = template_feat.shape
                    search_h, search_w = search.shape[2], search.shape[3]
                    pred_x, pred_y, _, _ = heatmap_to_prediction(
                        heatmap.cpu().numpy(), wt, ht, search_w, search_h, args.heatmap_stride)
                    gx, gy = gt_xy[0, 0].item(), gt_xy[0, 1].item()
                    errors.append(((pred_x - gx) ** 2 + (pred_y - gy) ** 2) ** 0.5)
            val_avg_error = sum(errors) / len(errors)
            val_within_5px = 100 * sum(1 for e in errors if e <= 5) / len(errors)
            val_msg = f"   val_avg_error: {val_avg_error:.1f}px   val_within_5px: {val_within_5px:.1f}%"

        print(f"Epoch {epoch + 1}/{args.epochs} — loss: {avg_loss:.6f}{val_msg}")

        # Save a checkpoint periodically so progress isn't lost if training
        # is interrupted or you want to stop early.
        if (epoch + 1) % 10 == 0 or (epoch + 1) == args.epochs:
            torch.save(model.state_dict(), args.output)
            print(f"  -> checkpoint saved to {args.output}")

    print(f"\nTraining complete. Final weights saved to {args.output}")


if __name__ == "__main__":
    main()