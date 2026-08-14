"""
pr_curve.py

Sweeps localize.py's existing confidence-gap threshold (the gap between the
best and second-best candidate score -- already computed internally, just
not previously swept/plotted) and reports precision vs. recall at several
noise levels, per the sponsor session's PR-curve guidance:
  - precision = of pairs reported as "confident" at a given threshold,
                the fraction that are actually correct (error <= tolerance)
  - recall    = fraction of ALL pairs that were both confidently reported
                AND correct
  - swept separately at Low/Medium/High noise (terciles of noise_sigma_search
    already present in the manifest -- NOTE: this bins existing randomly-
    generated noise levels rather than a fully controlled regenerate-at-
    fixed-noise sweep, which would be more rigorous if time allows)
  - reports the best-F1 threshold explicitly and states it, as the
    guidance requires ("choose the threshold with evidence... state it")

Matches each pair ONCE (the expensive part) and caches the raw score gap,
so the threshold sweep afterward is cheap rather than re-running the full
scale+rotation search per threshold value.

Usage:
    python3 pr_curve.py --manifest data/train_defects_full/manifest.csv --tolerance 5
    python3 pr_curve.py --manifest data/train_variable_scale/manifest.csv --tolerance 5 --limit 200
"""
import argparse
import os
import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from localize import (load_grayscale, downsample_reference, rotate_template,
                       find_candidates, _vectorized_nms)


def match_pair(ref_path, search_path, angle_range=4.0, angle_step=1.0,
                search_window_radius=150, scale_center=10.0, scale_range=1.0,
                scale_step=0.5, score_gate_frac=0.9):
    """Full scale+rotation search + score-gated tiebreak, run ONCE per pair.
    Returns pred_x, pred_y, and the RAW score gap (not yet thresholded),
    so the confidence cutoff can be swept cheaply afterward without
    re-matching."""
    search_arr_full = load_grayscale(search_path)
    ref_arr_full = load_grayscale(ref_path)

    H, W = search_arr_full.shape
    cx_img, cy_img = W / 2, H / 2
    x0 = int(max(0, cx_img - search_window_radius))
    x1 = int(min(W, cx_img + search_window_radius))
    y0 = int(max(0, cy_img - search_window_radius))
    y1 = int(min(H, cy_img + search_window_radius))
    search_arr = search_arr_full[y0:y1, x0:x1]

    angles = np.arange(-angle_range, angle_range + 0.01, angle_step)
    scales = np.arange(scale_center - scale_range, scale_center + scale_range + 0.01, scale_step)

    all_centers = []
    for scale in scales:
        template_arr = downsample_reference(ref_arr_full, factor=scale)
        for angle in angles:
            rotated_template = rotate_template(template_arr, angle) if angle != 0 else template_arr
            candidates, w, h = find_candidates(search_arr, rotated_template)
            for x, y, s in candidates:
                all_centers.append((x + w / 2 + x0, y + h / 2 + y0, s))

    if not all_centers:
        return cx_img, cy_img, 0.0

    base_template = downsample_reference(ref_arr_full, factor=scale_center)
    bh, bw = base_template.shape
    xs = np.array([c[0] for c in all_centers])
    ys = np.array([c[1] for c in all_centers])
    scores = np.array([c[2] for c in all_centers])
    final = _vectorized_nms(xs, ys, scores, bw, bh, nms_radius_ratio=0.5)

    centers = [(x, y, s) for x, y, s in final]
    best_score = max(c[2] for c in centers)
    gated = [c for c in centers if c[2] >= best_score * score_gate_frac]

    img_center = (cx_img, cy_img)
    best = min(gated, key=lambda c: (c[0] - img_center[0]) ** 2 + (c[1] - img_center[1]) ** 2)
    pred_x, pred_y, pred_score = best

    scores_sorted = sorted((c[2] for c in centers), reverse=True)
    gap = scores_sorted[0] - scores_sorted[1] if len(scores_sorted) > 1 else 1.0

    return pred_x, pred_y, gap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--tolerance", type=float, default=5.0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="results/pr_curve.png")
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    if args.limit:
        df = df.head(args.limit)

    print(f"Matching {len(df)} pairs (runs once, this is the slow part)...")
    records = []
    for i, row in df.iterrows():
        px, py, gap = match_pair(row["ref_path"], row["search_path"])
        err = ((px - row["gt_x"]) ** 2 + (py - row["gt_y"]) ** 2) ** 0.5
        records.append({
            "error": err,
            "gap": gap,
            "noise_sigma_search": row.get("noise_sigma_search", np.nan),
        })
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(df)}")

    res = pd.DataFrame(records)

    if res["noise_sigma_search"].notna().sum() >= 6:
        q1, q2 = res["noise_sigma_search"].quantile([1 / 3, 2 / 3])
        res["noise_bucket"] = pd.cut(
            res["noise_sigma_search"], bins=[-np.inf, q1, q2, np.inf],
            labels=["Low noise", "Medium noise", "High noise"]
        ).astype(str)
        buckets = ["Low noise", "Medium noise", "High noise"]
    else:
        res["noise_bucket"] = "All"
        buckets = ["All"]

    thresholds = np.linspace(0.0, max(res["gap"].quantile(0.95), 1e-6), 25)

    plt.figure(figsize=(7, 6))
    best_overall = (None, -1.0, None)

    for bucket in buckets:
        sub = res[res["noise_bucket"] == bucket]
        if len(sub) == 0:
            continue
        precisions, recalls = [], []
        for t in thresholds:
            reported = sub[sub["gap"] > t]
            if len(reported) == 0:
                precisions.append(1.0)
                recalls.append(0.0)
                continue
            correct_reported = (reported["error"] <= args.tolerance).sum()
            precision = correct_reported / len(reported)
            recall = correct_reported / len(sub)
            precisions.append(precision)
            recalls.append(recall)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            if f1 > best_overall[1]:
                best_overall = (bucket, f1, t)

        plt.plot(recalls, precisions, marker="o", label=f"{bucket} (n={len(sub)})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision vs Recall by Noise Level (tolerance={args.tolerance}px)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.ylim(-0.02, 1.02)
    plt.xlim(-0.02, 1.02)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot to {args.out}")

    print(f"\nBest F1 overall: bucket={best_overall[0]}  F1={best_overall[1]:.3f}  "
          f"confidence_gap_threshold={best_overall[2]:.4f}")
    print("State this threshold explicitly in the writeup/PPT, per the sponsor "
          "guidance ('choose the threshold with evidence... state it').")


if __name__ == "__main__":
    main()
