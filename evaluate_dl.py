"""
evaluate_dl.py — batch accuracy/timing evaluation for the trained DL model
"""

import argparse
import csv
import os
import time

import torch

from localize_dl import load_dl_model, localize_dl_with_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--weights", type=str, default="model_weights/siamese_localizer_canonical.pt")
    parser.add_argument("--tolerance", type=float, default=5.0)
    parser.add_argument("--csv-out", type=str, default="results/predictions_dl.csv",
                         help="Path to write per-pair predictions CSV")
    parser.add_argument("--device", type=str, default=None,
                         help="cuda or cpu; defaults to cuda if available")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    with open(args.manifest, "r") as f:
        rows = list(csv.DictReader(f))

    # Load the model ONCE — the single biggest fix over the original script.
    model = load_dl_model(args.weights, device=device)

    # GPU warm-up: the first CUDA call in a process pays a one-time
    # context-init + kernel-selection cost. One throwaway inference here
    # keeps that fixed cost out of the timed average below.
    if device == "cuda" and rows:
        localize_dl_with_model(model, rows[0]["ref_path"], rows[0]["search_path"], device=device)

    errors, times, results = [], [], []
    low_confidence_count = 0

    has_gt = len(rows) > 0 and "gt_x" in rows[0] and rows[0]["gt_x"] not in (None, "")
    if not has_gt:
        print("No ground truth (gt_x/gt_y) found in manifest -- running in "
              "blind-inference mode. Predictions will be written, but no "
              "accuracy/error stats can be computed.\n")

    for i, row in enumerate(rows):
        pair_id = row.get("pair_id", i)
        style = row.get("style", "unknown")

        row_has_gt = "gt_x" in row and row["gt_x"] not in (None, "")
        gt_x, gt_y = (float(row["gt_x"]), float(row["gt_y"])) if row_has_gt else (None, None)

        start = time.time()
        pred_x, pred_y, score, confident = localize_dl_with_model(
            model, row["ref_path"], row["search_path"], device=device)
        elapsed = time.time() - start
        times.append(elapsed)

        if row_has_gt:
            error = ((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2) ** 0.5
            errors.append(error)
            if not confident:
                low_confidence_count += 1
            error_str = f"error={error:6.2f}px  "
        else:
            error = None
            error_str = ""

        results.append((pair_id, style, row["ref_path"], row["search_path"],
                         gt_x, gt_y, pred_x, pred_y, error, elapsed, confident))

        print(f"pair {str(pair_id):>3} ({style:<6}) {error_str}"
              f"time={elapsed:.3f}s  confident={confident}")

    os.makedirs(os.path.dirname(args.csv_out) or ".", exist_ok=True)
    with open(args.csv_out, "w", newline="") as _f:
        _writer = csv.writer(_f)
        _writer.writerow(["pair_id", "style", "ref_path", "search_path",
                           "gt_x", "gt_y", "pred_x", "pred_y",
                           "error_px", "time_sec", "confident"])
        for _r in results:
            _writer.writerow(["" if v is None else v for v in _r])
    print(f"Wrote per-pair predictions to {args.csv_out}")

    if not errors:
        print("\n" + "=" * 60)
        print(f"Total pairs processed: {len(times)} (no ground truth supplied -- "
              f"predictions written to {args.csv_out}, no accuracy stats available)")
        avg_time = sum(times) / len(times) if times else 0.0
        print(f"Average computation time: {avg_time:.3f} sec/pair")
        return

    within_tol = sum(1 for e in errors if e <= args.tolerance)
    pct = 100 * within_tol / len(errors)

    print("\n" + "=" * 60)
    print(f"Total pairs evaluated: {len(errors)}")
    print(f"Within {args.tolerance}px tolerance: {within_tol}/{len(errors)} ({pct:.1f}%)")
    median_error = sorted(errors)[len(errors) // 2]
    print(f"Average error: {sum(errors)/len(errors):.2f} px   Median error: {median_error:.2f} px")
    print(f"Average computation time: {sum(times)/len(times):.3f} sec/pair")
    print(f"Low-confidence predictions: {low_confidence_count}/{len(errors)}")

    print("\nAccuracy at multiple tolerances (for a fair, complete picture):")
    for tol in [1, 2, 4, 5, 10, 20, 50, 100, 150]:
        within = sum(1 for e in errors if e <= tol)
        print(f"  within {tol:>4}px: {within}/{len(errors)} ({100*within/len(errors):.1f}%)")

    scored_results = [r for r in results if r[8] is not None]
    worst = max(scored_results, key=lambda r: r[8])
    print(f"\nWorst case: pair {worst[0]} ({worst[1]}) — error {worst[8]:.2f}px, confident={worst[10]}")


if __name__ == "__main__":
    main()
