"""
evaluate.py — runs localize.py's core logic across every pair in a manifest.csv
and reports accuracy + timing stats for the Results slide.

Usage:
    python evaluate.py --manifest data/train/manifest.csv --tolerance 5
"""

import argparse
import csv
import time

from localize import localize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--tolerance", type=float, default=5.0,
                         help="Pixel distance considered a 'correct' localization")
    parser.add_argument("--csv-out", type=str, default="results/predictions.csv",
                         help="Path to write per-pair predictions CSV")
    args = parser.parse_args()

    rows = []
    with open(args.manifest, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    errors = []
    times = []
    low_confidence_count = 0
    results = []

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
        pred_x, pred_y, score, confident, n_candidates = localize(row["ref_path"], row["search_path"])
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
                         gt_x, gt_y, pred_x, pred_y, error, elapsed, confident, n_candidates))
        print(f"pair {str(pair_id):>3} ({style:<6}) {error_str}"
              f"time={elapsed:.3f}s  confident={confident}  candidates={n_candidates}")

    import os as _os
    _os.makedirs(_os.path.dirname(args.csv_out) or ".", exist_ok=True)
    with open(args.csv_out, "w", newline="") as _f:
        _writer = csv.writer(_f)
        _writer.writerow(["pair_id", "style", "ref_path", "search_path",
                          "gt_x", "gt_y", "pred_x", "pred_y",
                          "error_px", "time_sec", "confident", "n_candidates"])
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
    avg_time = sum(times) / len(times)
    avg_error = sum(errors) / len(errors)
    median_error = sorted(errors)[len(errors) // 2]

    print("\n" + "=" * 60)
    print(f"Total pairs evaluated: {len(errors)}")
    print(f"Within {args.tolerance}px tolerance: {within_tol}/{len(errors)} ({pct:.1f}%)")
    print(f"Average error: {avg_error:.2f} px   Median error: {median_error:.2f} px")
    print(f"Average computation time: {avg_time:.3f} sec/pair")
    print(f"Low-confidence (flagged ambiguous) predictions: {low_confidence_count}/{len(errors)}")

    # Report accuracy at several tolerances, not just one — a single harsh
    # cutoff (e.g. 5px) makes a genuinely-working system look like it's
    # failing outright when the task's real difficulty (periodic
    # ambiguity, explicitly called out in the problem statement) makes
    # very tight tolerances hard to hit even for correct-ish matches.
    print("\nAccuracy at multiple tolerances (for a fair, complete picture):")
    for tol in [1, 2, 4, 5, 10, 20, 50, 100, 150]:
        within = sum(1 for e in errors if e <= tol)
        print(f"  within {tol:>4}px: {within}/{len(errors)} ({100*within/len(errors):.1f}%)")

    # surface worst case explicitly for the "honest failure" slide
    worst = max(results, key=lambda r: r[8])
    print(f"\nWorst case: pair {worst[0]} ({worst[1]}) — error {worst[8]:.2f}px, "
          f"confident={worst[10]}, candidates={worst[11]}")


if __name__ == "__main__":
    main()