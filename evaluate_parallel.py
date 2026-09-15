"""
evaluate_parallel.py -- multiprocessing version of evaluate.py's classical
(NCC/ZNCC) evaluation loop.

Matched against your actual localize.py / evaluate.py:
    localize(ref_path, search_path, confidence_gap_threshold=0.02,
             angle_range=4.0, angle_step=1.0, search_window_radius=150,
             scale_center=10.0, scale_range=1.0, scale_step=0.5,
             score_gate_frac=0.9)
        -> (pred_x, pred_y, pred_score, confident, n_candidates)

    manifest.csv columns used by evaluate.py: pair_id, style, ref_path,
    search_path, gt_x, gt_y (gt_x/gt_y optional -- blind-inference mode)

Output CSV has the SAME columns as evaluate.py's results/predictions.csv
(pair_id, style, ref_path, search_path, gt_x, gt_y, pred_x, pred_y,
error_px, time_sec, confident, n_candidates) so you can diff the two
files row-for-row as your correctness check before trusting any speedup
number.

WHY MULTIPROCESSING, NOT MULTITHREADING
    Python's GIL means threads don't give real parallelism for CPU-bound
    work like localize()'s cv2.matchTemplate + NMS. multiprocessing.Pool
    uses separate OS processes, each with its own interpreter -- that's
    what actually uses multiple CPU cores.

USAGE
    # Baseline (sequential) for comparison:
    time python evaluate.py --manifest data/train_canonical/manifest.csv \
        --csv-out results/predictions.csv

    # Parallel version:
    time python evaluate_parallel.py --manifest data/train_canonical/manifest.csv \
        --csv-out results/predictions_parallel.csv --workers 8

    # Correctness check -- these should show no differences other than
    # the time_sec column (which will legitimately differ):
    diff <(cut -d, -f1-9,11-12 results/predictions.csv) \
         <(cut -d, -f1-9,11-12 results/predictions_parallel.csv)
"""

import argparse
import csv
import os
import time
from multiprocessing import Pool, cpu_count

from localize import localize


def run_one_pair(row_and_index):
    """Runs in a worker process. Must be a top-level function (not a
    lambda/closure) so multiprocessing can pickle it. Mirrors the
    per-pair body of evaluate.py's main() loop exactly."""
    i, row = row_and_index
    pair_id = row.get("pair_id", i)
    style = row.get("style", "unknown")

    row_has_gt = "gt_x" in row and row["gt_x"] not in (None, "")
    gt_x, gt_y = (float(row["gt_x"]), float(row["gt_y"])) if row_has_gt else (None, None)

    start = time.time()
    pred_x, pred_y, score, confident, n_candidates = localize(row["ref_path"], row["search_path"])
    elapsed = time.time() - start

    if row_has_gt:
        error = ((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2) ** 0.5
    else:
        error = None

    return (pair_id, style, row["ref_path"], row["search_path"],
            gt_x, gt_y, pred_x, pred_y, error, elapsed, confident, n_candidates)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, required=True)
    ap.add_argument("--tolerance", type=float, default=5.0)
    ap.add_argument("--csv-out", type=str, default="results/predictions_parallel.csv")
    ap.add_argument("--workers", type=int, default=max(1, cpu_count() - 1),
                     help=f"Default: cpu_count()-1 = {max(1, cpu_count()-1)} on this machine")
    args = ap.parse_args()

    with open(args.manifest, "r") as f:
        rows = list(csv.DictReader(f))

    has_gt = len(rows) > 0 and "gt_x" in rows[0] and rows[0]["gt_x"] not in (None, "")
    if not has_gt:
        print("No ground truth (gt_x/gt_y) found in manifest -- running in "
              "blind-inference mode. Predictions will be written, but no "
              "accuracy/error stats can be computed.\n")

    print(f"[INFO] Loaded {len(rows)} pairs from {args.manifest}")
    print(f"[INFO] Using {args.workers} worker processes (machine has {cpu_count()} cores)")

    t0 = time.time()
    with Pool(processes=args.workers) as pool:
        results = pool.map(run_one_pair, list(enumerate(rows)))
    wall_clock = time.time() - t0

    times = [r[9] for r in results]
    errors = [r[8] for r in results if r[8] is not None]
    low_confidence_count = sum(1 for r in results if r[8] is not None and not r[10])

    os.makedirs(os.path.dirname(args.csv_out) or ".", exist_ok=True)
    with open(args.csv_out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pair_id", "style", "ref_path", "search_path",
                          "gt_x", "gt_y", "pred_x", "pred_y",
                          "error_px", "time_sec", "confident", "n_candidates"])
        for r in results:
            writer.writerow(["" if v is None else v for v in r])
    print(f"[OK] Wrote per-pair predictions to {args.csv_out}")

    print("\n" + "=" * 60)
    print(f"Total wall-clock time (parallel, {args.workers} workers): {wall_clock:.2f}s")
    print(f"Sum of per-pair compute time (sequential-equivalent): {sum(times):.2f}s")
    if wall_clock > 0:
        print(f"Effective speedup vs sequential sum: {sum(times)/wall_clock:.2f}x")
    print("(Compare 'Total wall-clock time' above against evaluate.py's own")
    print(" wall-clock time -- e.g. `time python evaluate.py ...` -- for your")
    print(" real before/after number. The 'effective speedup' line is a rough")
    print(" sanity check, not a replacement for that direct comparison.)")

    if not errors:
        avg_time = sum(times) / len(times) if times else 0.0
        print(f"\nTotal pairs processed: {len(times)} (no ground truth -- no accuracy stats)")
        print(f"Average computation time: {avg_time:.3f} sec/pair")
        return

    within_tol = sum(1 for e in errors if e <= args.tolerance)
    pct = 100 * within_tol / len(errors)
    avg_error = sum(errors) / len(errors)
    median_error = sorted(errors)[len(errors) // 2]

    print(f"\nTotal pairs evaluated: {len(errors)}")
    print(f"Within {args.tolerance}px tolerance: {within_tol}/{len(errors)} ({pct:.1f}%)")
    print(f"Average error: {avg_error:.2f} px   Median error: {median_error:.2f} px")
    print(f"Low-confidence (flagged ambiguous) predictions: {low_confidence_count}/{len(errors)}")

    print("\nAccuracy at multiple tolerances:")
    for tol in [1, 2, 4, 5, 10, 20, 50, 100, 150]:
        within = sum(1 for e in errors if e <= tol)
        print(f"  within {tol:>4}px: {within}/{len(errors)} ({100*within/len(errors):.1f}%)")

    print("\n[CHECK] These accuracy numbers should match evaluate.py's output")
    print("        exactly (same localize() calls, just reordered by which")
    print("        worker finished first) -- if they don't match, something")
    print("        is wrong with the parallelization, not just the timing.")


if __name__ == "__main__":
    main()
