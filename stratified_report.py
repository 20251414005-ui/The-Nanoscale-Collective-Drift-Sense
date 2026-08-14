"""
stratified_report.py -- breaks accuracy down by scale, rotation, and noise
buckets, rather than one flat number across all pairs. Reads
results/full_results.csv (built by merge_results.py) and requires no new
experiments -- purely an analysis pass over predictions you already have.

Satisfies the spec's validation requirement: "Results across multiple
noise levels, target positions, scales and rotations" (a distinct item from
the 1/2/4/5/10px threshold table).

Usage:
    python stratified_report.py --input results/full_results.csv \
        --tolerance 5.0 --out results/stratified_report.csv
"""

import argparse
import csv


def to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def bucket_terciles(rows, key):
    """Split rows into Low/Mid/High buckets by tercile of `key`, returning
    (bucket_name, lo, hi, rows_in_bucket) for each of the 3 buckets."""
    vals = sorted((to_float(r[key]), i) for i, r in enumerate(rows) if to_float(r[key]) is not None)
    n = len(vals)
    if n < 3:
        return []
    t1, t2 = n // 3, 2 * n // 3
    edges = [vals[0][0], vals[t1][0], vals[t2][0], vals[-1][0]]
    names = ["Low", "Mid", "High"]
    buckets = []
    for b in range(3):
        lo, hi = edges[b], edges[b + 1]
        idxs = [i for v, i in vals if (v >= lo and (v < hi or b == 2))]
        buckets.append((names[b], lo, hi, [rows[i] for i in idxs]))
    return buckets


def accuracy_at(rows, err_col, tolerance):
    errs = [to_float(r[err_col]) for r in rows if to_float(r[err_col]) is not None]
    if not errs:
        return None, 0
    within = sum(1 for e in errs if e <= tolerance)
    return 100.0 * within / len(errs), len(errs)


def report_dimension(rows, key, label, tolerance, out_lines):
    buckets = bucket_terciles(rows, key)
    if not buckets:
        out_lines.append(f"\n## By {label}\n(not enough data with valid {key} values)\n")
        return
    out_lines.append(f"\n## By {label}\n")
    out_lines.append(f"| Bucket | Range | N pairs | Classical @{tolerance}px | DL @{tolerance}px |")
    out_lines.append("|---|---|---|---|---|")
    for name, lo, hi, bucket_rows in buckets:
        c_acc, c_n = accuracy_at(bucket_rows, "classical_error_px", tolerance)
        d_acc, d_n = accuracy_at(bucket_rows, "dl_error_px", tolerance)
        c_str = f"{c_acc:.1f}%" if c_acc is not None else "n/a"
        d_str = f"{d_acc:.1f}%" if d_acc is not None else "n/a"
        out_lines.append(f"| {name} | {lo:.2f}\u2013{hi:.2f} | {len(bucket_rows)} | {c_str} | {d_str} |")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="results/full_results.csv")
    p.add_argument("--tolerance", type=float, default=5.0)
    p.add_argument("--out", default="results/stratified_report.md")
    args = p.parse_args()

    with open(args.input, "r", newline="") as f:
        rows = list(csv.DictReader(f))

    out_lines = [f"# Stratified Accuracy Report (tolerance = {args.tolerance}px)\n",
                 f"Built from {len(rows)} pairs in {args.input}. "
                 f"Buckets are terciles (roughly equal-sized groups) of each dimension's "
                 f"actual observed range in this dataset, not fixed external cutoffs."]

    if "scale" in rows[0]:
        report_dimension(rows, "scale", "Scale", args.tolerance, out_lines)
    if "rotation_deg" in rows[0]:
        report_dimension(rows, "rotation_deg", "Rotation (degrees)", args.tolerance, out_lines)
    if "noise_sigma_search" in rows[0]:
        report_dimension(rows, "noise_sigma_search", "Noise (search-image sigma)", args.tolerance, out_lines)

    report_str = "\n".join(out_lines)
    with open(args.out, "w") as f:
        f.write(report_str)

    print(report_str)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
