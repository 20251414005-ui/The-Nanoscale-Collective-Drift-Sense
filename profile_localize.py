"""
profile_localize.py — breaks down exactly where localize()'s time goes,
so we optimize the real bottleneck instead of guessing.

Usage:
    python profile_localize.py --manifest data/train/manifest.csv --n 20
"""

import argparse
import cProfile
import csv
import pstats
import time
from io import StringIO

import numpy as np
from PIL import Image

from localize import (load_grayscale, downsample_reference, find_candidates,
                       rotate_template, localize)


def stage_by_stage_timing(ref_path, search_path):
    """Manually times each major stage of localize() for ONE pair."""
    timings = {}

    t0 = time.time()
    search_arr_full = load_grayscale(search_path)
    ref_arr_full = load_grayscale(ref_path)
    timings["load images"] = time.time() - t0

    t0 = time.time()
    template_arr = downsample_reference(ref_arr_full, factor=10)
    timings["downsample reference"] = time.time() - t0

    H, W = search_arr_full.shape
    search_window_radius = 300
    cx_img, cy_img = W / 2, H / 2
    x0 = int(max(0, cx_img - search_window_radius))
    x1 = int(min(W, cx_img + search_window_radius))
    y0 = int(max(0, cy_img - search_window_radius))
    y1 = int(min(H, cy_img + search_window_radius))
    search_arr = search_arr_full[y0:y1, x0:x1]

    angles = np.arange(-4.0, 4.0 + 0.01, 1.0)

    t0 = time.time()
    rotated_templates = [rotate_template(template_arr, a) if a != 0 else template_arr for a in angles]
    timings["rotate template (x9 angles)"] = time.time() - t0

    t0 = time.time()
    all_candidates = []
    w = h = None
    for rt in rotated_templates:
        candidates, w, h = find_candidates(search_arr, rt)
        all_candidates.extend(candidates)
    timings["template match + NMS (x9 angles)"] = time.time() - t0

    t0 = time.time()
    all_candidates.sort(key=lambda c: -c[2])
    deduped = []
    for x, y, s in all_candidates:
        too_close = any(
            abs(x - cx) < w * 0.5 and abs(y - cy) < h * 0.5
            for cx, cy, _ in deduped
        )
        if not too_close:
            deduped.append((x, y, s))
    timings["final dedup across angles"] = time.time() - t0

    total = sum(timings.values())
    timings["TOTAL"] = total
    return timings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--n", type=int, default=20, help="Number of pairs to profile")
    args = parser.parse_args()

    with open(args.manifest, "r") as f:
        rows = list(csv.DictReader(f))[:args.n]

    print(f"=== Stage-by-stage timing (averaged over {len(rows)} pairs) ===\n")
    accum = {}
    for row in rows:
        timings = stage_by_stage_timing(row["ref_path"], row["search_path"])
        for k, v in timings.items():
            accum.setdefault(k, []).append(v)

    for stage, values in accum.items():
        avg = sum(values) / len(values)
        pct = (avg / accum["TOTAL"][0]) * 100 if stage != "TOTAL" else 100.0
        # recompute pct properly using each pair's own total
        print(f"  {stage:<32} avg={avg*1000:7.2f} ms")

    print("\n(Percentages below use the true per-pair total, not the first pair's)")
    total_avg = sum(accum["TOTAL"]) / len(accum["TOTAL"])
    for stage, values in accum.items():
        if stage == "TOTAL":
            continue
        avg = sum(values) / len(values)
        print(f"  {stage:<32} {100*avg/total_avg:5.1f}% of total")

    print(f"\n  {'TOTAL':<32} avg={total_avg*1000:7.2f} ms\n")

    print("=== cProfile function-level breakdown (single pair, most detailed view) ===\n")
    profiler = cProfile.Profile()
    profiler.enable()
    localize(rows[0]["ref_path"], rows[0]["search_path"])
    profiler.disable()

    s = StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
    ps.print_stats(15)  # top 15 functions by cumulative time
    print(s.getvalue())


if __name__ == "__main__":
    main()
