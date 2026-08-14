"""
analyze_pitch_aliasing.py (v2) — tests whether errors are genuine small
periodic "tile jumps" (1-5 repeats off, physically plausible for a
navigation-error scenario) vs. large unrelated misses, correcting the v1
flaw where a small pitch unit made almost any error look "close" to some
huge, physically meaningless multiple (e.g. 29x, 39x pitch).
"""

import argparse
import csv

import numpy as np

from localize import localize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--n", type=int, default=800)
    parser.add_argument("--residual_threshold", type=float, default=0.15)
    parser.add_argument("--max_jump", type=int, default=5,
                         help="Max repeats-away still considered a physically plausible tile jump")
    parser.add_argument("--correct_frac_pitch", type=float, default=0.5,
                         help="Error below this fraction of one pitch counts as 'correct'")
    args = parser.parse_args()

    with open(args.manifest, "r") as f:
        rows = list(csv.DictReader(f))[:args.n]

    results = []
    for row in rows:
        gt_x, gt_y = float(row["gt_x"]), float(row["gt_y"])
        pitch = float(row["pitch"])
        pred_x, pred_y, score, confident, n_candidates = localize(row["ref_path"], row["search_path"])
        error = ((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2) ** 0.5

        search_pitch = pitch / 10.0
        ratio = error / search_pitch if search_pitch > 0 else float("nan")
        nearest_int = round(ratio)
        residual = abs(ratio - nearest_int)

        results.append(dict(pair_id=row["pair_id"], style=row["style"], error=error,
                             search_pitch=search_pitch, nearest_int=nearest_int, residual=residual))

    n = len(results)
    correct = [r for r in results if r["error"] < args.correct_frac_pitch * r["search_pitch"]]
    small_jump = [r for r in results if r not in correct and r["residual"] <= args.residual_threshold
                  and 1 <= abs(r["nearest_int"]) <= args.max_jump]
    unrelated = [r for r in results if r not in correct and r not in small_jump]

    print("=" * 70)
    print(f"Total pairs analyzed: {n}\n")
    print(f"Category 1 — CORRECT (error < {args.correct_frac_pitch}x pitch):        "
          f"{len(correct):4d}/{n} ({100*len(correct)/n:5.1f}%)")
    print(f"Category 2 — SMALL TILE JUMP (1-{args.max_jump} repeats off, clean fit): "
          f"{len(small_jump):4d}/{n} ({100*len(small_jump)/n:5.1f}%)")
    print(f"Category 3 — LARGE / UNRELATED MISS:                    "
          f"{len(unrelated):4d}/{n} ({100*len(unrelated)/n:5.1f}%)")

    print(f"\nInterpretation:")
    print(f"  - Category 2 = genuine evidence of periodic confusion (locked onto a")
    print(f"    NEARBY repeat of the pattern, a plausible real navigation-error mode).")
    print(f"  - Category 3 = the algorithm landed somewhere with no clean periodic")
    print(f"    relationship to the true site at all — a different kind of failure")
    print(f"    (e.g. noise dominating the match, or search window not containing it).")

    if small_jump:
        best = min(small_jump, key=lambda r: r["residual"])
        print(f"\nBest example for 'honest failure case' (genuine small tile-jump):")
        print(f"  pair {best['pair_id']} ({best['style']}) — error={best['error']:.2f}px "
              f"= {best['nearest_int']:+d} x pitch (residual={best['residual']:.3f})")

    if unrelated:
        worst = max(unrelated, key=lambda r: r["error"])
        print(f"\nExample of an UNRELATED miss (different failure mode, also worth showing):")
        print(f"  pair {worst['pair_id']} ({worst['style']}) — error={worst['error']:.2f}px, "
              f"no clean periodic relationship (residual={worst['residual']:.3f}, "
              f"nearest_int={worst['nearest_int']})")


if __name__ == "__main__":
    main()
