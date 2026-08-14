"""
swap_test.py — sanity check for the DL localizer.

If the model learned real matching, giving it two DIFFERENT reference
images against the SAME search image should produce two DIFFERENT
predicted locations (since each reference truly belongs at a different
spot). If the model instead learned the cheap shortcut of "always guess
near the search image's center" (possible since our training data
constrains ground truth to a center-biased drift radius), the prediction
will barely change no matter which reference is given.

Usage:
    python swap_test.py --manifest data/train/manifest.csv --weights model_weights/siamese_localizer.pt
"""

import argparse
import csv

from localize_dl import localize_dl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--weights", type=str, default="model_weights/siamese_localizer.pt")
    parser.add_argument("--n_pairs", type=int, default=5, help="Number of search images to test")
    args = parser.parse_args()

    with open(args.manifest, "r") as f:
        rows = list(csv.DictReader(f))

    print("Testing whether predictions change when the reference image changes")
    print("(same search image, different references — a real matcher should")
    print(" give different answers; a shortcut-learner will give near-identical ones)\n")

    for i in range(min(args.n_pairs, len(rows))):
        search_path = rows[i]["search_path"]
        true_gt = (float(rows[i]["gt_x"]), float(rows[i]["gt_y"]))

        # correct reference (should predict near true_gt)
        correct_ref = rows[i]["ref_path"]
        x1, y1, _, _ = localize_dl(correct_ref, search_path, args.weights)

        # a DIFFERENT reference, paired with the SAME search image
        other_idx = (i + 1) % len(rows)
        wrong_ref = rows[other_idx]["ref_path"]
        x2, y2, _, _ = localize_dl(wrong_ref, search_path, args.weights)

        shift = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

        print(f"search {i}: true_gt={true_gt}")
        print(f"  correct ref -> pred ({x1:.1f}, {y1:.1f})")
        print(f"  DIFFERENT ref -> pred ({x2:.1f}, {y2:.1f})")
        print(f"  shift between the two predictions: {shift:.1f}px")
        print(f"  {'LIKELY SHORTCUT (barely moved)' if shift < 20 else 'RESPONDS TO REFERENCE (moved meaningfully)'}\n")


if __name__ == "__main__":
    main()