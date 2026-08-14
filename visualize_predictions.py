"""
visualize_predictions.py — draws predicted-location boxes on the search
image so results can be checked VISUALLY when no numeric ground truth
exists (e.g. AI-generated test pairs with no recorded true coordinates).

Usage:
    python visualize_predictions.py --search "data/rgb_test/Search Image.png" \
        --predictions 637.5 609.5 grayscale 583.0 589.0 rgb \
        --box_size 120 --output data/rgb_test/predictions_overlay.png
"""

import argparse

import cv2
import numpy as np


COLORS = {
    "grayscale": (0, 0, 255),   # red (BGR)
    "rgb": (0, 255, 0),          # green
}
DEFAULT_COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--search", type=str, required=True)
    parser.add_argument("--predictions", nargs="+", required=True,
                         help="Triplets: x y label, repeated. e.g. 637.5 609.5 grayscale 583.0 589.0 rgb")
    parser.add_argument("--box_size", type=int, default=120,
                         help="Side length of the drawn box, in pixels")
    parser.add_argument("--output", type=str, default="predictions_overlay.png")
    args = parser.parse_args()

    img = cv2.imread(args.search, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not load {args.search}")

    vals = args.predictions
    if len(vals) % 3 != 0:
        raise ValueError("Predictions must come in (x, y, label) triplets")

    half = args.box_size // 2
    for i in range(0, len(vals), 3):
        x, y, label = float(vals[i]), float(vals[i + 1]), vals[i + 2]
        color = COLORS.get(label, DEFAULT_COLORS[(i // 3) % len(DEFAULT_COLORS)])
        pt1 = (int(x - half), int(y - half))
        pt2 = (int(x + half), int(y + half))
        cv2.rectangle(img, pt1, pt2, color, thickness=4)
        cv2.putText(img, label, (pt1[0], max(0, pt1[1] - 10)),
                     cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
        # small crosshair at the exact predicted center
        cv2.drawMarker(img, (int(x), int(y)), color, cv2.MARKER_CROSS, 20, 3)

    cv2.imwrite(args.output, img)
    print(f"Saved overlay to: {args.output}")


if __name__ == "__main__":
    main()
