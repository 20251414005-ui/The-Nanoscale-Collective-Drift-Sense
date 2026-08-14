"""
localize_rgb.py — RGB-aware Drift-Sense localization

Genuinely uses color information (not a grayscale-flattened shortcut):
runs NCC template matching independently on each of the R, G, B channels,
then combines the three per-pixel correlation maps into one combined score
before finding candidate peaks. This lets color contrast between materials/
layers (present in real optical microscope images, unlike SEM grayscale)
actually contribute to disambiguating otherwise-identical-looking periodic
regions — the whole point of the RGB bonus, rather than just accepting a
color file and silently discarding two of its three channels.

Usage:
    python localize_rgb.py --ref "path/to/ref.png" --search "path/to/search.png"
"""

import argparse
import time

import cv2
import numpy as np


def load_rgb(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)  # BGR, 3-channel
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return img


def downsample_reference_rgb(ref_arr, factor):
    h, w = ref_arr.shape[:2]
    new_size = (max(1, int(round(w / factor))), max(1, int(round(h / factor))))
    return cv2.resize(ref_arr, new_size, interpolation=cv2.INTER_LANCZOS4)


def _vectorized_nms(xs, ys, scores, w, h, nms_radius_ratio=0.5):
    order = np.argsort(-scores)
    xs, ys, scores = xs[order], ys[order], scores[order]
    n = len(xs)
    suppressed = np.zeros(n, dtype=bool)
    accepted = []
    x_thresh = w * nms_radius_ratio
    y_thresh = h * nms_radius_ratio
    for i in range(n):
        if suppressed[i]:
            continue
        accepted.append((xs[i], ys[i], scores[i]))
        if i + 1 >= n:
            break
        remaining = ~suppressed[i + 1:]
        close_x = np.abs(xs[i + 1:] - xs[i]) < x_thresh
        close_y = np.abs(ys[i + 1:] - ys[i]) < y_thresh
        suppressed[i + 1:][remaining & close_x & close_y] = True
    return accepted


def find_candidates_rgb(search_arr, template_arr, score_threshold_ratio=0.85,
                         nms_radius_ratio=0.5, max_raw_candidates=500):
    """Matches each of the 3 channels independently, then averages the
    correlation maps. Averaging (rather than picking the best single
    channel) rewards regions that are consistently good matches across
    ALL color channels -- reducing the chance a single noisy channel
    creates a false positive, while still letting real color-contrast
    differences between candidate sites pull the combined score apart."""
    h, w = template_arr.shape[:2]
    combined = None
    for c in range(3):
        result_c = cv2.matchTemplate(search_arr[:, :, c], template_arr[:, :, c], cv2.TM_CCOEFF_NORMED)
        combined = result_c if combined is None else combined + result_c
    combined /= 3.0

    global_max = combined.max()
    threshold = score_threshold_ratio * global_max
    ys, xs = np.where(combined >= threshold)
    scores = combined[ys, xs]

    if len(scores) > max_raw_candidates:
        top_idx = np.argpartition(-scores, max_raw_candidates)[:max_raw_candidates]
        xs, ys, scores = xs[top_idx], ys[top_idx], scores[top_idx]

    candidates = _vectorized_nms(xs, ys, scores, w, h, nms_radius_ratio)
    return candidates, w, h


def rotate_template_rgb(template_arr, angle_deg):
    h, w = template_arr.shape[:2]
    center = (w / 2, h / 2)
    fill_val = tuple(float(template_arr[:, :, c].mean()) for c in range(3))
    rot_mat = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv2.warpAffine(template_arr, rot_mat, (w, h), flags=cv2.INTER_CUBIC,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=fill_val)


def localize_rgb(ref_path, search_path, confidence_gap_threshold=0.02,
                  angle_range=4.0, angle_step=2.0, search_window_radius=None,
                  scale_range=(9.0, 10.0, 11.0)):
    search_arr_full = load_rgb(search_path)
    ref_arr_full = load_rgb(ref_path)

    H, W = search_arr_full.shape[:2]
    cx_img, cy_img = W / 2, H / 2

    # search_window_radius scales with image size if not explicitly given,
    # instead of a fixed 150px tuned only for 1000px-scale images.
    if search_window_radius is None:
        search_window_radius = int(0.15 * max(H, W))

    x0 = int(max(0, cx_img - search_window_radius))
    x1 = int(min(W, cx_img + search_window_radius))
    y0 = int(max(0, cy_img - search_window_radius))
    y1 = int(min(H, cy_img + search_window_radius))
    search_arr = search_arr_full[y0:y1, x0:x1]

    angles = np.arange(-angle_range, angle_range + 0.01, angle_step)

    all_candidates = []
    for scale in scale_range:
        template_arr = downsample_reference_rgb(ref_arr_full, factor=scale)
        for angle in angles:
            rotated = rotate_template_rgb(template_arr, angle) if angle != 0 else template_arr
            candidates, w, h = find_candidates_rgb(search_arr, rotated)
            all_candidates.extend([(x, y, s, w, h) for x, y, s in candidates])

    if not all_candidates:
        return cx_img, cy_img, 0.0, False, 0

    all_candidates.sort(key=lambda c: -c[2])
    final = []
    for x, y, s, cw, ch in all_candidates:
        too_close = any(
            abs(x - fx) < max(cw, fw) * 0.5 and abs(y - fy) < max(ch, fh) * 0.5
            for fx, fy, fs, fw, fh in final
        )
        if not too_close:
            final.append((x, y, s, cw, ch))

    centers = [(x + cw / 2 + x0, y + ch / 2 + y0, s) for x, y, s, cw, ch in final]

    img_center = (cx_img, cy_img)
    best = min(centers, key=lambda c: (c[0] - img_center[0]) ** 2 + (c[1] - img_center[1]) ** 2)
    pred_x, pred_y, pred_score = best

    scores_sorted = sorted((c[2] for c in centers), reverse=True)
    confident = (scores_sorted[0] - scores_sorted[1] > confidence_gap_threshold) if len(scores_sorted) > 1 else True

    return pred_x, pred_y, pred_score, confident, len(centers)


def main():
    parser = argparse.ArgumentParser(description="Drift-Sense RGB-aware localization")
    parser.add_argument("--ref", type=str, required=True)
    parser.add_argument("--search", type=str, required=True)
    args = parser.parse_args()

    start = time.time()
    x, y, score, confident, n_candidates = localize_rgb(args.ref, args.search)
    elapsed = time.time() - start

    print(f"Predicted center: ({x:.1f}, {y:.1f})")
    print(f"Match score: {score:.4f}")
    print(f"Confidence: {'HIGH' if confident else 'LOW (periodic ambiguity detected)'}")
    print(f"Distinct candidate regions found: {n_candidates}")
    print(f"Computation time: {elapsed:.3f} sec")


if __name__ == "__main__":
    main()
