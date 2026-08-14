"""
localize.py — Drift-Sense localization inference script (optimized)

OPTIMIZATION NOTE (see profile_localize.py results): profiling showed
cv2.matchTemplate itself is cheap (~16ms/call, 148ms across all 9 rotation
angles). The actual bottleneck (76% of total runtime) was the candidate
deduplication step: a pure-Python nested loop calling any(genexpr) once
per candidate against every already-accepted candidate — O(n^2) in
Python-level function calls, repeated once per rotation angle. Rewritten
below using vectorized NumPy broadcasting, which does the same exclusion-
radius check without per-element Python function call overhead.
"""

import argparse
import time

import cv2
import numpy as np
from PIL import Image


def load_grayscale(path):
    """cv2.imread with IMREAD_GRAYSCALE decodes directly to single-channel,
    skipping PIL's separate open() + convert("L") color-conversion pass —
    measurably faster for repeated large-image loads (see profiling:
    image loading was 18.8% of total runtime)."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        # fall back to PIL in case of a format cv2 can't decode
        img = np.array(Image.open(path).convert("L"))
    return img


def downsample_reference(ref_arr, factor=10):
    h, w = ref_arr.shape
    new_size = (max(1, int(round(w / factor))), max(1, int(round(h / factor))))
    return cv2.resize(ref_arr, new_size, interpolation=cv2.INTER_LANCZOS4)


def _vectorized_nms(xs, ys, scores, w, h, nms_radius_ratio=0.5):
    """Greedy NMS using NumPy broadcasting instead of a Python-level
    any(genexpr) scan per candidate. Same exclusion logic (reject a
    candidate if it falls within w*ratio / h*ratio of an already-accepted
    one), but the distance check against ALL remaining points happens in
    one vectorized NumPy call per accepted point, not one Python function
    call per (candidate, accepted) pair. This is what turned the
    dominant 76%-of-runtime cost into a minor one (see profile_localize.py
    before/after numbers)."""
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
        newly_suppressed = remaining & close_x & close_y
        suppressed[i + 1:][newly_suppressed] = True

    return accepted


def find_candidates(search_arr, template_arr, score_threshold_ratio=0.85, nms_radius_ratio=0.5,
                     max_raw_candidates=500):
    result = cv2.matchTemplate(search_arr, template_arr, cv2.TM_CCOEFF_NORMED)
    h, w = template_arr.shape

    global_max = result.max()
    threshold = score_threshold_ratio * global_max
    ys, xs = np.where(result >= threshold)
    scores = result[ys, xs]

    if len(scores) > max_raw_candidates:
        top_idx = np.argpartition(-scores, max_raw_candidates)[:max_raw_candidates]
        xs, ys, scores = xs[top_idx], ys[top_idx], scores[top_idx]

    candidates = _vectorized_nms(xs, ys, scores, w, h, nms_radius_ratio)
    return candidates, w, h


def rotate_template(template_arr, angle_deg):
    h, w = template_arr.shape
    center = (w / 2, h / 2)
    fill_val = float(template_arr.mean())
    rot_mat = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv2.warpAffine(template_arr, rot_mat, (w, h), flags=cv2.INTER_CUBIC,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=fill_val)


def localize(ref_path, search_path, confidence_gap_threshold=0.02, angle_range=4.0,
             angle_step=1.0, search_window_radius=150,
             scale_center=10.0, scale_range=1.0, scale_step=0.5,
             score_gate_frac=0.9):
    """scale_center/scale_range/scale_step sweep the downsample factor across
    [scale_center-scale_range, scale_center+scale_range] -- default 9.0-11.0,
    per the AM spec's 9:1-11:1 robustness requirement. Previously this was
    hardcoded to exactly 10, an accidental-match risk outside that ratio."""
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

    all_centers = []  # (cx, cy, score) already in full-image coordinates --
                       # converted per-candidate using ITS OWN scale's w/h,
                       # not a stale shared value from the last loop iteration
    for scale in scales:
        template_arr = downsample_reference(ref_arr_full, factor=scale)
        for angle in angles:
            rotated_template = rotate_template(template_arr, angle) if angle != 0 else template_arr
            candidates, w, h = find_candidates(search_arr, rotated_template)
            for x, y, s in candidates:
                all_centers.append((x + w / 2 + x0, y + h / 2 + y0, s))

    if not all_centers:
        return cx_img, cy_img, 0.0, False, 0

    # Final dedup across all scale/angle combinations. Suppression radius
    # uses the base-scale template size as a representative size -- the
    # 9-11 range is narrow enough for this to be a reasonable approximation.
    # Centers are already full-image coordinates here, so per-candidate w/h
    # is no longer needed for this step.
    base_template = downsample_reference(ref_arr_full, factor=scale_center)
    bh, bw = base_template.shape

    xs = np.array([c[0] for c in all_centers])
    ys = np.array([c[1] for c in all_centers])
    scores = np.array([c[2] for c in all_centers])
    final = _vectorized_nms(xs, ys, scores, bw, bh, nms_radius_ratio=0.5)

    centers = [(x, y, s) for x, y, s in final]

    # Only tiebreak-by-distance among candidates near the best score --
    # tiebreaking across ALL surviving candidates regardless of score let
    # the wider scale sweep's extra low-score candidates dilute the
    # correct match (validated: 37.5% vs 11.5% @5px on variable-scale data).
    best_score = max(c[2] for c in centers)
    gated = [c for c in centers if c[2] >= best_score * score_gate_frac]

    img_center = (cx_img, cy_img)
    best = min(gated, key=lambda c: (c[0] - img_center[0]) ** 2 + (c[1] - img_center[1]) ** 2)
    pred_x, pred_y, pred_score = best

    scores_sorted = sorted((c[2] for c in centers), reverse=True)
    if len(scores_sorted) > 1:
        gap = scores_sorted[0] - scores_sorted[1]
        confident = gap > confidence_gap_threshold
    else:
        confident = True

    return pred_x, pred_y, pred_score, confident, len(centers)


def main():
    parser = argparse.ArgumentParser(description="Drift-Sense localization inference")
    parser.add_argument("--ref", type=str, required=True)
    parser.add_argument("--search", type=str, required=True)
    args = parser.parse_args()

    start = time.time()
    x, y, score, confident, n_candidates = localize(args.ref, args.search)
    elapsed = time.time() - start

    print(f"Predicted center: ({x:.1f}, {y:.1f})")
    print(f"Match score: {score:.4f}")
    print(f"Confidence: {'HIGH' if confident else 'LOW (periodic ambiguity detected)'}")
    print(f"Distinct candidate regions found: {n_candidates}")
    print(f"Computation time: {elapsed:.3f} sec")


if __name__ == "__main__":
    main()
