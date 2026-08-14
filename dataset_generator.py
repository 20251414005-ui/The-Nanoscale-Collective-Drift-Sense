"""
dataset_generator.py — Drift-Sense synthetic dataset generator

Generates labeled Reference/Search image pairs simulating wafer-inspection
navigation-error recovery, for DRAM-style or FinFET-style die layouts.

Usage:
    python dataset_generator.py --style dram --num_pairs 100 --output_dir data/train
    python dataset_generator.py --style finfet --num_pairs 100 --output_dir data/train
    python dataset_generator.py --style both --num_pairs 200 --output_dir data/train

Each run appends to manifest.csv in the output directory with columns:
    pair_id, style, ref_path, search_path, gt_x, gt_y, pitch,
    noise_sigma_ref, noise_sigma_search, rotation_deg, blur_ref, blur_search
"""

import argparse
import csv
import os
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


# ---------------------------------------------------------------------------
# Pattern generators
# ---------------------------------------------------------------------------

def maybe_collapse_gap(collapse_prob=0.12):
    """Flat per-junction probability that a defect (partial line fusion)
    occurs here. Independent of physical gap size -- appropriate when line
    width is small relative to pitch, so a size-gated threshold would almost
    never fire (confirmed via visual test)."""
    return random.random() < collapse_prob


def _draw_partial_bridge(draw, axis, gap_lo, gap_hi, size):
    """Fill a random sub-span of the gap [gap_lo, gap_hi] -- not the whole
    gap -- to look like a local defect/bridge rather than a full clean merge.
    axis='h' for horizontal lines (bridge is a horizontal band),
    axis='v' for vertical lines (bridge is a vertical band)."""
    gap_size = gap_hi - gap_lo
    if gap_size <= 1:
        return
    bridge_frac = random.uniform(0.4, 1.0)
    bridge_size = gap_size * bridge_frac
    start = gap_lo + random.uniform(0, gap_size - bridge_size)
    end = start + bridge_size
    if axis == "h":
        draw.rectangle([0, start, size, end], fill=200)
    else:
        draw.rectangle([start, 0, end, size], fill=200)


def draw_dram_pattern(size, pitch, line_width=3, dot_radius=3, jitter=0.6,
                       line_width_jitter_frac=0.5,
                       collapse_prob=0.12):
    """DRAM-style periodic grid: horizontal word-lines + vertical bit-lines
    crossing at right angles, with a contact/via dot at every intersection.
    Some junctions get a random partial line-fusion defect to break perfect
    periodicity."""
    img = Image.new("L", (size, size), color=20)
    draw = ImageDraw.Draw(img)

    y = 0
    prev_jy, prev_w = None, None
    while y < size:
        jy = y + random.uniform(-jitter, jitter)
        w = max(1, line_width * (1 + random.uniform(-line_width_jitter_frac, line_width_jitter_frac)))
        draw.line([(0, jy), (size, jy)], fill=200, width=int(round(w)))

        if prev_jy is not None and maybe_collapse_gap(collapse_prob):
            gap_lo = prev_jy + prev_w / 2
            gap_hi = jy - w / 2
            _draw_partial_bridge(draw, "h", gap_lo, gap_hi, size)

        prev_jy, prev_w = jy, w
        y += pitch

    x = 0
    prev_jx, prev_w = None, None
    while x < size:
        jx = x + random.uniform(-jitter, jitter)
        w = max(1, line_width * (1 + random.uniform(-line_width_jitter_frac, line_width_jitter_frac)))
        draw.line([(jx, 0), (jx, size)], fill=200, width=int(round(w)))

        if prev_jx is not None and maybe_collapse_gap(collapse_prob):
            gap_lo = prev_jx + prev_w / 2
            gap_hi = jx - w / 2
            _draw_partial_bridge(draw, "v", gap_lo, gap_hi, size)

        prev_jx, prev_w = jx, w
        x += pitch

    # vias/contacts at intersections -- unchanged from original
    y = 0
    while y < size:
        x = 0
        while x < size:
            jx = x + random.uniform(-jitter, jitter)
            jy = y + random.uniform(-jitter, jitter)
            draw.ellipse([jx - dot_radius, jy - dot_radius,
                          jx + dot_radius, jy + dot_radius], fill=255)
            x += pitch
        y += pitch

    return img


def draw_finfet_pattern(size, fin_pitch, gate_spacing=None, fin_width=2,
                         gate_width=6, jitter=0.5,
                         fin_width_jitter_frac=0.5,
                         collapse_prob=0.12):
    """FinFET-style pattern: dense parallel vertical fin lines, crossed by
    one or two horizontal gate bars at the intersection region. Some fins
    get a random partial fusion defect (see draw_dram_pattern for the same
    idea) to break the translation symmetry along the fin axis -- FinFET's
    main periodicity weak point."""
    img = Image.new("L", (size, size), color=20)
    draw = ImageDraw.Draw(img)

    # dense vertical fins, with occasional partial fusion between neighbors
    x = 0
    prev_jx, prev_w = None, None
    while x < size:
        jx = x + random.uniform(-jitter, jitter)
        w = max(1, fin_width * (1 + random.uniform(-fin_width_jitter_frac, fin_width_jitter_frac)))
        draw.line([(jx, 0), (jx, size)], fill=180, width=int(round(w)))

        if prev_jx is not None and maybe_collapse_gap(collapse_prob):
            gap_lo = prev_jx + prev_w / 2
            gap_hi = jx - w / 2
            _draw_partial_bridge(draw, "v", gap_lo, gap_hi, size)

        prev_jx, prev_w = jx, w
        x += fin_pitch

    # one or two horizontal gate bars -- unchanged from original
    if gate_spacing is None:
        gate_spacing = size // 3
    y = gate_spacing
    while y < size:
        jy = y + random.uniform(-jitter, jitter)
        draw.rectangle([0, jy - gate_width / 2, size, jy + gate_width / 2], fill=250)
        y += gate_spacing * random.choice([1, 2])

    return img


PATTERN_FUNCS = {
    "dram": draw_dram_pattern,
    "finfet": draw_finfet_pattern,
}


# ---------------------------------------------------------------------------
# Degradation steps (each justified — see citations.md)
# ---------------------------------------------------------------------------

def edge_brighten(img_arr, strength=20):
    """Mimic SEM secondary-electron edge contrast: brighten pixels near edges."""
    img = Image.fromarray(img_arr.astype(np.uint8))
    edges = img.filter(ImageFilter.FIND_EDGES)
    edges_arr = np.array(edges).astype(np.float32)
    out = img_arr.astype(np.float32) + strength * (edges_arr / 255.0)
    return np.clip(out, 0, 255)


def add_sensor_noise(img_arr, sigma):
    """Independent Gaussian sensor noise. Must be called separately for the
    reference and search image — never share a noise draw between them."""
    noise = np.random.normal(0, sigma, img_arr.shape)
    return np.clip(img_arr.astype(np.float32) + noise, 0, 255)


def apply_blur(img_arr, sigma):
    return np.array(Image.fromarray(img_arr.astype(np.uint8)).filter(ImageFilter.GaussianBlur(sigma)))


def apply_rotation(img_arr, angle_deg):
    img = Image.fromarray(img_arr.astype(np.uint8))
    rotated = img.rotate(angle_deg, resample=Image.BICUBIC, fillcolor=20)
    return np.array(rotated)


def add_shot_noise(img_arr, dose):
    """Poisson shot noise, signal-dependent (brighter pixels have
    proportionally less relative noise) -- the physically correct SEM
    noise model, per Foi et al. 2008's Poisson-Gaussian framework (see
    citations.md). `dose` is a proxy for electron count/dwell time --
    higher dose (slower, careful scan) means less relative noise."""
    img_f = np.clip(img_arr, 0, 255).astype(np.float64)
    counts = np.clip(img_f / 255.0 * dose, 0, None)
    noisy_counts = np.random.poisson(counts).astype(np.float64)
    noisy = noisy_counts / dose * 255.0
    return np.clip(noisy, 0, 255).astype(np.float32)


def add_charging_streaks(img_arr, streak_prob, intensity):
    """Occasional bright horizontal streaks from local sample charging
    (common on insulating/oxide regions under e-beam). streak_prob is
    expected streaks per 100 rows; intensity scales streak brightness."""
    if streak_prob <= 0 or intensity <= 0:
        return img_arr
    h, w = img_arr.shape
    out = img_arr.astype(np.float64).copy()
    expected = streak_prob * (h / 100.0)
    n_streaks = np.random.poisson(max(expected, 0))
    for _ in range(n_streaks):
        row = random.randint(0, h - 1)
        band = max(1, int(np.random.normal(2, 1)))
        lo, hi = max(row - band, 0), min(row + band, h)
        out[lo:hi, :] += intensity * random.uniform(0.5, 1.0) * 255.0 / 10.0
    return np.clip(out, 0, 255).astype(np.float32)


def add_salt_and_pepper_noise(img_arr, prob):
    """Impulse noise: a fraction `prob` of pixels forced to 0 or 255 --
    a stand-in for dead/hot detector pixels or discharge events."""
    if prob <= 0:
        return img_arr
    out = img_arr.copy()
    hit = np.random.random(img_arr.shape) < prob
    salt = np.random.random(img_arr.shape) < 0.5
    out[hit & salt] = 255
    out[hit & ~salt] = 0
    return out.astype(np.float32)


def apply_gamma(img_arr, gamma):
    """Nonlinear contrast/brightness response curve -- detector gain
    nonlinearity or contrast/brightness knob mis-calibration. gamma=1.0
    is a no-op."""
    if gamma == 1.0:
        return img_arr
    norm = np.clip(img_arr, 0, 255).astype(np.float64) / 255.0
    out = np.power(norm, gamma) * 255.0
    return np.clip(out, 0, 255).astype(np.float32)


# ---------------------------------------------------------------------------
# Pair generation
# ---------------------------------------------------------------------------

def generate_pair(style, search_size=1000):
    """
    Per official spec (PPT slide 5): BOTH reference and search images are
    1000x1000 pixels.
      - Reference = full-resolution 1000x1000 crop (1nm/px, 1um x 1um FOV),
        taken directly from the big canvas — NOT resized down.
      - Search = the same larger die region downsampled 10x to 1000x1000
        (10nm/px, 10um x 10um FOV). Because of that 10x resolution drop,
        the reference pattern appears inside the search image shrunk to
        roughly a 100x100 pixel footprint — that's a consequence of the
        ratio, not something we set directly.
    """
    pattern_func = PATTERN_FUNCS[style]

    # Pitch values grounded in published device dimensions (see citations.md):
    # - DRAM buried word-line pitch bottlenecks near 40nm at advanced nodes
    # - FinFET fin pitch is in the 20s-30nm range at 5nm-class nodes
    # At our chosen 1nm/pixel reference resolution, these map directly to
    # pixel pitch. (Earlier version used 8-14px, an unjustified guess that
    # aliased to sub-pixel period after the 10x downsample — fixed here.)
    if style == "dram":
        pitch = random.randint(35, 45)
        pattern_kwargs = dict(pitch=pitch)
    else:
        pitch = random.randint(20, 30)
        pattern_kwargs = dict(fin_pitch=pitch, gate_spacing=random.randint(45, 55))

    # Big canvas = 10x the search_size, so a search_size-sized crop of it
    # represents the same pixel count as the full canvas resized down.
    scale = random.uniform(9.0, 11.0)
    big_canvas_size = int(round(search_size * scale))
    full_layout = pattern_func(big_canvas_size, **pattern_kwargs)
    full_arr = np.array(full_layout, dtype=np.float32)

    # Reference: a direct 1000x1000 crop at FULL resolution (no resize) —
    # this is the "100x" high-magnification capture.
    ref_px = search_size  # 1000, per spec
    max_xy = big_canvas_size - ref_px

    # Ground truth placement models a real navigation-error scenario: the
    # search image is centered on the EXPECTED target location, and the
    # true reference site is offset only by a small drift amount (thermal
    # expansion, vibration, mechanical slack) — not placed uniformly
    # anywhere in the frame. This also makes the "closest to center"
    # tiebreak rule from the spec meaningful, since the true answer is
    # genuinely expected to be near center in the real problem.
    center_xy = max_xy / 2
    drift_budget = max_xy * 0.15  # max drift = 15% of canvas from center
    gt_x_full = int(np.clip(center_xy + random.uniform(-drift_budget, drift_budget), 0, max_xy))
    gt_y_full = int(np.clip(center_xy + random.uniform(-drift_budget, drift_budget), 0, max_xy))

    ref_crop = full_arr[gt_y_full:gt_y_full + ref_px, gt_x_full:gt_x_full + ref_px]
    ref_arr = ref_crop.copy()  # already 1000x1000 — no resizing

    # Search: the ENTIRE big canvas downsampled to 1000x1000 —
    # this is the "10x" low-magnification capture.
    search_img = Image.fromarray(full_arr.astype(np.uint8)).resize((search_size, search_size), Image.LANCZOS)
    search_arr = np.array(search_img, dtype=np.float32)

    # Ground truth: where the reference crop's CENTER lands in search-image
    # pixel coordinates (divide by 10 since search is downsampled 10x).
    gt_center_x = (gt_x_full + ref_px / 2) / scale
    gt_center_y = (gt_y_full + ref_px / 2) / scale

    # --- degradations, independently applied ---
    rotation_deg = random.uniform(-3, 3)  # small capture-angle variation
    ref_arr = apply_rotation(ref_arr, rotation_deg)

    ref_arr = edge_brighten(ref_arr, strength=random.uniform(15, 25))
    search_arr = edge_brighten(search_arr, strength=random.uniform(10, 20))

    noise_sigma_ref = random.uniform(4, 8)      # sharper capture -> less noise
    noise_sigma_search = random.uniform(10, 18)  # lower mag -> more noise
    ref_arr = add_sensor_noise(ref_arr, noise_sigma_ref)
    search_arr = add_sensor_noise(search_arr, noise_sigma_search)

    blur_ref = random.uniform(0.2, 0.6)
    blur_search = random.uniform(0.4, 0.9)
    ref_arr = apply_blur(ref_arr, blur_ref)
    search_arr = apply_blur(search_arr, blur_search)

    # Additional spec-listed degradations: physically-correct Poisson shot
    # noise, charging streaks, salt-and-pepper, and gamma/contrast variation.
    dose_ref = random.uniform(300, 600)      # careful capture -> less shot noise
    dose_search = random.uniform(80, 200)     # fast scan -> more shot noise
    ref_arr = add_shot_noise(ref_arr, dose_ref)
    search_arr = add_shot_noise(search_arr, dose_search)

    charging_intensity = random.uniform(0.3, 0.8) if random.random() < 0.3 else 0.0
    charging_prob = random.uniform(0.05, 0.4) if charging_intensity > 0 else 0.0
    search_arr = add_charging_streaks(search_arr, charging_prob, charging_intensity)

    salt_pepper_prob = random.uniform(0.0002, 0.003) if random.random() < 0.25 else 0.0
    search_arr = add_salt_and_pepper_noise(search_arr, salt_pepper_prob)

    gamma_ref = random.uniform(0.9, 1.1)
    gamma_search = random.uniform(0.85, 1.15)
    ref_arr = apply_gamma(ref_arr, gamma_ref)
    search_arr = apply_gamma(search_arr, gamma_search)

    return (ref_arr.astype(np.uint8), search_arr.astype(np.uint8),
            (gt_center_x, gt_center_y),
            dict(pitch=pitch, scale=round(scale, 3), noise_sigma_ref=round(noise_sigma_ref, 2),
                 noise_sigma_search=round(noise_sigma_search, 2),
                 rotation_deg=round(rotation_deg, 2),
                 blur_ref=round(blur_ref, 2), blur_search=round(blur_search, 2),
                dose_ref=round(dose_ref, 1), dose_search=round(dose_search, 1),
                charging_intensity=round(charging_intensity, 3),
                salt_pepper_prob=round(salt_pepper_prob, 4),
                gamma_ref=round(gamma_ref, 3), gamma_search=round(gamma_search, 3)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Drift-Sense synthetic dataset generator")
    parser.add_argument("--style", choices=["dram", "finfet", "both"], default="both",
                         help="Die architecture style to generate")
    parser.add_argument("--num_pairs", type=int, default=30,
                         help="Number of image pairs to generate")
    parser.add_argument("--output_dir", type=str, default="data/train",
                         help="Directory to save generated pairs and manifest.csv")
    parser.add_argument("--seed", type=int, default=None,
                         help="Random seed for reproducibility")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    manifest_path = os.path.join(args.output_dir, "manifest.csv")
    file_exists = os.path.exists(manifest_path)

    existing_count = 0
    if file_exists:
        with open(manifest_path, "r") as f:
            existing_count = sum(1 for _ in f) - 1  # minus header

    with open(manifest_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["pair_id", "style", "ref_path", "search_path",
                              "gt_x", "gt_y", "pitch", "scale", "noise_sigma_ref",
                              "noise_sigma_search", "rotation_deg", "blur_ref", "blur_search",
                              "dose_ref", "dose_search", "charging_intensity",
                              "salt_pepper_prob", "gamma_ref", "gamma_search"])

        for i in range(args.num_pairs):
            style = args.style if args.style != "both" else random.choice(["dram", "finfet"])
            pair_id = existing_count + i
            ref_arr, search_arr, (gt_x, gt_y), params = generate_pair(style)

            ref_path = os.path.join(args.output_dir, f"ref_{pair_id:04d}.png")
            search_path = os.path.join(args.output_dir, f"search_{pair_id:04d}.png")
            Image.fromarray(ref_arr).save(ref_path)
            Image.fromarray(search_arr).save(search_path)

            writer.writerow([pair_id, style, ref_path, search_path,
                              round(gt_x, 2), round(gt_y, 2), params["pitch"], params["scale"],
                              params["noise_sigma_ref"], params["noise_sigma_search"],
                              params["rotation_deg"], params["blur_ref"], params["blur_search"],
                              params["dose_ref"], params["dose_search"], params["charging_intensity"],
                              params["salt_pepper_prob"], params["gamma_ref"], params["gamma_search"]])

            print(f"[{i+1}/{args.num_pairs}] Generated pair {pair_id} ({style}) "
                  f"-> ground truth ({gt_x:.1f}, {gt_y:.1f})")

    print(f"\nDone. {args.num_pairs} pairs written to {args.output_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()