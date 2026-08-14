"""
localize_dl.py — Drift-Sense DL-based localization inference
"""

import argparse
import time

import numpy as np
import torch
from PIL import Image

from model import SiameseLocalizer


def load_and_preprocess(ref_path, search_path):
    ref_img = Image.open(ref_path).convert("L")
    w, h = ref_img.size
    template_img = ref_img.resize((w // 10, h // 10), Image.LANCZOS)
    template = np.array(template_img, dtype=np.float32) / 255.0

    search_img = Image.open(search_path).convert("L")
    search = np.array(search_img, dtype=np.float32) / 255.0

    template_t = torch.from_numpy(template).unsqueeze(0).unsqueeze(0)
    search_t = torch.from_numpy(search).unsqueeze(0).unsqueeze(0)
    return template_t, search_t


def heatmap_to_prediction(heatmap_np, wt, ht, search_w, search_h,
                           heatmap_stride=8, search_window_radius=300):
    ho, wo = heatmap_np.shape
    cx_img, cy_img = search_w / 2, search_h / 2

    yy, xx = np.meshgrid(np.arange(ho), np.arange(wo), indexing="ij")
    pixel_x = (xx + wt / 2) * heatmap_stride
    pixel_y = (yy + ht / 2) * heatmap_stride
    dist_from_center = np.sqrt((pixel_x - cx_img) ** 2 + (pixel_y - cy_img) ** 2)
    mask = dist_from_center <= search_window_radius

    masked_heatmap = np.where(mask, heatmap_np, -np.inf)
    peak_idx = np.unravel_index(np.argmax(masked_heatmap), masked_heatmap.shape)
    peak_y, peak_x = peak_idx
    peak_val = heatmap_np[peak_y, peak_x]

    valid_vals = heatmap_np[mask]
    mean_val = valid_vals.mean()
    std_val = valid_vals.std() + 1e-8
    z_score = (peak_val - mean_val) / std_val
    confident = z_score > 3.0

    pred_x = (peak_x + wt / 2) * heatmap_stride
    pred_y = (peak_y + ht / 2) * heatmap_stride

    return pred_x, pred_y, float(peak_val), confident


def load_dl_model(weights_path, device="cpu"):
    """Load the model ONCE. Reuse this across many images — reloading
    torch.load() per-image (the original design) added pure disk-I/O and
    deserialization overhead to every single timed prediction, which has
    nothing to do with actual inference speed and was inflating reported
    per-pair time, especially over hundreds of evaluation pairs."""
    model = SiameseLocalizer().to(device)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def localize_dl_with_model(model, ref_path, search_path, heatmap_stride=8, device="cpu",
                            search_window_radius=300):
    """Same inference logic as before, but takes an already-loaded model
    instead of a weights path — use this in loops/batch evaluation."""
    template_t, search_t = load_and_preprocess(ref_path, search_path)
    template_t, search_t = template_t.to(device), search_t.to(device)

    with torch.no_grad():
        template_feat = model.encoder(template_t)
        heatmap = model(template_t, search_t)[0, 0]

    _, _, ht, wt = template_feat.shape
    heatmap_np = heatmap.cpu().numpy()
    search_h, search_w = search_t.shape[2], search_t.shape[3]

    return heatmap_to_prediction(heatmap_np, wt, ht, search_w, search_h,
                                  heatmap_stride, search_window_radius)


def localize_dl(ref_path, search_path, weights_path, heatmap_stride=8, device="cpu",
                 search_window_radius=300):
    """Kept for single-image CLI use / backward compatibility. For
    evaluating many pairs, use load_dl_model() once + localize_dl_with_model()
    per pair instead (see evaluate_dl.py) — this version still reloads the
    model each call, which is fine for a single lookup but wasteful in a loop."""
    model = load_dl_model(weights_path, device=device)
    return localize_dl_with_model(model, ref_path, search_path, heatmap_stride, device,
                                   search_window_radius)


def main():
    parser = argparse.ArgumentParser(description="Drift-Sense DL localization inference")
    parser.add_argument("--ref", type=str, required=True)
    parser.add_argument("--search", type=str, required=True)
    parser.add_argument("--weights", type=str, default="model_weights/siamese_localizer.pt")
    parser.add_argument("--device", type=str, default=None,
                         help="cuda or cpu; defaults to cuda if available")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    start = time.time()
    x, y, score, confident = localize_dl(args.ref, args.search, args.weights, device=device)
    elapsed = time.time() - start

    print(f"Device: {device}")
    print(f"Predicted center: ({x:.1f}, {y:.1f})")
    print(f"Peak heatmap score: {score:.4f}")
    print(f"Confidence: {'HIGH' if confident else 'LOW (periodic ambiguity detected)'}")
    print(f"Computation time: {elapsed:.3f} sec")


if __name__ == "__main__":
    main()
