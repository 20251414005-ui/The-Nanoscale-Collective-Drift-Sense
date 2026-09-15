"""
profile_dl.py -- CUDA/CPU profiling for the DriftSense DL pipeline.

Matched against your actual localize_dl.py:
    load_dl_model(weights_path, device="cpu") -> model
    localize_dl_with_model(model, ref_path, search_path, heatmap_stride=8,
                            device="cpu", search_window_radius=300)
        -> (pred_x, pred_y, peak_val, confident)

WHAT THIS DOES
    Wraps localize_dl_with_model() in torch.profiler and reports a
    per-operator time breakdown, so you can say WHY inference takes
    ~0.041-0.050s/pair, not just THAT it does. Useful follow-on to the
    NMS bottleneck you already found in localize.py via profile_localize.py
    -- this is the DL-pipeline equivalent.

USAGE
    python profile_dl.py --manifest data/train_canonical/manifest.csv \
        --weights model_weights/siamese_localizer_canonical.pt \
        --n-pairs 20 --device cuda

OUTPUT
    - Console table: top-10 PyTorch ops by CUDA (or CPU) time
    - profiler_trace.json -- open at chrome://tracing or https://ui.perfetto.dev
      for a visual timeline
    - Optional deeper trace: run this whole script under
        nsys profile -o dl_nsys_report python profile_dl.py ...
      Nsight Systems is a separate NVIDIA download (not pip-installable),
      https://developer.nvidia.com/nsight-systems -- grab the CLI-only
      Linux build for use inside WSL2, then view the .nsys-rep file on
      the Windows side with the full Nsight Systems GUI.
"""

import argparse
import csv
import sys
from pathlib import Path

import torch
from torch.profiler import profile, ProfilerActivity, record_function

try:
    from localize_dl import load_dl_model, localize_dl_with_model
except ImportError as e:
    print(f"[FAIL] Could not import from localize_dl.py: {e}")
    sys.exit(1)


def load_pairs(manifest_path: str, n_pairs: int):
    pairs = []
    with open(manifest_path, newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= n_pairs:
                break
            pairs.append((row["ref_path"], row["search_path"]))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--weights", default="model_weights/siamese_localizer_canonical.pt")
    ap.add_argument("--n-pairs", type=int, default=20,
                     help="Profile this many pairs (keep small -- profiling adds overhead)")
    ap.add_argument("--device", default=None,
                     help="cuda or cpu; defaults to cuda if available, matching localize_dl.py's own default")
    ap.add_argument("--search-window-radius", type=float, default=300,
                     help="Must match what evaluate_dl.py uses, or the profiled heatmap search differs from your reported numbers")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    print(f"[INFO] Loading model from {args.weights}")
    model = load_dl_model(args.weights, device=device)

    pairs = load_pairs(args.manifest, args.n_pairs)
    print(f"[INFO] Profiling {len(pairs)} pairs")

    activities = [ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(ProfilerActivity.CUDA)

    # Warm-up run outside the profiler -- first CUDA call includes kernel
    # compilation/caching overhead that would otherwise skew the numbers
    localize_dl_with_model(model, *pairs[0], device=device,
                            search_window_radius=args.search_window_radius)

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        for ref_path, search_path in pairs:
            with record_function("localize_dl_with_model"):
                localize_dl_with_model(model, ref_path, search_path, device=device,
                                        search_window_radius=args.search_window_radius)

    sort_key = "cuda_time_total" if device == "cuda" else "cpu_time_total"
    print("\n" + "=" * 80)
    print(f"TOP 10 OPERATORS BY {sort_key.upper()}")
    print("=" * 80)
    print(prof.key_averages().table(sort_by=sort_key, row_limit=10))

    trace_path = Path("profiler_trace.json")
    prof.export_chrome_trace(str(trace_path))
    print(f"\n[OK] Trace exported to {trace_path.resolve()}")
    print("     Open in chrome://tracing or https://ui.perfetto.dev for a visual timeline")


if __name__ == "__main__":
    main()
