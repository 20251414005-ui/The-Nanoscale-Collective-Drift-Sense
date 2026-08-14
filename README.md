# DriftSense

**SEMICON India 2026 Hackathon — Applied Materials Problem Statement PS-2**
**Team:** Prasad, Pratyush

DriftSense locates a shrunk reference pattern inside a larger, noisier search
image of a semiconductor wafer. Given a 1000×1000 reference image and a
1000×1000 search image where the reference appears at roughly 10x lower
magnification (validated across a 9–11x range), the pipeline outputs the
center (x, y) of the matching region in the search image.

Two independent approaches are implemented and evaluated side by side:

- **Classical** — NCC/ZNCC template matching with a scale + rotation sweep
  and a score-gated tiebreak (`localize.py`)
- **Deep learning** — a Siamese network trained to localize the match
  (`localize_dl.py`, `model.py`, `train_model.py`)

## Results (canonical 800-pair dataset, seed=2026)

`data/train_canonical/` combines defect injection, variable 9–11x scale, and
the full spec-listed degradation set (Poisson shot noise, charging streaks,
salt-and-pepper, gamma). All numbers below are reproducible from this
dataset and its manifest.

| Metric | Classical (`localize.py`) | DL (`siamese_localizer_canonical.pt`) |
|---|---|---|
| @1px | 39.0% | 3.4% |
| @2px | 39.2% | 13.6% |
| @4px | 39.5% | 42.2% |
| @5px | **40.0%** | **56.2%** |
| @10px | 40.2% | 84.9% |
| Avg time / pair | 0.170 s | 0.041–0.050 s |
| Low-confidence flagged | 219 / 800 | 1 / 800 |

The DL model wins on both accuracy and speed once retrained on
defect-injected, variable-scale data. Full development history and root-cause
analysis of earlier failure modes are in
[`docs/DriftSense_Failure_Case_Analysis.pdf`](docs/DriftSense_Failure_Case_Analysis.pdf).

## Hardware, environment, and timing method

- **GPU:** NVIDIA RTX 5060 (personal laptop), CUDA 13.0
- **Development/results environment:** WSL2 (Ubuntu), Python 3.12.13,
  `torch==2.13.0+cu130`, `torchvision==0.28.0+cu130` — the numbers in the
  results table above were produced in this exact environment.
- **Timing method:** per-pair wall-clock time via `time.time()`, measured
  inside `evaluate.py`/`evaluate_dl.py` around the single call to
  `localize()`/`localize_dl_with_model()` only (excludes model/weight
  loading, which happens once outside the timed loop).
- **CPU fallback:** both pipelines run without a GPU (`--device cpu` for the
  DL pipeline; the classical pipeline is CPU-only by design), at reduced DL
  speed.

## Repository structure

```
drift-sense/
├── dataset_generator.py       # synthetic DRAM/FinFET pair generator (spec calls this generate_dataset.py — see note below)
├── localize.py                # classical NCC-based localizer
├── model.py                   # Siamese network architecture
├── train_model.py             # DL training script
├── localize_dl.py             # DL inference / localization
├── evaluate.py                # classical pipeline evaluation, writes results/predictions.csv
├── evaluate_dl.py             # DL pipeline evaluation, writes results/predictions_dl.csv
├── merge_results.py           # joins manifest + both prediction sets into results/full_results.csv
├── pr_curve.py                # precision-recall curve over confidence threshold
├── model_weights/
│   └── siamese_localizer_canonical.pt   # final trained weights used for reported results
├── data/
│   └── sample_pairs/           # ~30 representative reference/search image pairs
│       └── manifest.csv        # manifest for the shipped sample (full 800-row manifest reproducible via seed 2026)
├── results/
│   ├── predictions.csv         # classical pipeline per-pair predictions
│   ├── predictions_dl.csv      # DL pipeline per-pair predictions
│   └── full_results.csv        # merged: paths, ground truth, generation metadata, both predictions
├── docs/
│   ├── DriftSense_Failure_Case_Analysis.pdf
│   └── DriftSense_Citations.pdf
├── requirements.txt
└── README.md
```

> **Note on filenames:** the problem statement's suggested entry-point name is
> `generate_dataset.py`; this repository uses `dataset_generator.py` for the
> same functionality.

## Setup

Tested on WSL2 (Ubuntu), Python 3.12.13, with an RTX 5060 GPU. CUDA is
optional — the classical pipeline is CPU-only and the DL pipeline falls back
to CPU automatically if no compatible GPU is available.

```bash
git clone <repo-url>
cd drift-sense

python3 -m venv drift-sense-env
source drift-sense-env/bin/activate

pip install -r requirements.txt
```

**PyTorch is installed separately**, since CUDA-tagged builds are not
resolvable from PyPI and must come from PyTorch's own index:

```bash
# GPU (CUDA 13.0) — the exact build tested for this submission
pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu130

# CPU-only (no compatible GPU/driver)
pip install torch==2.13.0 torchvision==0.28.0
```

If your CUDA version differs, check it first (`python3 -c "import torch;
print(torch.version.cuda)"` inside an environment where GPU torch is already
installed) and substitute the matching tag in the `--index-url` above.

## Dry-run verification (clean environment)

To confirm the submission is runnable outside the original development
machine, the following was independently verified in a fresh folder
(`~/drift-sense-clean-test`), containing only the files listed in the
repository structure above, with a brand-new virtual environment
(Python 3.14.4 in this specific verification run — noted separately from the
3.12.13 environment that produced the reported accuracy numbers, to confirm
the code is not version-pinned):

1. `pip install -r requirements.txt` followed by the separate torch/
   torchvision install — succeeded cleanly.
2. `python3 -c "import torch; ... torch.cuda.is_available()"` — returned
   `True`, confirming GPU access works from a fresh environment, not just
   the original dev venv.
3. `evaluate.py --manifest data/sample_pairs/manifest.csv` — ran to
   completion against the shipped 30-pair sample.
4. `evaluate_dl.py --manifest data/sample_pairs/manifest.csv` — ran to
   completion; sample-size accuracy on 30 pairs (43.3% @5px) is consistent
   with expected variance from the full 800-pair result (56.2% @5px).
5. `train_model.py --manifest data/sample_pairs/manifest.csv --epochs 2
   --output /tmp/dryrun_weights.pt` — started, correctly held out validation
   pairs never seen during training, completed 2 epochs, and saved a
   checkpoint to a scratch path without touching the shipped
   `siamese_localizer_canonical.pt`.

All three scripts run using only `--manifest` and standard flags — no
source-code edits or hardcoded paths were needed, confirmed separately via
a grep audit of the codebase.

## Reproducing the full results

**Classical pipeline:**
```bash
python evaluate.py --manifest data/train_canonical/manifest.csv --csv-out results/predictions.csv
```

**DL pipeline** (weights default to the canonical model; override with `--weights` if needed):
```bash
python evaluate_dl.py --manifest data/train_canonical/manifest.csv --csv-out results/predictions_dl.csv
```

**Merge both prediction sets with ground truth and generation metadata into one file:**
```bash
python merge_results.py --manifest data/train_canonical/manifest.csv \
  --classical results/predictions.csv --dl results/predictions_dl.csv --out results/full_results.csv
```

**Precision-recall curve (classical confidence threshold):**
```bash
python pr_curve.py
```

Both evaluation scripts report accuracy at 1/2/4/5/10/20/50/100/150px
thresholds, plus mean, median, and worst-case error, and average inference
time per pair. Manifests without ground-truth columns (`gt_x`/`gt_y`) are
supported directly — predictions are still computed and written, with
accuracy statistics skipped rather than the script failing.

## Key technical findings

1. **Dense periodicity is the core failure mode**, confirmed independently
   across the classical and DL pipelines and across both teammates'
   algorithms cross-tested on both datasets — not a tile-confusion or
   window-size issue.
2. **Gap-collapse defect injection** (partial-bridge fusion of adjacent
   DRAM lines / FinFET fins, `collapse_prob=0.12`) breaks this periodicity
   and is the single largest driver of accuracy gains.
3. **A scale sweep alone regressed accuracy** (22.1% → 11.8%) because the
   "closest to center" tiebreak considered all surviving candidates
   regardless of score. Gating the tiebreak to near-best-score candidates
   (`score_gate_frac=0.9`) fixed this and outperformed the original
   fixed-10x version on every tested bucket, including extreme-scale pairs.
4. **DL confidence scores are unreliable** — confirmed across three
   datasets. The classical pipeline's confidence-gap flag is the more
   trustworthy failure signal and is proportional to its actual accuracy.
5. **Classical fails binary; DL fails gradually.** Classical accuracy is
   nearly flat from 1px to 5px (wrong-tile-entirely), while DL accuracy
   climbs steadily (3.4% → 56.2%), suggesting DL would benefit from further
   training or finer tolerance in ways a sub-pixel refinement step would not
   help the classical pipeline.

Full derivation and supporting evidence for each finding is in
[`docs/DriftSense_Failure_Case_Analysis.pdf`](docs/DriftSense_Failure_Case_Analysis.pdf).

## Known limitations

- The ~30-pair image sample shipped in this repo is representative, not
  exhaustive; the full 800-pair set is reproducible from the manifest and
  generator seed rather than shipped directly (repository size).
- Noise-level buckets used in the PR-curve analysis are terciles of the
  existing dataset's random noise distribution, not a fully controlled
  regenerate-at-fixed-noise sweep.
- RGB / optical microscope generalization (bonus criterion) is not yet
  covered by this submission.

## Citations

Noise modeling, DRAM/FinFET pitch dimensions, and wafer stage drift claims
used in dataset generation and the failure-case analysis are sourced and
verified in [`docs/DriftSense_Citations.pdf`](docs/DriftSense_Citations.pdf),
including the Poisson-Gaussian signal-dependent noise model from
Foi et al. (2008).
