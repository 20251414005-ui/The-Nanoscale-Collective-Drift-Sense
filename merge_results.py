"""
merge_results.py — builds a single submission-ready CSV combining:
  - manifest.csv                (ref/search paths, ground truth, per-pair generation metadata)
  - results/predictions.csv     (classical pipeline predictions)
  - results/predictions_dl.csv  (DL pipeline predictions)

joined on pair_id. Satisfies the spec's CSV/manifest deliverable: reference
path, search-image path, ground-truth x/y, predicted x/y, and per-pair
generation metadata, all in one file -- for BOTH pipelines side by side.

Usage:
    python merge_results.py \
        --manifest data/train_canonical/manifest.csv \
        --classical results/predictions.csv \
        --dl results/predictions_dl.csv \
        --out results/full_results.csv
"""

import argparse
import csv


def load_by_pair_id(path):
    with open(path, "r", newline="") as f:
        return {row["pair_id"]: row for row in csv.DictReader(f)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="data/train_canonical/manifest.csv")
    p.add_argument("--classical", default="results/predictions.csv")
    p.add_argument("--dl", default="results/predictions_dl.csv")
    p.add_argument("--out", default="results/full_results.csv")
    args = p.parse_args()

    manifest_rows = list(csv.DictReader(open(args.manifest, "r", newline="")))
    classical = load_by_pair_id(args.classical)
    dl = load_by_pair_id(args.dl)

    core_fields = {"pair_id", "style", "ref_path", "search_path", "gt_x", "gt_y"}
    manifest_fieldnames = list(manifest_rows[0].keys()) if manifest_rows else []
    metadata_fields = [c for c in manifest_fieldnames if c not in core_fields]

    out_fieldnames = (
        ["pair_id", "style", "ref_path", "search_path", "gt_x", "gt_y"]
        + metadata_fields
        + ["classical_pred_x", "classical_pred_y", "classical_error_px",
           "classical_time_sec", "classical_confident", "classical_n_candidates"]
        + ["dl_pred_x", "dl_pred_y", "dl_error_px", "dl_time_sec", "dl_confident"]
    )

    n_missing_classical = 0
    n_missing_dl = 0

    with open(args.out, "w", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=out_fieldnames)
        writer.writeheader()

        for row in manifest_rows:
            pid = row["pair_id"]
            out_row = {
                "pair_id": pid, "style": row["style"],
                "ref_path": row["ref_path"], "search_path": row["search_path"],
                "gt_x": row["gt_x"], "gt_y": row["gt_y"],
            }
            for field in metadata_fields:
                out_row[field] = row.get(field, "")

            c = classical.get(pid)
            if c:
                out_row["classical_pred_x"] = c["pred_x"]
                out_row["classical_pred_y"] = c["pred_y"]
                out_row["classical_error_px"] = c["error_px"]
                out_row["classical_time_sec"] = c["time_sec"]
                out_row["classical_confident"] = c["confident"]
                out_row["classical_n_candidates"] = c.get("n_candidates", "")
            else:
                n_missing_classical += 1
                for col in ["classical_pred_x", "classical_pred_y", "classical_error_px",
                            "classical_time_sec", "classical_confident", "classical_n_candidates"]:
                    out_row[col] = ""

            d = dl.get(pid)
            if d:
                out_row["dl_pred_x"] = d["pred_x"]
                out_row["dl_pred_y"] = d["pred_y"]
                out_row["dl_error_px"] = d["error_px"]
                out_row["dl_time_sec"] = d["time_sec"]
                out_row["dl_confident"] = d["confident"]
            else:
                n_missing_dl += 1
                for col in ["dl_pred_x", "dl_pred_y", "dl_error_px", "dl_time_sec", "dl_confident"]:
                    out_row[col] = ""

            writer.writerow(out_row)

    print(f"Wrote {len(manifest_rows)} rows to {args.out}")
    if n_missing_classical:
        print(f"WARNING: {n_missing_classical} pairs had no matching classical prediction "
              f"(evaluate.py may not have been run on the full manifest)")
    if n_missing_dl:
        print(f"WARNING: {n_missing_dl} pairs had no matching DL prediction "
              f"(evaluate_dl.py may not have been run on the full manifest)")


if __name__ == "__main__":
    main()
