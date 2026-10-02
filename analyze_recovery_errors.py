"""Describe validation errors and paired uncertainty without fitting a new model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from compare_model_runs import file_hash, load_validation, operating_metrics, write_report


def read_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"source": "string", "lesion_id": "string"})
    frame = frame.set_index(["source", "lesion_id"]).sort_index()
    if frame.index.has_duplicates or frame.index.to_frame().isna().any().any():
        raise ValueError("Prediction identities must be unique and nonmissing.")
    operating_metrics(frame.target.to_numpy(), frame.probability.to_numpy(), 0.95)
    return frame


def error_breakdown(frame: pd.DataFrame, threshold: float, column: str) -> list[dict]:
    rows = []
    for name, group in frame.groupby(column, dropna=False):
        truth = group.target.to_numpy()
        predicted = group.probability.to_numpy() >= threshold
        positives = truth == 1
        negatives = truth == 0
        tp = int(np.sum(positives & predicted))
        tn = int(np.sum(negatives & ~predicted))
        fp = int(np.sum(negatives & predicted))
        fn = int(np.sum(positives & ~predicted))
        rows.append({
            "group": str(name), "lesions": len(group), "positive_count": int(positives.sum()),
            "false_positives": fp, "false_negatives": fn,
            "sensitivity": tp / int(positives.sum()) if positives.any() else None,
            "specificity": tn / int(negatives.sum()) if negatives.any() else None,
        })
    return rows


def paired_bootstrap(truth: np.ndarray, baseline: np.ndarray, candidate: np.ndarray,
                     clusters: np.ndarray, baseline_threshold: float, candidate_threshold: float,
                     iterations: int = 1000, seed: int = 42) -> dict:
    if iterations < 100:
        raise ValueError("Use at least 100 bootstrap iterations.")
    if len({len(truth), len(baseline), len(candidate), len(clusters)}) != 1:
        raise ValueError("Bootstrap inputs must be aligned.")
    codes, names = pd.factorize(clusters)
    if (codes < 0).any():
        raise ValueError("Missing cluster identities.")
    members = [np.flatnonzero(codes == code) for code in range(len(names))]
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(iterations):
        indices = np.concatenate([members[i] for i in rng.integers(0, len(members), len(members))])
        y, old, new = truth[indices], baseline[indices], candidate[indices]
        if len(np.unique(y)) != 2:
            continue
        negative = y == 0
        deltas.append([
            average_precision_score(y, new) - average_precision_score(y, old),
            roc_auc_score(y, new) - roc_auc_score(y, old),
            np.mean(new[negative] < candidate_threshold) - np.mean(old[negative] < baseline_threshold),
        ])
    if len(deltas) < iterations * 0.8:
        raise ValueError("Too few bootstrap samples contain both classes.")
    bounds = np.quantile(np.asarray(deltas), [0.025, 0.975], axis=0)
    return {
        "cluster_count": len(members), "valid_iterations": len(deltas), "seed": seed,
        "delta_intervals_95": {name: {"lower": float(bounds[0, i]), "upper": float(bounds[1, i])}
                               for i, name in enumerate(("average_precision", "roc_auc", "specificity"))},
        "limitations": "Paired split-group bootstrap conditional on validation-selected models, blends and fixed thresholds. Not an independent significance test; does not account for model/threshold selection or training-seed variance. Missing patient identities cannot be recovered; lesion grouping is used where patient grouping is unavailable.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--baseline-run-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=1000)
    args = parser.parse_args()
    root = args.experiment_dir
    status = json.loads((root / "status.json").read_text(encoding="utf-8"))
    if status.get("status") != "complete":
        raise ValueError("Pilot must complete before error analysis.")
    frame = read_predictions(root / "baseline_validation_predictions.csv")
    finalized, _ = load_validation(args.baseline_run_dir)
    if not frame.index.equals(finalized.index) or not np.array_equal(frame.target, finalized.target):
        raise ValueError("Pilot validation differs from the finalized baseline.")
    split = pd.read_csv(args.baseline_run_dir / "val_split.csv", dtype="string", low_memory=False)
    groups = split.groupby(["source", "lesion_id"])
    if (groups["split_group_id"].nunique(dropna=False) != 1).any():
        raise ValueError("A lesion maps to multiple split groups.")
    metadata = groups[["split_group_id", "image_id", "image_path"]].first().sort_index()
    if not frame.index.equals(metadata.index):
        raise ValueError("Validation metadata does not match predictions.")
    frame = frame.join(metadata, validate="one_to_one")
    reference = operating_metrics(frame.target.to_numpy(), frame.probability.to_numpy(), 0.95)
    baseline_threshold = reference["threshold"]
    report = {
        "selection_split": "validation", "tta_transforms": 1, "baseline_metrics": reference,
        "by_source": error_breakdown(frame.reset_index(), baseline_threshold, "source"),
        "by_diagnosis": error_breakdown(frame.reset_index(), baseline_threshold, "diagnosis"),
        "comparisons": {},
        "prediction_files_sha256": {"baseline": file_hash(root / "baseline_validation_predictions.csv")},
    }
    decisions = {}
    for name in ("baseline", "control_finetune", "added_siim_finetune"):
        candidate = read_predictions(root / f"{name}_validation_predictions.csv")
        if not frame.index.equals(candidate.index) or not np.array_equal(frame.target, candidate.target):
            raise ValueError("Candidate validation identities or labels differ.")
        values = candidate.probability.to_numpy()
        metrics = operating_metrics(frame.target.to_numpy(), values, 0.95)
        decisions[name] = values >= metrics["threshold"]
        if name == "baseline":
            continue
        report["prediction_files_sha256"][name] = file_hash(root / f"{name}_validation_predictions.csv")
        report["comparisons"][name] = {
            "metrics": metrics,
            "by_source": error_breakdown(candidate.reset_index(), metrics["threshold"], "source"),
            "uncertainty": paired_bootstrap(
                frame.target.to_numpy(), frame.probability.to_numpy(), values,
                frame.split_group_id.to_numpy(), baseline_threshold, metrics["threshold"], args.iterations,
            ),
        }
    errors = frame.reset_index()
    predicted = decisions["baseline"]
    truth = frame.target.to_numpy()
    errors["error_type"] = np.where(truth == 1, "false_negative", "false_positive")
    errors["persists_in_both_pilots"] = (
        (decisions["control_finetune"] != truth) & (decisions["added_siim_finetune"] != truth)
    )
    errors = errors.loc[predicted != truth].copy()
    errors["review_priority"] = np.where(errors.error_type.eq("false_negative"), 1, 2)
    errors["error_confidence"] = np.where(errors.error_type.eq("false_negative"),
                                          1 - errors.probability, errors.probability)
    errors.sort_values(["review_priority", "persists_in_both_pilots", "error_confidence"],
                       ascending=[True, False, False]).to_csv(root / "validation_error_review.csv", index=False)
    report["persistent_false_negatives"] = int(
        (errors.error_type.eq("false_negative") & errors.persists_in_both_pilots).sum())
    report["review_warning"] = "Development error review only. Scores are uncalibrated single-view values, not deployed clinical probabilities. Do not move validation images into training, remove difficult examples, or alter labels without independent adjudication."
    write_report(root / "error_analysis.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
