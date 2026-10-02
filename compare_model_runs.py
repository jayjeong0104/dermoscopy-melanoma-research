"""Compare finalized runs using validation data only; never replace model files."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def file_hash(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load_validation(run_dir: Path) -> tuple[pd.DataFrame, dict]:
    status = json.loads((run_dir / "finalization_status.json").read_text(encoding="utf-8"))
    if status.get("status") != "complete":
        raise ValueError(f"Run is not finalized: {run_dir}")
    model_hash = file_hash(run_dir / "best_model.keras")
    if model_hash != status.get("artifact_sha256", {}).get("best_model.keras"):
        raise ValueError(f"Model differs from finalized model: {run_dir}")
    if file_hash(run_dir / "val_split.csv") != status.get("artifact_sha256", {}).get("val_split.csv"):
        raise ValueError(f"Validation split differs from finalized split: {run_dir}")
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    calibration = json.loads((run_dir / "calibration.json").read_text(encoding="utf-8"))
    path = run_dir / "val_lesion_predictions.csv"
    frame = pd.read_csv(path, dtype={"source": "string", "lesion_id": "string"})
    required = {"source", "lesion_id", "target", "probability"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing validation columns: {sorted(required - set(frame.columns))}")
    if frame[list(required)].isna().any().any():
        raise ValueError("Validation data contains missing identities, labels or scores.")
    frame = frame.set_index(["source", "lesion_id"]).sort_index()
    if frame.index.has_duplicates:
        raise ValueError("Duplicate validation lesion identities.")
    # Check against the split file as well as the other run, not just row counts.
    split = pd.read_csv(run_dir / "val_split.csv", dtype={"source": "string", "lesion_id": "string"}, low_memory=False)
    grouped = split.groupby(["source", "lesion_id"])["target"]
    if (grouped.nunique() != 1).any():
        raise ValueError("Conflicting validation labels within a lesion.")
    expected = grouped.first().sort_index()
    if not frame.index.equals(expected.index) or not np.array_equal(frame.target, expected):
        raise ValueError("Validation predictions do not match the saved split.")
    info = {
        "run_dir": str(run_dir.resolve()),
        "model_sha256": model_hash,
        "validation_predictions_sha256": file_hash(path),
        "binary_target": config.get("binary_target", "high_risk"),
        "tta_transforms": calibration.get("tta_transforms", config.get("tta_transforms", 1)),
        "tta_scheme": calibration.get("tta_scheme", config.get("tta_scheme", "legacy_v1")),
        "image_size": config["image_size"],
        "binary_inference_metadata": config.get("binary_inference_metadata", "unspecified"),
    }
    return frame, info


def operating_metrics(truth: np.ndarray, probabilities: np.ndarray, minimum_recall: float) -> dict:
    truth = np.asarray(truth)
    probabilities = np.asarray(probabilities, dtype="float64")
    if truth.shape != probabilities.shape or truth.ndim != 1:
        raise ValueError("Labels and probabilities must be aligned one-dimensional arrays.")
    if set(np.unique(truth)) != {0, 1}:
        raise ValueError("Both binary classes are required.")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Probabilities must be finite and between zero and one.")
    if not 0 < minimum_recall <= 1:
        raise ValueError("Minimum recall must be in (0, 1].")
    positives = np.sort(probabilities[truth == 1])[::-1]
    required_positives = int(np.ceil(minimum_recall * len(positives)))
    threshold = float(positives[required_positives - 1])
    predicted = probabilities >= threshold
    tp = int(np.sum(predicted & (truth == 1)))
    fp = int(np.sum(predicted & (truth == 0)))
    fn = int(np.sum(~predicted & (truth == 1)))
    tn = int(np.sum(~predicted & (truth == 0)))
    return {
        "threshold": threshold, "recall": tp / (tp + fn),
        "specificity": tn / (tn + fp), "precision": tp / (tp + fp),
        "accuracy": (tp + tn) / len(truth),
        "true_positives": tp, "false_positives": fp,
        "false_negatives": fn, "true_negatives": tn,
        "pr_auc": float(average_precision_score(truth, probabilities)),
        "roc_auc": float(roc_auc_score(truth, probabilities)),
    }


def compare_frames(baseline: pd.DataFrame, candidate: pd.DataFrame,
                   minimum_recall: float = 0.95, minimum_gain: float = 0.005) -> dict:
    if not np.isfinite(minimum_gain) or minimum_gain <= 0:
        raise ValueError("Minimum gain must be positive and finite.")
    if not baseline.index.equals(candidate.index):
        raise ValueError("Runs must use identical validation lesion identities and ordering.")
    if not np.array_equal(baseline.target, candidate.target):
        raise ValueError("Runs disagree on validation labels.")
    truth = baseline.target.to_numpy()
    old = operating_metrics(truth, baseline.probability.to_numpy(), minimum_recall)
    new = operating_metrics(truth, candidate.probability.to_numpy(), minimum_recall)
    criteria = {
        "pr_auc_improves": new["pr_auc"] >= old["pr_auc"] + minimum_gain,
        "specificity_improves_at_target_recall": new["specificity"] >= old["specificity"] + minimum_gain,
        "roc_auc_does_not_regress": new["roc_auc"] >= old["roc_auc"],
    }
    return {
        "selection_split": "validation", "evaluation_unit": "lesion",
        "minimum_recall": minimum_recall, "minimum_absolute_gain": minimum_gain,
        "sample_count": len(truth), "baseline_metrics": old, "candidate_metrics": new,
        "criteria": criteria, "candidate_passes": all(criteria.values()),
        "decision": "candidate_eligible_for_confirmation" if all(criteria.values()) else "retain_baseline",
        "limitations": "Development gate, not proof of generalization or statistical significance. Thresholds are fitted on validation only; no test or external results are read. Independent confirmation is still required.",
    }


def compare_runs(baseline_dir: Path, candidate_dir: Path,
                 minimum_recall: float = 0.95, minimum_gain: float = 0.005) -> dict:
    baseline, baseline_info = load_validation(baseline_dir)
    candidate, candidate_info = load_validation(candidate_dir)
    for field in ("binary_target", "tta_transforms", "tta_scheme", "image_size", "binary_inference_metadata"):
        if baseline_info[field] != candidate_info[field]:
            raise ValueError(f"Incompatible evaluation setting: {field}")
    report = compare_frames(baseline, candidate, minimum_recall, minimum_gain)
    report.update(baseline=baseline_info, candidate=candidate_info)
    selected = candidate_info if report["candidate_passes"] else baseline_info
    report["recommended_research_reference"] = selected
    return report


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run-dir", type=Path, required=True)
    parser.add_argument("--candidate-run-dir", type=Path, required=True)
    parser.add_argument("--minimum-recall", type=float, default=0.95)
    parser.add_argument("--minimum-gain", type=float, default=0.005)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare_runs(args.baseline_run_dir, args.candidate_run_dir, args.minimum_recall, args.minimum_gain)
    write_report(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
