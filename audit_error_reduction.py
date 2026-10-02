"""Audit fixed development decisions without tuning thresholds or changing models."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from artifact_io import save_csv
from compare_model_runs import file_hash
from foundation_common import BASELINE, KEYS, REFERENCE, ROOT, lesion_metadata, save_json, split_frame
from operating_policy import crossfit_policy, validate_folds

TRANSITIONS = (
    "false_alarm_fixed", "false_alarm_introduced", "melanoma_recovered",
    "melanoma_newly_missed", "persistent_false_alarm", "persistent_missed_melanoma",
    "correct_in_both",
)
LIMITATION = (
    "Audit of fixed, reused-development predictions, not a new independent test or training run. "
    "No threshold selection, model promotion, source-specific correction, or relabeling. "
    "Review rows remain validation-only and must not be used for training. "
    "Non-melanoma includes other cancers and is not equivalent to healthy skin."
)


def validate_predictions(frame):
    required = KEYS + ["policy_fold", "probability", "threshold", "prediction"]
    if not set(required).issubset(frame.columns) or frame[required].isna().any().any():
        raise ValueError("Predictions need complete identities, folds, scores, thresholds and decisions.")
    validate_folds(frame, frame.policy_fold.to_numpy())
    if (not frame.prediction.isin([0, 1]).all()
            or not np.isfinite(frame[["probability", "threshold"]].to_numpy()).all()
            or not frame.probability.between(0, 1).all()
            or not frame.prediction.eq(frame.probability.ge(frame.threshold).astype(int)).all()):
        raise ValueError("Invalid or inconsistent saved scores and decisions.")


def metrics(frame, column):
    y, calls = frame.target.to_numpy(), frame[column].to_numpy(dtype=bool)
    tp, fp = int(np.sum(calls & (y == 1))), int(np.sum(calls & (y == 0)))
    fn, tn = int(np.sum(~calls & (y == 1))), int(np.sum(~calls & (y == 0)))
    return {"lesions": len(y), "melanomas": tp + fn, "non_melanomas": fp + tn,
        "true_positives": tp, "false_positives": fp, "false_negatives": fn, "true_negatives": tn,
        "errors": fp + fn, "accuracy": (tp + tn) / len(y),
        "precision": tp / (tp + fp) if tp + fp else None,
        "sensitivity": tp / (tp + fn) if tp + fn else None,
        "specificity": tn / (tn + fp) if tn + fp else None}


def paired_rows(reference, candidate, metadata):
    for frame in (reference, candidate):
        validate_predictions(frame)
    reference = reference.sort_values(KEYS[:2]).reset_index(drop=True)
    candidate = candidate.sort_values(KEYS[:2]).reset_index(drop=True)
    if not reference[KEYS + ["policy_fold"]].equals(candidate[KEYS + ["policy_fold"]]):
        raise ValueError("Paired lesion identities, targets or calibration groups differ.")
    frame = reference.rename(columns={key: "retained_" + key for key in
        ("probability", "threshold", "prediction")})
    for key in ("probability", "threshold", "prediction"):
        frame["candidate_" + key] = candidate[key]
    frame = frame.merge(metadata[KEYS + ["diagnosis", "image_id", "image_path"]],
        on=KEYS, how="outer", validate="one_to_one", indicator=True)
    if not frame._merge.eq("both").all() or frame.diagnosis.isna().any():
        raise ValueError("Review metadata does not match scored lesions.")
    if not frame.target.eq(frame.diagnosis.eq("MEL").astype(int)).all():
        raise ValueError("Diagnosis and melanoma target disagree.")
    positive = frame.target.eq(1)
    old, new = frame.retained_prediction.eq(1), frame.candidate_prediction.eq(1)
    frame["transition"] = np.select([
        ~positive & old & ~new, ~positive & ~old & new,
        positive & ~old & new, positive & old & ~new,
        ~positive & old & new, positive & ~old & ~new,
    ], TRANSITIONS[:6], default=TRANSITIONS[6])
    return frame.drop(columns="_merge")


def summarize(frame):
    before, after = metrics(frame, "retained_prediction"), metrics(frame, "candidate_prediction")
    counts = {name: int(frame.transition.eq(name).sum()) for name in TRANSITIONS}
    return {"retained": before, "candidate": after, "transitions": counts,
        "net_fewer_errors": before["errors"] - after["errors"],
        "net_fewer_false_alarms": counts["false_alarm_fixed"] - counts["false_alarm_introduced"],
        "net_fewer_missed_melanomas": counts["melanoma_recovered"] - counts["melanoma_newly_missed"]}


def compare(reference, candidate, metadata):
    frame = paired_rows(reference, candidate, metadata)
    result = summarize(frame)
    before, after = result["retained"], result["candidate"]
    result["diagnostic_checks_not_a_replacement_promotion_gate"] = {
        "fewer_total_errors": after["errors"] < before["errors"],
        "higher_precision": (after["precision"] is not None and before["precision"] is not None
                             and after["precision"] > before["precision"]),
        "no_net_increase_in_missed_melanomas": after["false_negatives"] <= before["false_negatives"],
        "observed_sensitivity_at_least_95_percent": after["sensitivity"] >= .95,
    }
    result["subgroups"] = {}
    for dimension in ("source", "diagnosis", "policy_fold"):
        result["subgroups"][dimension] = {str(name): summarize(group)
            for name, group in frame.groupby(dimension, sort=True)}
    sources = result["subgroups"]["source"]
    result["source_regression_warnings"] = [{"source": name,
        "additional_false_alarms": max(0, -group["net_fewer_false_alarms"]),
        "additional_missed_melanomas": max(0, -group["net_fewer_missed_melanomas"])}
        for name, group in sources.items()
        if group["net_fewer_false_alarms"] < 0 or group["net_fewer_missed_melanomas"] < 0]
    result["diagnostic_checks_not_a_replacement_promotion_gate"].update({
        "no_source_has_more_missed_melanomas": all(
            group["net_fewer_missed_melanomas"] >= 0 for group in sources.values()),
        "no_source_has_more_false_alarms": all(
            group["net_fewer_false_alarms"] >= 0 for group in sources.values()),
    })
    result["subgroup_caution"] = "Descriptive, sometimes small groups; not proof of a source effect or subgroup equivalence."
    return result, frame


def read_verified(path, record):
    if file_hash(path) != record["predictions_sha256"]:
        raise ValueError(f"Archived prediction hash mismatch: {path}")
    frame = pd.read_csv(path)
    validate_predictions(frame)
    _, rebuilt = crossfit_policy(frame, frame.probability.to_numpy(),
        minimum_recall=record["metrics"]["minimum_calibration_recall"],
        folds=frame.policy_fold.to_numpy())
    if (not np.array_equal(frame.prediction, rebuilt.prediction)
            or not np.allclose(frame.threshold, rebuilt.threshold, rtol=1e-12, atol=1e-15)):
        raise ValueError("Archived operating decisions do not reproduce the declared policy.")
    return frame


def audit(directory, candidate_name, output):
    directory, output = directory.resolve(), output.resolve()
    if directory == output or directory in output.parents or output in directory.parents:
        raise ValueError("Keep audit outputs separate from experiment artifacts.")
    report_path = directory / "validation_report.json"
    complete = json.loads((directory / "complete.json").read_text(encoding="utf-8"))
    if file_hash(report_path) != complete["report_sha256"]:
        raise ValueError("Completed experiment report hash mismatch.")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if candidate_name not in report["results"] or Path(candidate_name).name != candidate_name:
        raise ValueError("Select a named result from the completed experiment.")
    candidate_path = directory / f"{candidate_name}_validation_predictions.csv"
    candidate = read_verified(candidate_path, report["results"][candidate_name])
    old_report_path = REFERENCE.parent / "seed42_validation_report.json"
    old_report = json.loads(old_report_path.read_text(encoding="utf-8"))
    reference = read_verified(REFERENCE, old_report["results"]["baseline"])
    val_path = BASELINE / "val_split.csv"
    source_plan_path = ROOT / "artifacts/foundation_recovery_20260909/experiment_plan.json"
    source_plan = json.loads(source_plan_path.read_text(encoding="utf-8"))
    expected = [value for key, value in source_plan["input_sha256"].items()
                if key.replace("\\", "/").endswith("/val_split.csv")]
    if expected != [file_hash(val_path)]:
        raise ValueError("Validation metadata differs from the experiment's frozen input.")
    result, rows = compare(reference, candidate, lesion_metadata(split_frame("val")))
    result.update({"candidate": {**result["candidate"], "name": candidate_name},
        "original_experiment_decision": report["decision"],
        "retained_calibration_target": old_report["results"]["baseline"]["metrics"]["minimum_calibration_recall"],
        "candidate_calibration_target": report["results"][candidate_name]["metrics"]["minimum_calibration_recall"],
        "scope": LIMITATION, "test_inference": False, "automatic_promotion": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_sha256": {str(path): file_hash(path) for path in
            (report_path, candidate_path, REFERENCE, old_report_path, val_path, source_plan_path)},
        "code_sha256": {name: file_hash(ROOT / name) for name in
            ("audit_error_reduction.py", "operating_policy.py", "foundation_common.py", "artifact_io.py")}})
    output.mkdir(parents=True, exist_ok=True)
    review = rows.loc[rows.transition.ne("correct_in_both")].copy()
    review["review_status"] = "not_reviewed"
    review["allowed_use"] = "validation_audit_only_not_training"
    save_csv(review.sort_values(["transition", "source", "lesion_id"]), output / "error_review.csv")
    result["review_sha256"] = file_hash(output / "error_review.csv")
    save_json(output / "error_reduction.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, default=ROOT / "artifacts/convergence_recovery_20261001")
    parser.add_argument("--candidate", default="fusion_start0")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/error_reduction_20261001")
    args = parser.parse_args()
    result = audit(args.experiment_dir, args.candidate, args.output_dir)
    print(json.dumps({key: result[key] for key in
        ("retained", "candidate", "transitions", "source_regression_warnings",
         "diagnostic_checks_not_a_replacement_promotion_gate")}, indent=2))
