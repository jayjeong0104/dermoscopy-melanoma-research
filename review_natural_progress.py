"""Matched development comparison of two completed clean checkpoints and one fixed ensemble."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import shutil

import numpy as np

from artifact_io import save_csv
from compare_model_runs import file_hash
from checkpoint_lesion_scores import load_checkpoint_scores
from development_evaluation import evaluate_development
from extract_clean_reference import OUTPUT as REFERENCE_FEATURES, RUN as REFERENCE_RUN
from foundation_common import ROOT, KEYS, save_json
from operating_policy import paired_decision_intervals
from run_clean_panderm_review import execution_lock
from run_precision_rebuild import saved_checkpoint_blend, verify_completed_artifacts

OUTPUT = ROOT / "artifacts/progress_review_20261007/natural_comparison_v2"
FEATURES = OUTPUT.parent / "natural_features"
CORRECTED = OUTPUT.parent / "corrected_clean_heads"
RUN = ROOT / "artifacts/natural_lesion_20261006/training/run_20261006_174318"
SOURCES = ("review_natural_progress.py", "checkpoint_lesion_scores.py", "development_evaluation.py", "operating_policy.py",
           "extract_clean_reference.py", "run_precision_rebuild.py", "foundation_common.py", "artifact_io.py",
           "compare_model_runs.py", "run_clean_panderm_review.py")


def prepare(output):
    output.mkdir(parents=True, exist_ok=True)
    corrected_inputs = verify_completed_artifacts(CORRECTED, ["development_scores.csv", "experiment_plan.json"])
    protocol = {"version": 2, "objective": "95% melanoma sensitivity with fewer false positives on clinical dermoscopy",
        "fixed_candidates": ["reference_saved", "natural_saved", "equal_checkpoint_ensemble", "natural_diagnosis",
            "corrected_clean_b1", "corrected_clean_b1_plus_panderm"],
        "corrected_head_evidence_sha256": {p.relative_to(CORRECTED).as_posix(): file_hash(p) for p in corrected_inputs},
        "aggregation": "Checkpoint direct score uses sigmoid(mean(clipped image logits)); diagnosis uses mean probabilities. Linear feature heads use mean embeddings.",
        "preflight_correction": "v1 was prepared but never executed; it would have used arithmetic-mean direct scores. Kept separately for audit.",
        "ensemble": "Unoptimized 50:50 arithmetic mean of the two saved checkpoint blends",
        "reference": saved_checkpoint_blend(REFERENCE_RUN), "natural": saved_checkpoint_blend(RUN),
        "calibration_target": .95, "development_policy_seed": 20261006, "calibration_folds": 5,
        "evaluation": "Each candidate independently uses identical group-disjoint calibration/scoring folds",
        "test_or_hiba_access": False, "automatic_promotion": False,
        "limitations": ["Both checkpoints were selected on reused validation; cross-fitting thresholds does not create independent evidence.",
            "No guaranteed 95% sensitivity on scored groups or unseen data.",
            "A fixed ensemble tests complementary errors; no grid search or outer-validation weight selection.",
            "Only one training seed; conditional paired intervals do not cover training variability."],
        "code_sha256": {name: file_hash(ROOT / name) for name in SOURCES}}
    path = output / "experiment_plan.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("Protocol changed; use a fresh output directory.")
    save_json(path, protocol)
    snapshot = output / "code_snapshot"
    snapshot.mkdir(exist_ok=True)
    for name, digest in protocol["code_sha256"].items():
        target = snapshot / name
        if target.exists() and file_hash(target) != digest:
            raise ValueError("Comparison code snapshot changed.")
        if not target.exists():
            shutil.copyfile(ROOT / name, target)
    return protocol


def align_reference(reference, natural):
    for frame in (reference, natural):
        if frame[KEYS].isna().any().any() or frame.duplicated(["source", "lesion_id"]).any():
            raise ValueError("Invalid or duplicated lesion identities.")
    if len(reference) != len(natural):
        raise ValueError("Checkpoint cohorts differ in size.")
    import pandas as pd
    indices = pd.MultiIndex.from_frame(reference[KEYS]).get_indexer(pd.MultiIndex.from_frame(natural[KEYS]))
    if (indices < 0).any():
        raise ValueError("Checkpoint labels or groups differ.")
    aligned = reference.iloc[indices].reset_index(drop=True)
    for name in ("diagnosis", "image_count"):
        if not np.array_equal(aligned[name], natural[name]):
            raise ValueError(f"Checkpoint cohorts differ in {name}.")
    return aligned


def execute(output, protocol):
    if (output / "complete.json").exists():
        for name, digest in json.loads((output / "complete.json").read_text())["outputs_sha256"].items():
            if file_hash(output / name) != digest:
                raise ValueError("Completed comparison changed.")
        return
    reference = load_checkpoint_scores(REFERENCE_FEATURES, REFERENCE_RUN, "val")
    natural = load_checkpoint_scores(FEATURES, RUN, "val")
    reference = align_reference(reference, natural)
    binding = {"features": {label: {name: file_hash(directory / name) for name in
        ("plan.json", "val_complete.json")} for label, directory in
        (("reference", REFERENCE_FEATURES), ("natural", FEATURES))},
        "runtime": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scikit-learn")}}
    path = output / "input_binding.json"
    if path.exists() and json.loads(path.read_text()) != binding:
        raise ValueError("Feature inputs or runtime changed.")
    save_json(path, binding)
    frame = natural.copy()
    for name, values in (("reference", reference), ("natural", natural)):
        weight = protocol[name]["direct_weight"]
        frame[f"{name}_saved"] = weight * values.direct_probability + (1 - weight) * values.diagnosis_probability
    frame["equal_checkpoint_ensemble"] = .5 * (frame.reference_saved + frame.natural_saved)
    frame["natural_diagnosis"] = frame.diagnosis_probability
    import pandas as pd
    corrected = align_reference(pd.read_csv(CORRECTED / "development_scores.csv"), frame)
    for name in ("corrected_clean_b1", "corrected_clean_b1_plus_panderm"):
        frame[name] = corrected[name]
    report, decisions = evaluate_development(frame, score_columns=protocol["fixed_candidates"], blend_columns=None,
        minimum_recall=protocol["calibration_target"], seed=protocol["development_policy_seed"],
        n_splits=protocol["calibration_folds"])
    baseline = decisions.loc[decisions.candidate.eq("reference_saved")].reset_index(drop=True)
    report["paired_intervals_against_reference"] = {name: paired_decision_intervals(baseline,
        decisions.loc[decisions.candidate.eq(name)].reset_index(drop=True), iterations=1000)
        for name in ("natural_saved", "equal_checkpoint_ensemble")}
    report["protocol_limitations"] = protocol["limitations"]
    save_json(output / "development_report.json", report)
    save_csv(decisions, output / "development_decisions.csv")
    save_csv(frame, output / "development_scores.csv")
    for name, result in report["results"].items():
        print(name, json.dumps(result["metrics"]), flush=True)
    save_json(output / "complete.json", {"completed_at": datetime.now(timezone.utc).isoformat(),
        "automatic_promotion": False, "outputs_sha256": {p.relative_to(output).as_posix(): file_hash(p)
            for p in output.rglob("*") if p.is_file() and p.name != "complete.json" and p.suffix in (".json", ".csv", ".py")}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.prepare_only:
        prepare(args.output)
    else:
        with execution_lock(args.output):
            execute(args.output, prepare(args.output))


if __name__ == "__main__":
    main()
