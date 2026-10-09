"""Fit frozen training-selected heads and report every fixed development candidate."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from artifact_io import save_csv
from compare_model_runs import file_hash
from converged_probe import probabilities
from development_evaluation import evaluate_development
from foundation_common import ROOT, KEYS, save_json
from operating_policy import paired_decision_intervals
from run_clean_panderm_review import execution_lock, fit_cached
from run_precision_rebuild import OUTPUT as ORIGINAL, FEATURES, feature_arms, verify_completed_artifacts

AUDIT = ROOT / "artifacts/training_selection_objective_audit"
OUTPUT = ROOT / "artifacts/progress_review_20261007/corrected_clean_heads"
CODE = ("run_corrected_clean_heads.py", "high_recall_selection.py", "audit_training_head_selection.py",
        "run_precision_rebuild.py", "run_clean_panderm_review.py", "converged_probe.py",
        "development_evaluation.py", "operating_policy.py", "foundation_common.py", "artifact_io.py")


def now():
    return datetime.now(timezone.utc).isoformat()


def prepare(output):
    original_inputs = verify_completed_artifacts(ORIGINAL,
        ["experiment_plan.json", "development_scores.csv", "selection.json"])
    audit_inputs = verify_completed_artifacts(AUDIT, ["training_rank_quality.json", "training_oof_scores.csv"])
    audit = json.loads((AUDIT / "training_rank_quality.json").read_text())
    for name, expected in audit["code_sha256"].items():
        if file_hash(ROOT / name) != expected:
            raise ValueError("Training-selection audit code changed before the corrected fit.")
    original = json.loads((ORIGINAL / "experiment_plan.json").read_text())
    chosen = {}
    for arm in original["arms"]:
        selected = audit["results"][arm]["selected_candidate"]
        if selected is None:
            raise ValueError(f"No informative training-selected candidate for {arm}.")
        l2 = float(selected.removeprefix("l2_"))
        if l2 not in original["l2_grid"]:
            raise ValueError("Selection is outside the frozen original training grid.")
        chosen[arm] = l2
    inputs = [*original_inputs, *audit_inputs, FEATURES / "plan.json",
              FEATURES / "train_complete.json", FEATURES / "val_complete.json"]
    protocol = {"version": 1, "selected_l2": chosen, "starts": [0, 137], "threads": 2,
        "calibration_target": original["calibration_target"],
        "development_policy_seed": original["development_policy_seed"],
        "selection": "Already frozen from training-only mean within-source specificity at95% recall across inner scoring folds; no new grid or outer selection.",
        "selection_audit_sha256": file_hash(AUDIT / "training_rank_quality.json"),
        "inputs_sha256": {path.relative_to(ROOT).as_posix(): file_hash(path) for path in inputs},
        "code_sha256": {name: file_hash(ROOT / name) for name in CODE},
        "runtime": {name: importlib.metadata.version(name) for name in
            ("numpy", "pandas", "scipy", "scikit-learn", "h5py", "threadpoolctl")},
        "automatic_promotion": False, "test_or_hiba_access": False,
        "limitations": ["Retrospective methodological correction after inspecting earlier development outcomes, not prospective independent confirmation.",
            "Training-only inner head scores condition on an encoder that learned every training group.",
            "Encoder/checkpoint and earlier blend selection used the reused development cohort.",
            "Source-specific descriptive training thresholds only choose ranking quality; deployed95%source recall is not guaranteed.",
            "All candidates are reported on identical cached rows and grouped calibration folds. No outer-result candidate selection.",
            "Outputs are research feature heads, not an end-to-end JPEG deployment package."]}
    output.mkdir(parents=True, exist_ok=True)
    path = output / "experiment_plan.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("Corrected-study inputs/code changed; use a fresh directory.")
    save_json(path, protocol)
    for name in CODE:
        destination = output / "code_snapshot" / name
        destination.parent.mkdir(exist_ok=True)
        data = (ROOT / name).read_bytes()
        if destination.exists() and destination.read_bytes() != data:
            raise ValueError("Corrected-study code snapshot changed.")
        if not destination.exists():
            destination.write_bytes(data)
    return protocol


def execute(output, protocol):
    if (output / "complete.json").exists():
        marker = json.loads((output / "complete.json").read_text())
        for name, expected in marker["outputs_sha256"].items():
            if file_hash(output / name) != expected:
                raise ValueError("Completed corrected-study artifact changed.")
        return
    meta, arms = feature_arms("train")
    heads = {}
    for arm, l2 in protocol["selected_l2"].items():
        save_json(output / "status.json", {"status": "running", "phase": "training_selected_final_fit",
            "arm": arm, "l2": l2, "pid": os.getpid(), "updated_at": now()})
        features = arms[arm]
        fitted = {seed: fit_cached(output, f"{arm}_l2{l2:g}_start{seed}", features,
            meta.target.to_numpy(), l2, seed) for seed in protocol["starts"]}
        delta = float(np.max(np.abs(probabilities(fitted[0], features) - probabilities(fitted[137], features))))
        if delta > 1e-5:
            raise ValueError("Corrected head did not agree across optimization starts.")
        heads[arm] = fitted[0]
        save_json(output / f"{arm}_stability.json", {"maximum_training_probability_difference": delta,
            "interpretation": "Numerical optimizer stability, not independent training generalization variance."})
    del arms
    save_json(output / "training_complete.json", {"completed_at": now(),
        "selected_l2": protocol["selected_l2"], "outer_validation_used_for_selection": False})
    meta, arms = feature_arms("val")
    reference = pd.read_csv(ORIGINAL / "development_scores.csv")
    if not meta[KEYS].equals(reference[KEYS]):
        raise ValueError("Original and corrected development lesion identities differ.")
    for column in ("direct_probability", "diagnosis_probability"):
        np.testing.assert_allclose(meta[column], reference[column], atol=1e-7, rtol=0.)
    columns = ["direct_probability", "diagnosis_probability", "saved_checkpoint_blend", "panderm_only",
               "clean_b1", "clean_b1_plus_panderm"]
    for arm in protocol["selected_l2"]:
        name = "corrected_" + arm
        reference[name] = probabilities(heads[arm], arms[arm])
        columns.append(name)
    report, decisions = evaluate_development(reference, score_columns=columns, blend_columns=None,
        minimum_recall=protocol["calibration_target"], seed=protocol["development_policy_seed"])
    report["protocol_limitations"] = protocol["limitations"]
    report["paired_intervals"] = {}
    for arm in protocol["selected_l2"]:
        name = "corrected_" + arm
        candidate = decisions.loc[decisions.candidate.eq(name)].reset_index(drop=True)
        report["paired_intervals"][name] = {
            baseline: paired_decision_intervals(
                decisions.loc[decisions.candidate.eq(baseline)].reset_index(drop=True), candidate, iterations=1000)
            for baseline in (arm, "saved_checkpoint_blend")}
    for values in report["results"].values():
        values["all_source_recall_floor_met"] = all(
            metrics["recall"] >= protocol["calibration_target"] for metrics in values["by_source"].values())
    save_json(output / "development_report.json", report)
    save_csv(reference, output / "development_scores.csv")
    save_csv(decisions, output / "development_decisions.csv")
    save_json(output / "complete.json", {"completed_at": now(), "automatic_promotion": False,
        "outputs_sha256": {path.relative_to(output).as_posix(): file_hash(path) for path in output.rglob("*")
            if path.is_file() and path.suffix in {".json", ".npz", ".csv", ".py"}
            and path.name not in {"complete.json", "status.json", "progress.json"}}})
    save_json(output / "status.json", {"status": "complete", "updated_at": now(), "automatic_promotion": False})
    print(json.dumps({name: values["metrics"] for name, values in report["results"].items()}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    with execution_lock(args.output), threadpool_limits(limits=2):
        protocol = prepare(args.output)
        if not args.prepare_only:
            execute(args.output, protocol)


if __name__ == "__main__":
    main()
