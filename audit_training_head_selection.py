"""Audit a completed head-selection objective using existing training-only caches."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

from artifact_io import save_csv
from compare_model_runs import file_hash
from converged_probe import probabilities
from foundation_common import ROOT, KEYS, save_json
from high_recall_selection import evaluate_training_rank_quality
from run_precision_rebuild import OUTPUT as STUDY, FEATURES, feature_arms, verify_completed_artifacts
from run_clean_panderm_review import fit_cached

OUTPUT = ROOT / "artifacts/training_selection_objective_audit"


def audit(output=OUTPUT):
    if output.exists():
        raise ValueError("Use a fresh audit directory; existing evidence is not overwritten.")
    original = json.loads((STUDY / "experiment_plan.json").read_text())
    selections = json.loads((STUDY / "selection.json").read_text())
    names = ["experiment_plan.json", "selection.json", "training_folds.csv"]
    for arm in original["arms"]:
        for l2 in original["l2_grid"]:
            for fold in range(original["folds"]):
                names.extend(f"{arm}_l2{l2:g}_fold{fold}.{extension}" for extension in ("json", "npz"))
    verified = verify_completed_artifacts(STUDY, names)
    meta, arms = feature_arms("train")
    saved_folds = pd.read_csv(STUDY / "training_folds.csv")
    if not saved_folds[KEYS].equals(meta[KEYS]):
        raise ValueError("Training image groups/labels changed since the original fit.")
    folds, truth = saved_folds.inner_fold.to_numpy(), meta.target.to_numpy()
    result, tables = {}, []
    with threadpool_limits(limits=2):
        for arm, features in arms.items():
            frame = meta[KEYS].assign(inner_fold=folds).copy()
            candidates = []
            for l2 in original["l2_grid"]:
                name = f"l2_{l2:g}"
                candidates.append(name)
                scores = np.full(len(meta), np.nan)
                for fold in range(original["folds"]):
                    scored, calibrated = folds == fold, folds == (fold + 1) % original["folds"]
                    fitted = ~(scored | calibrated)
                    # Every named fit was verified above; fit_cached only loads it
                    # after verifying exact fitted feature and label fingerprints.
                    head = fit_cached(STUDY, f"{arm}_l2{l2:g}_fold{fold}",
                                      features[fitted], truth[fitted], l2, 0)
                    scores[scored] = probabilities(head, features[scored])
                previous = next(value for value in selections[arm]["candidates"] if value["l2"] == l2)
                observed = float(average_precision_score(truth, scores))
                if not np.isclose(observed, previous["pr_auc"], rtol=0., atol=1e-9):
                    raise ValueError("Training scoring does not reproduce the archived candidate AP.")
                frame[name] = scores
            report = evaluate_training_rank_quality(frame, candidates,
                minimum_recall=original["calibration_target"])
            report["original_selected_l2"] = selections[arm]["selected"]["l2"]
            report["original_calibrated_policy_candidates"] = selections[arm]["candidates"]
            result[arm] = report
            tables.append(frame.assign(arm=arm))
    output.mkdir(parents=True)
    save_csv(pd.concat(tables, ignore_index=True), output / "training_oof_scores.csv")
    report = {"status": "completed", "scope": "Retrospective training-only objective audit; no new fits or outer predictions.",
        "test_or_hiba_access": False, "automatic_promotion": False,
        "inputs_sha256": {path.relative_to(ROOT).as_posix(): file_hash(path) for path in
            [*verified, FEATURES / "plan.json", FEATURES / "train_complete.json"]},
        "code_sha256": {name: file_hash(ROOT / name) for name in
            ("audit_training_head_selection.py", "high_recall_selection.py", "run_precision_rebuild.py", "converged_probe.py")},
        "results": result,
        "limitations": ["Alternative objective was designed after inspecting earlier development results; this audit does not establish prospective generalization.",
            "Only training out-of-fold scores select the alternative L2. Existing encoder saw all training groups.",
            "Descriptive within-source thresholds are not a deployment policy or calibrated-recall guarantee.",
            "Original checkpoints and frozen study results remain unchanged."]}
    save_json(output / "training_rank_quality.json", report)
    save_json(output / "complete.json", {"outputs_sha256": {
        name: file_hash(output / name) for name in ("training_oof_scores.csv", "training_rank_quality.json")}})
    print(json.dumps({arm: {"original_l2": values["original_selected_l2"],
        "training_only_alternative": values["selected_candidate"],
        "quality": {name: item["macro_specificity_at_target"] for name, item in values["candidates"].items()}}
        for arm, values in result.items()}, indent=2))


if __name__ == "__main__":
    audit()
