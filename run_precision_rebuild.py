"""Bounded, training-selected clean feature-head study; never evaluates test data."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from artifact_io import save_csv
from compare_model_runs import file_hash, operating_metrics
from converged_probe import probabilities
from development_evaluation import evaluate_development
from extract_clean_reference import OUTPUT as FEATURES, RUN, load_extracted
from foundation_common import ROOT, KEYS, save_json
from operating_policy import decision_metrics, paired_decision_intervals
from run_clean_panderm_review import only_panderm, grouped_folds, fit_cached, selection_key, execution_lock

OUTPUT = ROOT / "artifacts/precision_rebuild_20261006/head_study"


def now():
    return datetime.now(timezone.utc).isoformat()


def verify_completed_artifacts(directory, names):
    """Verify original completion evidence, accepting recorded Windows separators."""
    marker = directory / "complete.json"
    recorded = json.loads(marker.read_text())["outputs_sha256"]
    normalized = {name.replace("\\", "/"): digest for name, digest in recorded.items()}
    if len(normalized) != len(recorded):
        raise ValueError("Ambiguous paths in completion evidence.")
    paths = [directory / name for name in names]
    for name, path in zip(names, paths):
        if normalized.get(name.replace("\\", "/")) != file_hash(path):
            raise ValueError(f"Artifact differs from original completion evidence: {name}")
    return [marker, *paths]


def saved_checkpoint_blend(run=RUN):
    directory = run.parent.parent
    names = [(run / name).relative_to(directory).as_posix() for name in
             ("best_finetune_model.keras", "run_config.json", "training_finetune.csv")]
    paths = verify_completed_artifacts(directory, names)
    config = json.loads((run / "run_config.json").read_text())
    if config.get("checkpoint_metric") != "lesion_pr_auc":
        raise ValueError("Saved blend recovery supports the recorded lesion-AP checkpoint rule only.")
    history = pd.read_csv(run / "training_finetune.csv")
    columns = ["epoch", "val_lesion_pr_auc", "val_lesion_direct_weight"]
    if history.empty or history.epoch.duplicated().any() or not np.isfinite(history[columns].to_numpy()).all():
        raise ValueError("Invalid fine-tuning checkpoint history.")
    # ModelCheckpoint(mode='max') retains the first maximum, including exact ties.
    selected = history.loc[history.val_lesion_pr_auc.idxmax()]
    weight = float(selected.val_lesion_direct_weight)
    if not 0 <= weight <= 1:
        raise ValueError("Invalid saved checkpoint blend weight.")
    return {"direct_weight": weight, "epoch_zero_based": int(selected.epoch),
        "selection_metric": "val_lesion_pr_auc", "selection_value": float(selected.val_lesion_pr_auc),
        "checkpoint_sha256": file_hash(run / "best_finetune_model.keras"),
        "evidence_sha256": {path.relative_to(directory).as_posix(): file_hash(path) for path in paths}}


def plan(output):
    output.mkdir(parents=True, exist_ok=True)
    value = {"version": 4, "arms": ["clean_b1", "clean_b1_plus_panderm"],
        "l2_grid": [.01, .1, 1.], "final_starts": [0, 137], "folds": 5,
        "calibration_target": .95, "development_policy_seed": 20261006,
        "fit_unit": "one equally weighted lesion", "features": "mean of image embeddings per lesion",
        "selection": "Training only: each rotation uses 60% fitting,20% calibration,20% scoring; prefer every-source recall>=.95 then macro specificity, else worst-source recall.",
        "purpose": "Test whether natural lesion-weighted converged binary heads and complementary clean features reduce false alarms.",
        "comparison": "All predeclared arms and existing direct/diagnosis scores share grouped development calibration folds. No validation-driven arm promotion.",
        "test_or_hiba_access": False, "automatic_promotion": False,
        "saved_checkpoint_blend": saved_checkpoint_blend(),
        "limitations": ["Clean B1 encoder already trained on all training groups and was checkpoint-selected on validation; inner head scores are conditional and optimistic.",
            "Validation is repeatedly inspected development data, not independent confirmation.",
            "PanDerm private pretraining overlap is not exhaustively auditable.",
            "This frozen-feature experiment does not measure the effect of changing the image-training sampler."],
        "code_sha256": {name: file_hash(ROOT / name) for name in
            ("run_precision_rebuild.py", "converged_probe.py", "development_evaluation.py", "run_clean_panderm_review.py")}}
    path = output / "experiment_plan.json"
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError("Study code/protocol changed; use a new directory.")
    save_json(path, value)
    return value


def aligned_panderm(split, clean_meta):
    _, meta, features = only_panderm(split)
    for frame in (meta, clean_meta):
        if frame[KEYS].isna().any().any() or frame.duplicated(["source", "lesion_id"]).any():
            raise ValueError("PanDerm and clean feature lesions must have unique complete identities.")
    if features.shape != (len(meta), 768) or not np.isfinite(features).all():
        raise ValueError("Invalid aligned PanDerm feature dimensions or values.")
    index = pd.MultiIndex.from_frame(meta[KEYS])
    wanted = pd.MultiIndex.from_frame(clean_meta[KEYS])
    locations = index.get_indexer(wanted)
    if not index.is_unique or len(meta) != len(clean_meta) or (locations < 0).any():
        raise ValueError("PanDerm and clean feature lesion identities or labels differ.")
    if "image_count" in meta and "image_count" in clean_meta:
        if not np.array_equal(meta.image_count.to_numpy()[locations], clean_meta.image_count.to_numpy()):
            raise ValueError("PanDerm and clean feature lesions contain different image counts.")
    return features[locations]


def feature_arms(split):
    meta, clean = load_extracted(FEATURES, split)
    external = aligned_panderm(split, meta)
    return meta, {"clean_b1": clean, "clean_b1_plus_panderm": np.concatenate([clean, external], axis=1)}


def execute(output, protocol):
    from run_clean_panderm_review import OUTPUT as PANDERM_OUTPUT
    baseline_head = PANDERM_OUTPUT / "final_start0.npz"
    baseline_inputs = verify_completed_artifacts(PANDERM_OUTPUT,
        ["final_start0.npz", "final_start0.json", "experiment_plan.json"])
    baseline_marker = json.loads((PANDERM_OUTPUT / "final_start0.json").read_text())
    if file_hash(baseline_head) != baseline_marker["parameters_sha256"]:
        raise ValueError("Existing pure-PanDerm head differs from its completed fit.")
    if saved_checkpoint_blend() != protocol["saved_checkpoint_blend"]:
        raise ValueError("Saved checkpoint blend evidence changed after preparation.")
    binding = {"inputs_sha256": {path.relative_to(ROOT).as_posix(): file_hash(path) for path in
        [FEATURES / "plan.json", FEATURES / "train_complete.json", FEATURES / "val_complete.json",
         *baseline_inputs]},
        "runtime": {name: importlib.metadata.version(name) for name in
                    ("numpy", "pandas", "scipy", "scikit-learn", "h5py", "threadpoolctl")}}
    binding_path = output / "input_binding.json"
    if binding_path.exists() and json.loads(binding_path.read_text()) != binding:
        raise ValueError("Feature inputs, baseline head, or CPU runtime changed.")
    save_json(binding_path, binding)
    if (output / "complete.json").exists():
        for name, expected in json.loads((output / "complete.json").read_text())["outputs_sha256"].items():
            if file_hash(output / name) != expected:
                raise ValueError("Completed study artifact changed.")
        return
    meta, arms = feature_arms("train")
    folds, y = grouped_folds(meta), meta.target.to_numpy()
    save_csv(meta[KEYS].assign(inner_fold=folds), output / "training_folds.csv")
    heads, selections = {}, {}
    for name, features in arms.items():
        candidates = []
        for l2 in protocol["l2_grid"]:
            scores, calls = np.full(len(meta), np.nan), np.full(len(meta), -1)
            for fold in range(5):
                scored, calibrated = folds == fold, folds == (fold + 1) % 5
                fitted = ~(scored | calibrated)
                save_json(output / "status.json", {"status": "running", "phase": "training_only_head_selection",
                    "arm": name, "l2": l2, "fold": fold, "pid": os.getpid(), "updated_at": now()})
                head = fit_cached(output, f"{name}_l2{l2:g}_fold{fold}", features[fitted], y[fitted], l2, 0)
                threshold = operating_metrics(y[calibrated], probabilities(head, features[calibrated]), protocol["calibration_target"])["threshold"]
                scores[scored] = probabilities(head, features[scored])
                calls[scored] = scores[scored] >= threshold
            if not np.isfinite(scores).all() or (calls < 0).any():
                raise ValueError("Incomplete training-only scoring.")
            report = {"l2": l2, "pr_auc": float(average_precision_score(y, scores)),
                "metrics": decision_metrics(y, calls), "by_source": {
                    str(source): decision_metrics(y[meta.source == source], calls[meta.source == source])
                    for source in sorted(meta.source.unique())}}
            candidates.append(report)
            save_json(output / f"{name}_l2{l2:g}_cv.json", report)
        selected = max(candidates, key=selection_key)
        selections[name] = {"selected": selected, "used_outer_validation": False, "candidates": candidates}
        save_json(output / "selection.json", selections)
        starts = {seed: fit_cached(output, f"{name}_final_start{seed}", features, y, selected["l2"], seed)
                  for seed in protocol["final_starts"]}
        delta = float(np.max(np.abs(probabilities(starts[0], features) - probabilities(starts[137], features))))
        if delta > 1e-5:
            raise ValueError("Final heads are not stable across optimization starts.")
        heads[name] = starts[0]
        save_json(output / f"{name}_stability.json", {"maximum_training_probability_difference": delta})
    del arms
    meta, arms = feature_arms("val")
    save_json(output / "status.json", {"status": "running", "phase": "fixed_development_comparison", "updated_at": now()})
    for name, features in arms.items():
        meta[name] = probabilities(heads[name], features)
    direct_weight = protocol["saved_checkpoint_blend"]["direct_weight"]
    meta["saved_checkpoint_blend"] = direct_weight * meta.direct_probability + (1 - direct_weight) * meta.diagnosis_probability
    from converged_probe import load_parameters
    meta["panderm_only"] = probabilities(load_parameters(PANDERM_OUTPUT / "final_start0.npz"), aligned_panderm("val", meta))
    columns = ("direct_probability", "diagnosis_probability", "saved_checkpoint_blend", "panderm_only", *protocol["arms"])
    report, decisions = evaluate_development(meta, score_columns=columns, blend_columns=None,
        minimum_recall=protocol["calibration_target"], seed=protocol["development_policy_seed"])
    report["protocol_limitations"] = protocol["limitations"]
    reference = decisions.loc[decisions.candidate.eq("saved_checkpoint_blend")].reset_index(drop=True)
    report["paired_intervals_against_saved_blend"] = {
        name: paired_decision_intervals(reference, decisions.loc[decisions.candidate.eq(name)].reset_index(drop=True), iterations=1000)
        for name in protocol["arms"]}
    save_json(output / "development_report.json", report)
    save_csv(decisions, output / "development_decisions.csv")
    save_csv(meta, output / "development_scores.csv")
    for name, values in report["results"].items():
        print(name, json.dumps(values["metrics"]), flush=True)
    save_json(output / "complete.json", {"completed_at": now(), "automatic_promotion": False,
        "outputs_sha256": {path.name: file_hash(path) for path in output.iterdir()
            if path.is_file() and path.suffix in {".json", ".npz", ".csv"} and path.name not in {"status.json", "progress.json", "complete.json"}}})
    save_json(output / "status.json", {"status": "complete", "updated_at": now(), "automatic_promotion": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.prepare_only:
        plan(args.output)
        return
    with execution_lock(args.output):
        protocol = plan(args.output)
        execute(args.output, protocol)


if __name__ == "__main__":
    main()
