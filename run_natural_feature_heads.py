"""Training-only head selection for the natural-lesion image checkpoint."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from artifact_io import save_csv
from compare_model_runs import file_hash, operating_metrics
from converged_probe import probabilities
from development_evaluation import confusion_metrics, evaluate_development
from extract_clean_reference import load_extracted
from foundation_common import ROOT, KEYS, save_json
from high_recall_selection import evaluate_training_rank_quality
from operating_policy import paired_decision_intervals
from run_clean_panderm_review import SOURCE as PANDERM_SOURCE, execution_lock, fit_cached, grouped_folds
from run_precision_rebuild import aligned_panderm, saved_checkpoint_blend

RUN = ROOT / "artifacts/natural_lesion_20261006/training/run_20261006_174318"
FEATURES = ROOT / "artifacts/progress_review_20261007/natural_features"
OUTPUT = ROOT / "artifacts/progress_review_20261007/natural_heads"
PANDERM_AUDIT = ROOT / "artifacts/progress_review_20261007/panderm_selection_audit/training_rank_quality.json"
CODE = ("run_natural_feature_heads.py", "high_recall_selection.py", "converged_probe.py",
        "development_evaluation.py", "extract_clean_reference.py", "run_clean_panderm_review.py",
        "run_precision_rebuild.py", "foundation_common.py", "operating_policy.py", "artifact_io.py",
        "checkpoint_lesion_scores.py", "compare_model_runs.py", "extract_foundation_features.py")


def now():
    return datetime.now(timezone.utc).isoformat()


def panderm_input_hashes():
    cache = PANDERM_SOURCE / "features/panderm_full_frame"
    paths = [PANDERM_SOURCE / "experiment_plan.json", cache / "complete.json",
             cache / "train_manifest.json", cache / "val_manifest.json"]
    return {path.relative_to(ROOT).as_posix(): file_hash(path) for path in paths}


def prepare(output):
    source = json.loads((FEATURES / "plan.json").read_text())
    reference = saved_checkpoint_blend(RUN)
    if source["model_sha256"] != reference["checkpoint_sha256"]:
        raise ValueError("Natural features do not belong to the selected natural checkpoint.")
    panderm_audit = json.loads(PANDERM_AUDIT.read_text())
    if panderm_audit["criterion_sha256"] != file_hash(ROOT / "high_recall_selection.py"):
        raise ValueError("Pure-PanDerm training selection used a different criterion.")
    for name, expected in panderm_audit["inputs_sha256"].items():
        if file_hash(ROOT / name) != expected:
            raise ValueError("Pure-PanDerm training selection evidence changed.")
    fixed_panderm_l2 = float(panderm_audit["selected_candidate"].removeprefix("l2_"))
    protocol = {"version": 2, "arms": ["natural_b1", "natural_b1_plus_panderm"],
        "evaluation_arms": ["natural_b1", "natural_b1_plus_panderm", "corrected_panderm_only"],
        "fixed_panderm_l2": fixed_panderm_l2, "panderm_training_selection_sha256": file_hash(PANDERM_AUDIT),
        "l2_grid": [.01, .1, 1.], "folds": 5, "training_fold_seed": 20261002,
        "final_starts": [0, 137], "threads": 2, "calibration_target": .95,
        "development_policy_seed": 20261006,
        "selection": "Fixed training-only mean within-source specificity at95% recall over inner scoring folds; each head fitted on60%, separate20%calibration and20%scoring. Source-fold ranking thresholds use scoring labels only for hyperparameter selection, never deployment.",
        "selection_ties": "Worst-source specificity, then macro average precision, then declared L2 order.",
        "reference_blend": reference, "feature_plan_sha256": file_hash(FEATURES / "plan.json"),
        "panderm_feature_inputs_sha256": panderm_input_hashes(),
        "checkpoint_score_aggregation": "Direct sigmoid(mean image logits), arithmetic mean diagnosis probabilities; verified per-image cache.",
        "code_sha256": {name: file_hash(ROOT / name) for name in CODE},
        "runtime": {name: importlib.metadata.version(name) for name in
            ("numpy", "pandas", "scipy", "scikit-learn", "h5py", "threadpoolctl")},
        "automatic_promotion": False, "test_or_hiba_access": False,
        "limitations": ["Encoder saw all training groups, so inner head scores condition on that training exposure.",
            "Encoder and checkpoint were chosen using reused development validation; this is not independent confirmation.",
            "No L2 grid, ranking rule, or source gate changes after results. Every selected arm is reported.",
            "Empirical calibration target95% does not guarantee95% on scoring cases or within every source.",
            "PanDerm private pretraining overlap remains unverified.",
            "Feature heads are research artifacts, not a complete JPEG clinical deployment package."]}
    output.mkdir(parents=True, exist_ok=True)
    path = output / "experiment_plan.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("Natural head protocol changed; use a fresh output directory.")
    save_json(path, protocol)
    for name in CODE:
        path = output / "code_snapshot" / name
        path.parent.mkdir(exist_ok=True)
        data = (ROOT / name).read_bytes()
        if path.exists() and path.read_bytes() != data:
            raise ValueError("Natural head source snapshot changed.")
        if not path.exists():
            path.write_bytes(data)
    return protocol


def feature_arms(split):
    meta, clean = load_extracted(FEATURES, split)
    panderm = aligned_panderm(split, meta)
    return meta, {"natural_b1": clean, "natural_b1_plus_panderm": np.concatenate([clean, panderm], axis=1),
        "corrected_panderm_only": panderm}


def checkpoint_scores(split):
    from checkpoint_lesion_scores import load_checkpoint_scores
    return load_checkpoint_scores(FEATURES, RUN, split=split)


def execute(output, protocol):
    if (output / "complete.json").exists():
        marker = json.loads((output / "complete.json").read_text())
        for name, expected in marker["outputs_sha256"].items():
            if file_hash(output / name) != expected:
                raise ValueError("Completed natural head artifact changed.")
        return
    binding = {"feature_plan_sha256": file_hash(FEATURES / "plan.json"),
        "train_complete_sha256": file_hash(FEATURES / "train_complete.json"),
        "panderm_feature_inputs_sha256": panderm_input_hashes()}
    if binding["feature_plan_sha256"] != protocol["feature_plan_sha256"]:
        raise ValueError("Natural feature extraction changed after protocol preparation.")
    if binding["panderm_feature_inputs_sha256"] != protocol["panderm_feature_inputs_sha256"]:
        raise ValueError("PanDerm feature inputs changed after protocol preparation.")
    path = output / "training_input_binding.json"
    if path.exists() and json.loads(path.read_text()) != binding:
        raise ValueError("Training feature binding changed.")
    save_json(path, binding)
    meta, arms = feature_arms("train")
    folds, truth = grouped_folds(meta), meta.target.to_numpy()
    save_csv(meta[KEYS].assign(inner_fold=folds), output / "training_folds.csv")
    heads, selections = {}, {}
    for arm in protocol["arms"]:
        features = arms[arm]
        training_scores = meta[KEYS].assign(inner_fold=folds).copy()
        policy_reports = {}
        for l2 in protocol["l2_grid"]:
            name = f"l2_{l2:g}"
            scores, calls = np.full(len(meta), np.nan), np.full(len(meta), -1)
            for fold in range(protocol["folds"]):
                score, calibration = folds == fold, folds == (fold + 1) % protocol["folds"]
                fit = ~(score | calibration)
                save_json(output / "status.json", {"status": "running", "phase": "training_only_selection",
                    "arm": arm, "l2": l2, "fold": fold, "updated_at": now(), "pid": os.getpid()})
                head = fit_cached(output, f"{arm}_l2{l2:g}_fold{fold}", features[fit], truth[fit], l2, 0)
                threshold = operating_metrics(truth[calibration], probabilities(head, features[calibration]),
                    protocol["calibration_target"])["threshold"]
                scores[score] = probabilities(head, features[score])
                calls[score] = scores[score] >= threshold
            if not np.isfinite(scores).all() or (calls < 0).any():
                raise ValueError("Incomplete training scoring.")
            training_scores[name] = scores
            policy_reports[name] = {"metrics": confusion_metrics(truth, calls, scores),
                "by_source": {str(source): confusion_metrics(truth[meta.source == source],
                    calls[meta.source == source], scores[meta.source == source]) for source in sorted(meta.source.unique())}}
        quality = evaluate_training_rank_quality(training_scores, tuple(policy_reports),
            minimum_recall=protocol["calibration_target"])
        selected = quality["selected_candidate"]
        if selected is None:
            raise ValueError(f"No informative training-ranked candidate for {arm}.")
        l2 = float(selected.removeprefix("l2_"))
        selections[arm] = {"selected_l2": l2, "training_rank_quality": quality,
            "disjoint_calibration_scoring_diagnostics": policy_reports,
            "outer_validation_used_for_selection": False}
        save_csv(training_scores, output / f"{arm}_training_oof_scores.csv")
        save_json(output / "selection.json", selections)
        fitted = {seed: fit_cached(output, f"{arm}_final_start{seed}", features, truth, l2, seed)
            for seed in protocol["final_starts"]}
        delta = float(np.max(np.abs(probabilities(fitted[0], features) - probabilities(fitted[137], features))))
        if delta > 1e-5:
            raise ValueError("Natural head optimization starts disagree.")
        heads[arm] = fitted[0]
        save_json(output / f"{arm}_stability.json", {"maximum_training_probability_difference": delta,
            "interpretation": "Optimizer stability, not independent generalization variance."})
    arm, l2 = "corrected_panderm_only", protocol["fixed_panderm_l2"]
    features = arms[arm]
    selections[arm] = {"selected_l2": l2, "selection_source": "Previously completed training-only PanDerm OOF criterion audit.",
        "outer_validation_used_for_selection": False}
    save_json(output / "selection.json", selections)
    fitted = {seed: fit_cached(output, f"{arm}_final_start{seed}", features, truth, l2, seed)
        for seed in protocol["final_starts"]}
    delta = float(np.max(np.abs(probabilities(fitted[0], features) - probabilities(fitted[137], features))))
    if delta > 1e-5:
        raise ValueError("Fixed pure-PanDerm optimization starts disagree.")
    heads[arm] = fitted[0]
    save_json(output / f"{arm}_stability.json", {"maximum_training_probability_difference": delta,
        "interpretation": "Optimizer stability, not independent generalization variance."})
    del arms
    save_json(output / "training_complete.json", {"completed_at": now(), "outer_validation_used_for_selection": False,
        "selection_sha256": file_hash(output / "selection.json"),
        "heads_sha256": {arm: file_hash(output / f"{arm}_final_start0.npz") for arm in heads}})
    # This is the first access to natural validation feature arrays in this study.
    meta, arms = feature_arms("val")
    checkpoint = checkpoint_scores("val")
    if not meta[KEYS].equals(checkpoint[KEYS]):
        raise ValueError("Canonical checkpoint score identities differ from feature rows.")
    meta["feature_cache_arithmetic_mean_direct_probability"] = meta.direct_probability
    meta["direct_probability"] = checkpoint.direct_probability.to_numpy()
    meta["diagnosis_probability"] = checkpoint.diagnosis_probability.to_numpy()
    save_json(output / "validation_input_binding.json", {"val_complete_sha256": file_hash(FEATURES / "val_complete.json")})
    weight = protocol["reference_blend"]["direct_weight"]
    meta["natural_saved_checkpoint_blend"] = weight * meta.direct_probability + (1 - weight) * meta.diagnosis_probability
    for arm, features in arms.items():
        meta[arm] = probabilities(heads[arm], features)
    columns = ("direct_probability", "diagnosis_probability", "natural_saved_checkpoint_blend", *protocol["evaluation_arms"])
    report, decisions = evaluate_development(meta, score_columns=columns, blend_columns=None,
        minimum_recall=protocol["calibration_target"], seed=protocol["development_policy_seed"])
    report["protocol_limitations"] = protocol["limitations"]
    reference = decisions.loc[decisions.candidate.eq("natural_saved_checkpoint_blend")].reset_index(drop=True)
    report["paired_intervals_against_natural_saved_blend"] = {arm: paired_decision_intervals(reference,
        decisions.loc[decisions.candidate.eq(arm)].reset_index(drop=True), iterations=1000) for arm in protocol["evaluation_arms"]}
    for values in report["results"].values():
        values["all_source_recall_floor_met"] = all(
            item["recall"] >= protocol["calibration_target"] for item in values["by_source"].values())
    save_json(output / "development_report.json", report)
    save_csv(meta, output / "development_scores.csv")
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
    parser.add_argument("--wait-for-training-features", action="store_true")
    args = parser.parse_args()
    with execution_lock(args.output), threadpool_limits(limits=2):
        protocol = prepare(args.output)
        if args.prepare_only:
            return
        while not (FEATURES / "train_complete.json").exists():
            if not args.wait_for_training_features:
                raise FileNotFoundError("Natural training features are not complete.")
            save_json(args.output / "status.json", {"status": "waiting_for_training_features", "updated_at": now(), "pid": os.getpid()})
            time.sleep(10)
        execute(args.output, protocol)


if __name__ == "__main__":
    main()
