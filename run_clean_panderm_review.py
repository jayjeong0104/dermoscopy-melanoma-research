"""Train a PanDerm-only probe without the locally contaminated incumbent lineage.

This is reused-development evidence, not independent or clinical validation.
The official encoder's private pretraining identities remain unauditable.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import scipy
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

from artifact_io import save_csv
from compare_model_runs import file_hash, operating_metrics
from converged_probe import fit, load_parameters, probabilities, save_parameters
from extract_foundation_features import load_features
from foundation_common import ROOT, BASELINE, KEYS, aggregate_features, fingerprint, save_json
from operating_policy import crossfit_policy, decision_metrics, validate_folds
from targeted_sampling import assert_training_only

SOURCE = ROOT / "artifacts/context_recovery_20261001"
OUTPUT = ROOT / "artifacts/clean_panderm_review_20261005"
GRID = (.001, .01, .1)
CODE = ("run_clean_panderm_review.py", "converged_probe.py", "export_converged_head.py",
    "extract_foundation_features.py", "foundation_common.py", "operating_policy.py",
    "compare_model_runs.py", "targeted_sampling.py", "artifact_io.py")
LIMITATION = (
    "Only the official frozen PanDerm Base full-frame embeddings enter this head; no incumbent "
    "features, checkpoint or predictions are used. Local encoder warm-start contamination is "
    "excluded by construction, but private PanDerm pretraining overlap is unverified. The "
    "validation cohort and full-frame recipe have already informed development, so this is "
    "not an independent confirmation set. No comparison against contaminated retained-model "
    "metrics establishes a performance gain. Patient separation is limited by available IDs. "
    "Target is melanoma versus non-melanoma. No test/HIBA access, image extraction, promotion, "
    "or clinical-use claim."
)


def now():
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def execution_lock(output):
    output.mkdir(parents=True, exist_ok=True)
    with (output / "study.lock").open("a+b") as stream:
        if os.name == "nt":
            import msvcrt
            if stream.tell() == 0:
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            if os.name == "nt":
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def prepare(output):
    feature_root = SOURCE / "features/panderm_full_frame"
    source_plan = json.loads((SOURCE / "experiment_plan.json").read_text())
    if file_hash(SOURCE / "code_snapshot/extract_panderm_context.py") != source_plan["code_sha256"]["extract_panderm_context.py"]:
        raise ValueError("Archived frozen-encoder extraction source changed.")
    for original, expected in source_plan["input_sha256"].items():
        name = original.replace("\\", "/").rsplit("/", 1)[-1]
        if name in {"train_split.csv", "val_split.csv", "panderm_base.pth", "modeling_finetune.py"}:
            path = BASELINE / name if name.endswith("_split.csv") else ROOT / "artifacts/pretrained/panderm_base" / name
            if file_hash(path) != expected:
                raise ValueError(f"Feature source input changed: {name}")
    inputs = [SOURCE / "experiment_plan.json", feature_root / "complete.json",
        feature_root / "train_manifest.json", feature_root / "val_manifest.json",
        BASELINE / "train_split.csv", BASELINE / "val_split.csv",
        ROOT / "artifacts/pretrained/panderm_base/panderm_base.pth",
        ROOT / "artifacts/pretrained/panderm_base/modeling_finetune.py",
        SOURCE / "code_snapshot/extract_panderm_context.py"]
    plan = {"version": 1, "created_for": "2026-10-05 accuracy and lineage review",
        "input_sha256": {str(path.relative_to(ROOT)): file_hash(path) for path in inputs},
        "code_sha256": {name: file_hash(ROOT / name) for name in CODE},
        "runtime": {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__},
        "feature_recipe": "768-dimensional frozen official PanDerm Base full-frame lesion means only",
        "incumbent_features_or_predictions": False, "l2_grid": list(GRID),
        "training_protocol": "Five grouped source/target-stratified folds, seed 20261002; rotate 60% fitting, 20% calibration, 20% scoring. Standardization fits fitting rows only.",
        "selection": "Prefer every source OOF sensitivity >= .95, then source-macro specificity and pooled AP; otherwise highest worst-source sensitivity, macro specificity, AP.",
        "calibration_target": .96, "final_starts": [0, 137],
        "maximum_gradient": 1e-6, "start_probability_tolerance": 1e-4,
        "outer_selection": False, "test_or_hiba_access": False, "automatic_promotion": False,
        "limitations": LIMITATION}
    output.mkdir(parents=True, exist_ok=True)
    path = output / "experiment_plan.json"
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError("Protocol, source, inputs or runtime changed; refusing silent resume.")
    save_json(path, plan)
    snapshot = output / "code_snapshot"
    snapshot.mkdir(exist_ok=True)
    for name in CODE:
        destination = snapshot / name
        if destination.exists() and file_hash(destination) != plan["code_sha256"][name]:
            raise ValueError("Source snapshot changed.")
        if not destination.exists():
            shutil.copy2(ROOT / name, destination)
    return plan


def only_panderm(split):
    if split not in {"train", "val"}:
        raise ValueError("Only train and reused development validation are permitted.")
    rows, values = load_features(SOURCE, "panderm_full_frame", split)
    if values.shape[1] != 768:
        raise ValueError("Expected PanDerm-only features; fused features are forbidden.")
    meta, features = aggregate_features(rows, values)
    return rows, meta, features


def grouped_folds(meta):
    folds = np.full(len(meta), -1, dtype=int)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20261002)
    strata = meta.source.astype(str) + "::" + meta.target.astype(str)
    for fold, (_, held) in enumerate(splitter.split(meta, strata, meta.split_group_id)):
        folds[held] = fold
    validate_folds(meta, folds)
    return folds


def selection_key(report):
    worst = min(item["recall"] for item in report["by_source"].values())
    macro = float(np.mean([item["specificity"] for item in report["by_source"].values()]))
    return (1, macro, report["pr_auc"], 0) if worst >= .95 else (0, worst, macro, report["pr_auc"])


def fit_cached(output, name, x, y, l2, seed):
    spec = {"shape": list(x.shape), "l2": l2, "seed": seed,
        "features_sha256": hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest(),
        "labels_sha256": fingerprint(y.tolist()), "solver_sha256": file_hash(ROOT / "converged_probe.py")}
    path, marker = output / f"{name}.npz", output / f"{name}.json"
    if marker.exists():
        report = json.loads(marker.read_text())
        if report["fit_spec"] != spec or file_hash(path) != report["parameters_sha256"]:
            raise ValueError("Completed fit or training inputs changed.")
        return load_parameters(path)
    save_json(output / "progress.json", {"fit": name, "pid": os.getpid(), "updated_at": now()})
    print(f"Fitting {name}", flush=True)
    parameters, report = fit(x, y, l2, seed)
    save_parameters(path, parameters)
    report.update({"fit_spec": spec, "parameters_sha256": file_hash(path)})
    save_json(marker, report)
    return parameters


def run(output, plan):
    if (output / "complete.json").exists():
        for name, expected in json.loads((output / "complete.json").read_text())["outputs_sha256"].items():
            if file_hash(output / name) != expected:
                raise ValueError("Completed output changed.")
        print("Completed study verified; no rerun.", flush=True)
        return
    save_json(output / "status.json", {"status": "running", "phase": "training_only_selection", "pid": os.getpid(), "updated_at": now()})
    train_images, meta, x = only_panderm("train")
    folds, y = grouped_folds(meta), meta.target.to_numpy()
    save_csv(meta[KEYS].assign(inner_fold=folds), output / "training_cv_folds.csv")
    choices = []
    for l2 in plan["l2_grid"]:
        scores, calls, thresholds = np.full(len(y), np.nan), np.full(len(y), -1), np.full(len(y), np.nan)
        for fold in range(5):
            scoring, calibration = folds == fold, folds == (fold + 1) % 5
            fitting = ~(scoring | calibration)
            groups = [set(meta.loc[mask, "split_group_id"]) for mask in (fitting, calibration, scoring)]
            if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
                raise ValueError("Fitting/calibration/scoring groups overlap.")
            head = fit_cached(output, f"cv_l2_{l2:g}_fold{fold}", x[fitting], y[fitting], l2, 0)
            threshold = operating_metrics(y[calibration], probabilities(head, x[calibration]), plan["calibration_target"])["threshold"]
            scores[scoring] = probabilities(head, x[scoring])
            thresholds[scoring] = threshold
            calls[scoring] = (scores[scoring] >= threshold).astype(int)
        if not np.isfinite(scores).all() or (calls < 0).any():
            raise ValueError("Incomplete OOF scoring.")
        report = {"l2": l2, "pr_auc": float(average_precision_score(y, scores)),
            "metrics": decision_metrics(y, calls),
            "by_source": {str(source): decision_metrics(y[meta.source == source], calls[meta.source == source]) for source in sorted(meta.source.unique())}}
        choices.append(report)
        save_json(output / f"cv_l2_{l2:g}_summary.json", report)
        save_csv(meta[KEYS].assign(probability=scores, threshold=thresholds, prediction=calls, scoring_fold=folds), output / f"cv_l2_{l2:g}_predictions.csv")
    selected = max(choices, key=selection_key)
    save_json(output / "selection.json", {"selected": selected, "used_outer_validation": False,
        "all_source_sensitivity_floor_met": min(item["recall"] for item in selected["by_source"].values()) >= .95})
    heads = {seed: fit_cached(output, f"final_start{seed}", x, y, selected["l2"], seed) for seed in plan["final_starts"]}
    stability = float(np.max(np.abs(probabilities(heads[0], x) - probabilities(heads[137], x))))
    if stability > plan["start_probability_tolerance"]:
        raise ValueError("Final fits are not start stable.")
    from export_converged_head import export
    export(output / "final_start0.npz", output / "panderm_only_research_head.keras", x[:512])
    del x
    save_json(output / "status.json", {"status": "running", "phase": "fixed_outer_development_evaluation", "pid": os.getpid(), "updated_at": now()})
    val_images, meta, x = only_panderm("val")
    assert_training_only(train_images, [val_images])
    results = {}
    for seed in plan["final_starts"]:
        metrics, predictions = crossfit_policy(meta, probabilities(heads[seed], x), minimum_recall=plan["calibration_target"])
        path = output / f"start{seed}_validation_predictions.csv"
        save_csv(predictions, path)
        results[f"start{seed}"] = {"metrics": metrics, "predictions_sha256": file_hash(path),
            "by_source": {str(source): decision_metrics(group.target.to_numpy(), group.prediction.to_numpy()) for source, group in predictions.groupby("source")}}
        print(f"start{seed}: accuracy={metrics['accuracy']:.5f}, sensitivity={metrics['recall']:.5f}, AP={metrics['pr_auc']:.5f}", flush=True)
    save_json(output / "validation_report.json", {"results": results, "start_probability_difference": stability,
        "decision": "research_baseline_only_independent_confirmation_required", "retained_model_comparison_valid": False,
        "local_incumbent_lineage_excluded": True, "pretrained_overlap_verified": False,
        "test_or_hiba_access": False, "automatic_promotion": False, "limitations": LIMITATION})
    outputs = {path.name: file_hash(path) for path in output.iterdir() if path.is_file() and path.suffix in {".json", ".csv", ".npz", ".keras"} and path.name not in {"status.json", "progress.json", "complete.json"}}
    save_json(output / "complete.json", {"outputs_sha256": outputs, "completed_at": now()})
    save_json(output / "status.json", {"status": "complete", "updated_at": now(), "automatic_promotion": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, default=OUTPUT)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    output = args.experiment_dir.resolve()
    with execution_lock(output):
        plan = prepare(output)
        if args.prepare_only:
            return
        try:
            with threadpool_limits(limits=2):
                run(output, plan)
        except BaseException as error:
            save_json(output / "status.json", {"status": "failed", "error": str(error), "updated_at": now()})
            raise


if __name__ == "__main__":
    main()
