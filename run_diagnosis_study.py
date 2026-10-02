"""Matched diagnosis-aware versus binary heads with source-aware train-only selection."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import pandas as pd
import scipy
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

import converged_probe as binary
import diagnosis_probe as diagnosis
from artifact_io import save_csv
from audit_error_reduction import compare
from compare_model_runs import file_hash, operating_metrics
from extract_foundation_features import load_features
from foundation_common import ROOT, KEYS, REFERENCE, LIMITATION, aggregate_features, save_json, fingerprint
from operating_policy import crossfit_policy, decision_metrics, promotion_gate, validate_folds
from run_context_study import assert_context_idle
from run_convergence_study import local_path, now
from study_precision import paired_precision

CONTEXT = ROOT / "artifacts/context_recovery_20261001"
SOURCE = ROOT / "artifacts/foundation_recovery_20260909"
OUTPUT = ROOT / "artifacts/diagnosis_recovery_20261001"
ARMS = ("binary", "diagnosis")
GRID = (.001, .01, .1)
CODE = ("run_diagnosis_study.py", "diagnosis_probe.py", "queue_diagnosis_study.py",
    "launch_diagnosis_study.sh", "audit_error_reduction.py", "converged_probe.py", "artifact_io.py",
    "foundation_common.py", "extract_foundation_features.py", "operating_policy.py", "compare_model_runs.py",
    "study_precision.py", "run_context_study.py", "run_convergence_study.py", "run_foundation_experiment.py",
    "context_study_common.py", "targeted_sampling.py", "export_converged_head.py")


def prepare(output):
    parent_path = CONTEXT / "experiment_plan.json"
    parent = json.loads(parent_path.read_text())
    inputs = dict(parent["input_sha256"])
    inputs[str(parent_path)] = file_hash(parent_path)
    plan = {"version": 1, "input_sha256": inputs,
        "code_sha256": {name: file_hash(ROOT / name) for name in CODE},
        "runtime": {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__},
        "hypothesis": "Learning existing diagnosis categories separately may distinguish heterogeneous melanoma lookalikes better than one binary linear boundary.",
        "feature_recipe": "Fixed incumbent plus full-frame PanDerm lesion embeddings, 2048 dimensions; do not choose the view using outer results.",
        "arms": ARMS, "primary_candidate": "diagnosis", "classes": diagnosis.CLASSES,
        "l2_grid": GRID, "class_or_source_weights": None,
        "inner_protocol": "Five grouped source/target-stratified folds seed 20261002. Rotate 60% fit, 20% threshold calibration, 20% scoring; all groups disjoint in each rotation.",
        "selection": "Per arm: prefer all-source OOF sensitivity >=.95, then source-macro specificity and pooled AP. If none feasible, highest worst-source sensitivity then macro specificity then AP. No source-specific thresholds.",
        "calibration_target": .96, "final_starts": [0, 137], "maximum_gradient": 1e-6,
        "start_probability_tolerance": 1e-4, "selection_uses_outer_validation": False,
        "gate": "Both starts: original retained gate; accuracy and precision improve; positive lower paired precision interval; no source has more missed melanomas or false alarms; recall/specificity/AP/ROC nondecreasing versus matched binary control. No automatic promotion.",
        "test_inference": False, "automatic_promotion": False, "limitations": LIMITATION}
    plan = json.loads(json.dumps(plan))
    output.mkdir(parents=True, exist_ok=True)
    path = output / "experiment_plan.json"
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError("Diagnosis protocol, code, inputs or runtime changed; refusing silent resume.")
    for name, expected in inputs.items():
        if file_hash(local_path(name)) != expected:
            raise ValueError(f"Frozen input changed: {name}")
    save_json(path, plan)
    snapshot = output / "code_snapshot"
    snapshot.mkdir(exist_ok=True)
    for name in CODE:
        destination = snapshot / name
        if destination.exists() and file_hash(destination) != plan["code_sha256"][name]:
            raise ValueError("Diagnosis source snapshot changed.")
        if not destination.exists():
            shutil.copy2(ROOT / name, destination)
    return plan


def load_split(split):
    if split not in {"train", "val"}:
        raise ValueError("Test inference is disabled.")
    meta, blocks = None, []
    for directory, encoder in ((SOURCE, "incumbent"), (CONTEXT, "panderm_full_frame")):
        rows, values = load_features(directory, encoder, split)
        frame, values = aggregate_features(rows, values)
        if meta is not None and not meta[KEYS + ["diagnosis"]].equals(frame[KEYS + ["diagnosis"]]):
            raise ValueError("Diagnosis feature identities differ.")
        meta = frame
        blocks.append(values)
    return meta, np.concatenate(blocks, axis=1)


def inner_folds(meta):
    folds = np.full(len(meta), -1, dtype=int)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20261002)
    for fold, (_, held) in enumerate(splitter.split(meta, meta.source.astype(str) + "::" + meta.target.astype(str), meta.split_group_id)):
        folds[held] = fold
    validate_folds(meta, folds)
    return folds


def masks(folds, scoring_fold):
    scoring, calibration = folds == scoring_fold, folds == (scoring_fold + 1) % 5
    return ~(scoring | calibration), calibration, scoring


def selection_key(report):
    worst = min(item["recall"] for item in report["by_source"].values())
    specificity = float(np.mean([item["specificity"] for item in report["by_source"].values()]))
    return (1, specificity, report["pr_auc"], 0) if worst >= .95 else (0, worst, specificity, report["pr_auc"])


def scores(arm, parameters, x):
    if arm == "binary":
        return binary.probabilities(parameters, x)
    return diagnosis.probabilities(parameters, x)[:, list(parameters["classes"]).index("MEL")]


def fit_cached(output, name, arm, x, labels, l2, seed):
    spec = {"arm": arm, "l2": l2, "seed": seed, "shape": list(x.shape),
        "features_sha256": hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest(),
        "labels_sha256": fingerprint(np.asarray(labels).tolist())}
    path, marker = output / f"{name}.npz", output / f"{name}.json"
    module = binary if arm == "binary" else diagnosis
    if marker.exists():
        report = json.loads(marker.read_text())
        if report["fit_spec"] != spec or file_hash(path) != report["parameters_sha256"]:
            raise ValueError("Completed fit or training inputs changed.")
        return module.load_parameters(path), report
    started, iterations = time.monotonic(), 0
    save_json(output / "progress.json", {"pid": os.getpid(), "fit": name,
        "iterations": 0, "elapsed_seconds": 0., "updated_at": now()})
    print(f"Starting {name}", flush=True)

    def progress(_):
        nonlocal iterations
        iterations += 1
        if iterations % 25 == 0:
            save_json(output / "progress.json", {"pid": os.getpid(), "fit": name,
                "iterations": iterations, "elapsed_seconds": round(time.monotonic() - started, 1), "updated_at": now()})
            print(f"{name}: iteration {iterations}", flush=True)

    parameters, report = module.fit(x, labels, l2, seed, callback=progress)
    binary.save_parameters(path, parameters)
    report.update({"fit_spec": spec, "parameters_sha256": file_hash(path), "elapsed_seconds": time.monotonic() - started})
    save_json(marker, report)
    return parameters, report


def run(output, plan):
    dependency = json.loads((CONTEXT / "complete.json").read_text())
    if file_hash(CONTEXT / "validation_report.json") != dependency["report_sha256"]:
        raise ValueError("Context report changed.")
    if (output / "complete.json").exists():
        marker = json.loads((output / "complete.json").read_text())
        for name, expected in marker["outputs_sha256"].items():
            if file_hash(output / name) != expected:
                raise ValueError("Completed diagnosis output changed.")
        print("Diagnosis study already complete; no refitting.", flush=True)
        return
    save_json(output / "dependency.json", {"context_report_sha256": dependency["report_sha256"],
        "context_features_sha256": file_hash(CONTEXT / "features/panderm_full_frame/complete.json"),
        "outer_metrics_used_for_selection": False})
    save_json(output / "status.json", {"status": "running", "phase": "training_only_selection", "pid": os.getpid(), "updated_at": now()})
    meta, x = load_split("train")
    folds, y = inner_folds(meta), meta.target.to_numpy()
    save_csv(meta[KEYS].assign(inner_fold=folds), output / "training_cv_folds.csv")
    selection, models = {}, {}
    for arm in ARMS:
        labels = y if arm == "binary" else meta.diagnosis.to_numpy()
        choices = []
        for l2 in GRID:
            probabilities, calls = np.full(len(y), np.nan), np.full(len(y), -1, dtype=int)
            thresholds = np.full(len(y), np.nan)
            for fold in range(5):
                fit_rows, calibrate, score = masks(folds, fold)
                parameters, _ = fit_cached(output, f"cv_{arm}_l2_{l2:g}_fold{fold}", arm,
                    x[fit_rows], labels[fit_rows], l2, 0)
                threshold = operating_metrics(y[calibrate], scores(arm, parameters, x[calibrate]), plan["calibration_target"])["threshold"]
                probabilities[score] = scores(arm, parameters, x[score])
                thresholds[score] = threshold
                calls[score] = (probabilities[score] >= threshold).astype(int)
            if not np.isfinite(probabilities).all() or (calls < 0).any():
                raise ValueError("Incomplete training-only out-of-fold scores.")
            report = {"l2": l2, "pr_auc": float(average_precision_score(y, probabilities)),
                "metrics": decision_metrics(y, calls), "by_source": {str(source): decision_metrics(y[meta.source == source], calls[meta.source == source]) for source in sorted(meta.source.unique())}}
            choices.append(report)
            save_json(output / f"cv_{arm}_l2_{l2:g}_summary.json", report)
            save_csv(meta[KEYS].assign(probability=probabilities, threshold=thresholds,
                prediction=calls, scoring_fold=folds, calibration_fold=(folds + 1) % 5),
                output / f"cv_{arm}_l2_{l2:g}_predictions.csv")
        selection[arm] = max(choices, key=selection_key)
    save_json(output / "selection.json", {"selected": selection, "used_outer_validation": False})
    save_json(output / "status.json", {"status": "running", "phase": "final_training_and_export", "pid": os.getpid(), "updated_at": now()})
    stability = {}
    for arm in ARMS:
        labels = y if arm == "binary" else meta.diagnosis.to_numpy()
        models[arm] = {seed: fit_cached(output, f"final_{arm}_start{seed}", arm, x, labels,
            selection[arm]["l2"], seed)[0] for seed in plan["final_starts"]}
        probability_function = binary.probabilities if arm == "binary" else diagnosis.probabilities
        difference = np.abs(probability_function(models[arm][0], x) - probability_function(models[arm][137], x))
        stability[arm] = float(difference.max())
        if difference.max() > plan["start_probability_tolerance"]:
            raise ValueError("Diagnosis study start stability failed.")
    from export_converged_head import export
    export(output / "final_binary_start0.npz", output / "binary_research_head.keras", x[:512])
    diagnosis.export(output / "final_diagnosis_start0.npz", output / "diagnosis_research_head.keras", x[:512])
    del x
    save_json(output / "status.json", {"status": "running", "phase": "outer_development_evaluation", "pid": os.getpid(), "updated_at": now()})
    meta, x = load_split("val")
    reference = pd.read_csv(REFERENCE)
    if not meta[KEYS].equals(reference[KEYS]):
        raise ValueError("Diagnosis validation identities differ.")
    baseline, reference = crossfit_policy(reference, reference.probability.to_numpy(), folds=reference.policy_fold.to_numpy())
    results = {}
    for arm in ARMS:
        for seed in plan["final_starts"]:
            name = f"{arm}_start{seed}"
            metrics, decisions = crossfit_policy(reference, scores(arm, models[arm][seed], x),
                minimum_recall=plan["calibration_target"], folds=reference.policy_fold.to_numpy())
            path = output / f"{name}_validation_predictions.csv"
            save_csv(decisions, path)
            audit, _ = compare(reference, decisions, meta)
            results[name] = {"metrics": metrics, "predictions_sha256": file_hash(path),
                "gate": promotion_gate(baseline, metrics), "paired_precision": paired_precision(reference, decisions),
                "paired_error_audit": audit}
            print(f"{name}: precision={metrics['precision']:.5f}, sensitivity={metrics['recall']:.5f}", flush=True)
    passing = []
    for seed in plan["final_starts"]:
        candidate, control = results[f"diagnosis_start{seed}"], results[f"binary_start{seed}"]
        guards = {key: candidate["metrics"][key] >= control["metrics"][key] for key in ("recall", "specificity", "pr_auc", "roc_auc")}
        candidate["matched_control_guards"] = guards
        passing.append(bool(candidate["gate"]["passes"] and all(guards.values())
            and all(candidate["paired_error_audit"]["diagnostic_checks_not_a_replacement_promotion_gate"].values())
            and candidate["paired_precision"]["delta_intervals_95"]["precision"]["lower"] > 0))
    save_json(output / "validation_report.json", {"baseline": baseline, "results": results,
        "both_starts_pass": all(passing), "start_stability": stability,
        "decision": "promising_needs_independent_confirmation" if all(passing) else "retain_baseline",
        "test_inference": False, "automatic_promotion": False, "limitations": LIMITATION})
    outputs = {path.name: file_hash(path) for suffix in ("*.npz", "*.keras", "*.json", "*.csv")
        for path in output.glob(suffix) if path.name not in {"status.json", "progress.json", "complete.json"}}
    save_json(output / "complete.json", {"report_sha256": file_hash(output / "validation_report.json"), "outputs_sha256": outputs, "completed_at": now()})
    save_json(output / "status.json", {"status": "complete", "updated_at": now(), "automatic_promotion": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, default=OUTPUT)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    output = args.experiment_dir.resolve()
    if args.prepare_only:
        prepare(output)
        print(f"Prepared fixed diagnosis protocol: {output}")
        return
    import fcntl
    with (ROOT / "artifacts/recovery_pilot.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert_context_idle()
        plan = prepare(output)
        try:
            with threadpool_limits(limits=2):
                run(output, plan)
        except BaseException as error:
            save_json(output / "status.json", {"status": "failed", "error": str(error), "updated_at": now()})
            raise


if __name__ == "__main__":
    main()
