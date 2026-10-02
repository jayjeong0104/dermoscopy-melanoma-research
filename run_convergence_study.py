"""Train-only selection of converged heads and complementary feature fusion."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import re
import shutil
import sys
import time

import numpy as np
import pandas as pd
import scipy
from threadpoolctl import threadpool_limits

from artifact_io import save_csv
from compare_model_runs import file_hash, operating_metrics
from converged_probe import audit_keras_head, fit, load_parameters, probabilities, save_parameters
from extract_foundation_features import load_features
from foundation_common import ROOT, KEYS, REFERENCE, LIMITATION, aggregate_features, inner_partition, save_json
from operating_policy import crossfit_policy, paired_decision_intervals, promotion_gate, decision_metrics
from study_precision import paired_precision

SOURCE = ROOT / "artifacts/foundation_recovery_20260909"
OUTPUT = ROOT / "artifacts/convergence_recovery_20261001"
ARMS = ("incumbent", "panderm_base", "fusion")
GRID = (.0001, .001, .01, .1)
CODE = ("run_convergence_study.py", "converged_probe.py", "artifact_io.py", "foundation_common.py",
        "extract_foundation_features.py", "operating_policy.py", "compare_model_runs.py",
        "study_precision.py", "targeted_sampling.py")


def now():
    return datetime.now(timezone.utc).isoformat()


def local_path(value):
    match = re.match(r"^/mnt/([a-z])/(.*)$", str(value))
    return Path(f"{match[1]}:/{match[2]}") if os.name == "nt" and match else Path(value)


def verify_source(source):
    plan = json.loads((source / "experiment_plan.json").read_text())
    if json.loads((source / "status.json").read_text())["status"] != "complete":
        raise ValueError("Source experiment must be complete; do not compete with its training.")
    marker = json.loads((source / "phase_done/report.json").read_text())
    if file_hash(source / "validation_report.json") != marker["sha256"]:
        raise ValueError("Source validation report changed.")
    for value, expected in plan["input_sha256"].items():
        if file_hash(local_path(value)) != expected:
            raise ValueError(f"Source input changed: {value}")
    for name in ("extract_panderm_base", "train_panderm_base", "extract_incumbent", "train_incumbent"):
        marker = json.loads((source / "phase_done" / f"{name}.json").read_text())
        for value, expected in marker["outputs"].items():
            if file_hash(source / value) != expected:
                raise ValueError(f"Completed source output changed: {value}")
    return {"plan_sha256": file_hash(source / "experiment_plan.json"),
            "report_sha256": file_hash(source / "validation_report.json"),
            "io_repair_sha256": file_hash(source / "io_repair.json"),
            "phase_sha256": {p.name: file_hash(p) for p in sorted((source / "phase_done").glob("*.json"))}}


def prepare(output, source):
    provenance = verify_source(source)
    plan = {"version": 1, "source": str(source), "source_provenance": provenance,
        "code_sha256": {name: file_hash(ROOT / name) for name in CODE},
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "scipy": scipy.__version__, "pandas": pd.__version__},
        "arms": ARMS, "l2_grid": GRID, "objective": "mean binary crossentropy + lambda * sum(weights**2); unpenalized intercept",
        "solver": "float64 L-BFGS-B, analytic gradient, maxiter 2000, gtol 1e-8, ftol 1e-15",
        "convergence_acceptance": "Solver success and full-training maximum absolute gradient <= 1e-6",
        "final_starts": [0, 137], "stability_max_probability_difference": 1e-4,
        "selection": "Same grouped training-only 80/20 holdout; specificity at fitted .95 sensitivity, AP tie-break. Select L2 per arm and nominate one arm before outer validation.",
        "fusion": "Concatenated incumbent and PanDerm mean lesion embeddings, exact identity alignment; no metadata inputs",
        "outer_evaluation": "Both starts, fixed existing group-crossfit folds, .95 calibration target. No outer model selection.",
        "gate": "Existing promotion gate plus positive lower conditional paired precision and specificity intervals for both starts of the train-only nominated arm",
        "diagnostic": "Saved Adam full-training gradient/objective and fixed-L2 .01 converged objective; not a new validation-selected candidate",
        "test_inference": False, "automatic_promotion": False, "limitations": LIMITATION}
    # Compare parsed JSON so tuple/list serialization does not create false changes.
    plan = json.loads(json.dumps(plan))
    path = output / "experiment_plan.json"
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError("Study inputs, protocol, runtime or code changed; refusing silent resume.")
    save_json(path, plan)
    snapshot = output / "code_snapshot"
    snapshot.mkdir(exist_ok=True)
    for name in CODE:
        if not (snapshot / name).exists():
            shutil.copy2(ROOT / name, snapshot / name)
        elif file_hash(snapshot / name) != plan["code_sha256"][name]:
            raise ValueError("Code snapshot changed.")
    return plan


def load_split(source, split):
    frames, arrays = {}, {}
    for encoder in ARMS[:2]:
        rows, features = load_features(source, encoder, split)
        frames[encoder], arrays[encoder] = aggregate_features(rows, features)
    if not frames["incumbent"][KEYS].equals(frames["panderm_base"][KEYS]):
        raise ValueError("Cannot fuse different lesion or label identities.")
    arrays["fusion"] = np.concatenate([arrays[name] for name in ARMS[:2]], axis=1)
    return frames["incumbent"], arrays


def train_cached(output, name, x, y, l2, seed, extra=None):
    path, marker = output / f"{name}.npz", output / f"{name}.json"
    if marker.exists():
        report = json.loads(marker.read_text())
        if report["parameters_sha256"] != file_hash(path) or report["l2"] != l2 or report["seed"] != seed:
            raise ValueError("Completed head changed.")
        return load_parameters(path), report
    iterations = 0
    started = time.monotonic()

    def progress(_):
        nonlocal iterations
        iterations += 1
        if iterations % 25 == 0:
            save_json(output / "progress.json", {"pid": os.getpid(), "fit": name,
                "iteration": iterations, "elapsed_seconds": round(time.monotonic() - started, 1), "updated_at": now()})
            print(f"{name}: iteration {iterations}", flush=True)

    parameters, report = fit(x, y, l2, seed, progress)
    save_parameters(path, parameters)
    report.update({"parameters_sha256": file_hash(path), "training_rows": len(y),
                   "elapsed_seconds": time.monotonic() - started})
    if extra is not None:
        report["inner_metrics"] = operating_metrics(extra[1], probabilities(parameters, extra[0]), .95)
    save_json(marker, report)
    print(f"{name}: converged in {report['iterations']} iterations, gradient={report['gradient_max_abs']:.3g}", flush=True)
    return parameters, report


def run(output, source):
    plan = prepare(output, source)
    final_path = output / "validation_report.json"
    if final_path.exists():
        if json.loads((output / "complete.json").read_text())["report_sha256"] != file_hash(final_path):
            raise ValueError("Final study report changed.")
        print("Study already complete; no refitting.")
        return
    save_json(output / "status.json", {"status": "running", "phase": "training_only_selection", "pid": os.getpid(), "updated_at": now()})
    meta, arrays = load_split(source, "train")
    y, held = meta.target.to_numpy(dtype=int), inner_partition(meta)
    save_csv(meta[KEYS].assign(inner_holdout=held.astype(int)), output / "inner_partition.csv")
    selection, diagnostics = {}, {}
    for arm in ARMS:
        x = arrays[arm]
        choices = []
        for l2 in GRID:
            _, report = train_cached(output, f"inner_{arm}_l2_{l2:g}", x[~held], y[~held], l2, 0,
                                      extra=(x[held], y[held]))
            choices.append(report)
        selection[arm] = max(choices, key=lambda r: (r["inner_metrics"]["specificity"], r["inner_metrics"]["pr_auc"]))
        if arm != "fusion":
            diagnostics[arm] = {"adam": {str(seed): audit_keras_head(source / "heads" / arm / f"seed{seed}.keras", x, y)
                                        for seed in (42, 137)}}
    nominated = max(ARMS, key=lambda arm: (selection[arm]["inner_metrics"]["specificity"], selection[arm]["inner_metrics"]["pr_auc"]))
    save_json(output / "selection.json", {"selected": selection, "nominated_arm": nominated,
              "selection_used_outer_validation": False, "limitations": LIMITATION})
    save_json(output / "status.json", {"status": "running", "phase": "final_training_and_stability", "pid": os.getpid(), "updated_at": now()})
    models, stability = {}, {}
    for arm in ARMS:
        x, l2 = arrays[arm], selection[arm]["l2"]
        models[arm] = {}
        for seed in plan["final_starts"]:
            models[arm][seed], _ = train_cached(output, f"final_{arm}_start{seed}", x, y, l2, seed)
        delta = np.abs(probabilities(models[arm][0], x) - probabilities(models[arm][137], x))
        stability[arm] = {"maximum_training_probability_difference": float(delta.max()),
                          "mean_training_probability_difference": float(delta.mean())}
        if delta.max() > plan["stability_max_probability_difference"]:
            raise RuntimeError(f"Converged head start stability failed: {arm}")
        if arm != "fusion":
            _, audit = train_cached(output, f"diagnostic_{arm}_l2_0.01", x, y, .01, 0)
            diagnostics[arm]["converged_same_l2"] = audit
    save_json(output / "optimization_audit.json", {"diagnostics": diagnostics, "stability": stability})
    del arrays
    # No validation features are opened until every configuration and final fit is fixed.
    save_json(output / "status.json", {"status": "running", "phase": "outer_development_evaluation", "pid": os.getpid(), "updated_at": now()})
    val, features = load_split(source, "val")
    reference = pd.read_csv(REFERENCE)
    if not val[KEYS].equals(reference[KEYS]):
        raise ValueError("Validation identities differ from retained reference.")
    baseline, reference = crossfit_policy(reference, reference.probability.to_numpy(), folds=reference.policy_fold.to_numpy())
    results = {}
    for arm in ARMS:
        for seed in plan["final_starts"]:
            name = f"{arm}_start{seed}"
            metrics, decisions = crossfit_policy(reference, probabilities(models[arm][seed], features[arm]), folds=reference.policy_fold.to_numpy())
            path = output / f"{name}_validation_predictions.csv"
            save_csv(decisions, path)
            intervals = paired_precision(reference, decisions)
            specificity = paired_decision_intervals(reference, decisions)
            gate = promotion_gate(baseline, metrics)
            passes = (gate["passes"] and intervals["delta_intervals_95"]["precision"]["lower"] > 0
                      and specificity["delta_intervals_95"]["specificity"]["lower"] > 0)
            results[name] = {"metrics": metrics, "gate": gate, "paired_precision": intervals,
                "paired_specificity": specificity, "passes": bool(passes), "predictions_sha256": file_hash(path),
                "by_source": {str(key): decision_metrics(group.target.to_numpy(), group.prediction.to_numpy())
                              for key, group in decisions.groupby("source")}}
            print(f"{name}: precision={metrics['precision']:.5f}, sensitivity={metrics['recall']:.5f}, FP={metrics['false_positives']}, FN={metrics['false_negatives']}", flush=True)
    passed = all(results[f"{nominated}_start{seed}"]["passes"] for seed in plan["final_starts"])
    save_json(final_path, {"baseline": baseline, "results": results, "train_only_nominated_arm": nominated,
        "nominated_arm_passes": passed, "decision": "promising_needs_independent_confirmation" if passed else "retain_baseline",
        "optimization_audit": diagnostics, "training_stability": stability, "test_inference": False,
        "automatic_promotion": False, "limitations": LIMITATION})
    save_json(output / "complete.json", {"report_sha256": file_hash(final_path), "completed_at": now()})
    save_json(output / "status.json", {"status": "complete", "updated_at": now(), "automatic_promotion": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--source-dir", type=Path, default=SOURCE)
    args = parser.parse_args()
    output, source = args.output_dir.resolve(), args.source_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    # OS-held lock is released even if Python is killed, unlike a PID-only lockfile.
    with (output / "run.lock").open("a+b") as lock:
        if os.name == "nt":
            import msvcrt
            lock.seek(0, 2)
            if lock.tell() == 0:
                lock.write(b"0")
                lock.flush()
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            with threadpool_limits(limits=2):
                run(output, source)
        except BaseException as error:
            save_json(output / "status.json", {"status": "failed", "updated_at": now(), "error": str(error)})
            raise


if __name__ == "__main__":
    main()
