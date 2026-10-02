"""Controlled full-frame versus center-crop fusion with grouped training CV."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import scipy
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

from artifact_io import save_csv
from compare_model_runs import file_hash
from context_study_common import verify_context_plan
from converged_probe import probabilities
from extract_foundation_features import load_features
from foundation_common import ROOT, KEYS, REFERENCE, LIMITATION, aggregate_features, save_json
from operating_policy import crossfit_policy, promotion_gate, paired_decision_intervals, decision_metrics
from run_convergence_study import verify_source, train_cached, now
from run_foundation_experiment import assert_idle, FEATURE_PYTHON
from study_precision import paired_precision

SOURCE = ROOT / "artifacts/foundation_recovery_20260909"
GRID = (.001, .01, .1)
ARMS = ("center_fusion", "full_frame_fusion")
CODE = ("run_context_study.py", "context_study_common.py", "extract_panderm_context.py",
    "converged_probe.py", "run_convergence_study.py", "export_converged_head.py", "artifact_io.py",
    "foundation_common.py", "extract_foundation_features.py", "operating_policy.py",
    "compare_model_runs.py", "study_precision.py", "targeted_sampling.py", "run_foundation_experiment.py",
    "run_wsl_python.sh", "launch_context_study.sh")


def assert_context_idle():
    assert_idle()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            argv = (entry / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        if argv and b"python" in argv[0] and any(
                Path(os.fsdecode(arg)).name == "extract_panderm_context.py" for arg in argv[1:] if arg):
            raise RuntimeError(f"An orphaned context extractor is still active: PID {entry.name}")


def prepare(output):
    previous = json.loads((SOURCE / "experiment_plan.json").read_text())
    inputs = dict(previous["input_sha256"])
    inputs[str(ROOT / "artifacts/convergence_recovery_20261001/validation_report.json")] = file_hash(
        ROOT / "artifacts/convergence_recovery_20261001/validation_report.json")
    plan = {"version": 1, "source_provenance": verify_source(SOURCE),
        "input_sha256": inputs, "code_sha256": {name: file_hash(ROOT / name) for name in CODE},
        "runtime": {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__},
        "hypothesis": "Preserving the full frame may recover context lost by the center crop; paired frozen-feature recipes have equal dimensions.",
        "arms": ARMS, "l2_grid": GRID, "selection": "Three source/target-stratified grouped training folds, seed 20261001; mean specificity@95 then mean AP. Choose L2 per arm before outer validation.",
        "primary_candidate": "full_frame_fusion", "matched_control": "center_fusion",
        "classifier": "Float64 converged logistic head; max gradient <=1e-6, final starts 0/137 must agree within 1e-4 on training probabilities",
        "calibration_target": .96,
        "calibration_rationale": "Prospective follow-up uses a conservative target after prior reused-development sensitivity undershot .95. Prior study remains failed; this is not independent confirmation.",
        "gate": "Both candidate starts pass original retained-model gate, paired precision/specificity lower CI >0, and recall/specificity/AP/ROC nondecreasing versus matched .96-calibrated center-fusion control.",
        "preprocessing": "Full-frame aspect-preserving bilinear resize to fit 224x224, mean-color centered padding; no tissue segmentation or label changes",
        "test_inference": False, "automatic_promotion": False, "limitations": LIMITATION}
    plan = json.loads(json.dumps(plan))
    path = output / "experiment_plan.json"
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError("Context protocol, source artifacts or runtime changed.")
    save_json(path, plan)
    snapshot = output / "code_snapshot"
    snapshot.mkdir(exist_ok=True)
    for name in CODE:
        destination = snapshot / name
        if destination.exists() and file_hash(destination) != plan["code_sha256"][name]:
            raise ValueError("Context source snapshot changed.")
        if not destination.exists():
            shutil.copy2(ROOT / name, destination)
    return plan


def grouped_folds(meta):
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=20261001)
    strata = meta.source.astype(str) + "::" + meta.target.astype(str)
    folds = np.full(len(meta), -1, dtype=int)
    for fold, (_, held) in enumerate(splitter.split(meta, strata, meta.split_group_id)):
        folds[held] = fold
    if meta.assign(fold=folds).groupby("split_group_id").fold.nunique().max() != 1:
        raise ValueError("Training groups cross inner CV folds.")
    if any(set(meta.loc[folds == fold, "target"]) != {0, 1} for fold in range(3)):
        raise ValueError("Each inner fold needs both classes.")
    return folds


def features(output, split):
    meta, blocks = None, []
    for directory, encoder in ((SOURCE, "incumbent"), (SOURCE, "panderm_base"), (output, "panderm_full_frame")):
        rows, values = load_features(directory, encoder, split)
        frame, values = aggregate_features(rows, values)
        if meta is not None and not meta[KEYS].equals(frame[KEYS]):
            raise ValueError("Context feature identities differ.")
        meta = frame
        blocks.append(values)
    return meta, {"center_fusion": np.concatenate(blocks[:2], axis=1),
                  "full_frame_fusion": np.concatenate([blocks[0], blocks[2]], axis=1)}


def heads(output, plan):
    meta, x = features(output, "train")
    y, folds = meta.target.to_numpy(), grouped_folds(meta)
    save_csv(meta[KEYS].assign(inner_fold=folds), output / "training_cv_folds.csv")
    selected = {}
    for arm in ARMS:
        choices = []
        for l2 in GRID:
            reports = []
            for fold in range(3):
                training = folds != fold
                _, report = train_cached(output, f"cv_{arm}_l2_{l2:g}_fold{fold}",
                    x[arm][training], y[training], l2, 0, extra=(x[arm][~training], y[~training]))
                reports.append(report["inner_metrics"])
            choices.append({"l2": l2, "mean_specificity": float(np.mean([r["specificity"] for r in reports])),
                            "mean_ap": float(np.mean([r["pr_auc"] for r in reports])), "fold_metrics": reports})
        selected[arm] = max(choices, key=lambda r: (r["mean_specificity"], r["mean_ap"]))
    save_json(output / "selection.json", {"selected": selected, "outer_validation_used": False})
    models, stability = {}, {}
    for arm in ARMS:
        models[arm] = {}
        for seed in (0, 137):
            models[arm][seed], _ = train_cached(output, f"final_{arm}_start{seed}", x[arm], y, selected[arm]["l2"], seed)
        delta = np.abs(probabilities(models[arm][0], x[arm]) - probabilities(models[arm][137], x[arm]))
        stability[arm] = float(delta.max())
        if delta.max() > 1e-4:
            raise ValueError("Final context classifier is not start-stable.")
    from export_converged_head import export
    for arm in ARMS:
        export(output / f"final_{arm}_start0.npz", output / f"{arm}_research_head.keras", x[arm][:512])
    del x
    val, x = features(output, "val")
    reference = pd.read_csv(REFERENCE)
    if not val[KEYS].equals(reference[KEYS]):
        raise ValueError("Outer validation differs from retained reference.")
    baseline, reference = crossfit_policy(reference, reference.probability.to_numpy(), folds=reference.policy_fold.to_numpy())
    results = {}
    for arm in ARMS:
        for seed in (0, 137):
            metrics, decisions = crossfit_policy(reference, probabilities(models[arm][seed], x[arm]),
                minimum_recall=plan["calibration_target"], folds=reference.policy_fold.to_numpy())
            path = output / f"{arm}_start{seed}_validation_predictions.csv"
            save_csv(decisions, path)
            results[f"{arm}_start{seed}"] = {"metrics": metrics, "predictions_sha256": file_hash(path),
                "paired_precision": paired_precision(reference, decisions),
                "paired_decisions": paired_decision_intervals(reference, decisions),
                "gate": promotion_gate(baseline, metrics),
                "by_source": {str(k): decision_metrics(g.target.to_numpy(), g.prediction.to_numpy()) for k, g in decisions.groupby("source")}}
            print(f"{arm} start {seed}: precision={metrics['precision']:.5f}, sensitivity={metrics['recall']:.5f}", flush=True)
    passing = []
    for seed in (0, 137):
        candidate, control = results[f"full_frame_fusion_start{seed}"], results[f"center_fusion_start{seed}"]
        guards = {key: candidate["metrics"][key] >= control["metrics"][key] for key in ("recall", "specificity", "pr_auc", "roc_auc")}
        candidate["matched_control_guards"] = guards
        passed = (candidate["gate"]["passes"] and all(guards.values())
            and candidate["paired_precision"]["delta_intervals_95"]["precision"]["lower"] > 0
            and candidate["paired_decisions"]["delta_intervals_95"]["specificity"]["lower"] > 0)
        passing.append(bool(passed))
    save_json(output / "validation_report.json", {"baseline": baseline, "results": results,
        "start_stability": stability, "both_starts_pass": all(passing),
        "decision": "promising_needs_independent_confirmation" if all(passing) else "retain_baseline",
        "test_inference": False, "automatic_promotion": False, "limitations": LIMITATION})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.experiment_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (ROOT / "artifacts/recovery_pilot.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert_context_idle()
        plan = prepare(output)
        if (output / "complete.json").exists():
            complete = json.loads((output / "complete.json").read_text())
            if complete["report_sha256"] != file_hash(output / "validation_report.json"):
                raise ValueError("Completed context report changed.")
            print("Context study already complete; not restarting.")
            return
        try:
            if not (output / "features/panderm_full_frame/complete.json").exists():
                environment = os.environ.copy()
                environment["LD_LIBRARY_PATH"] = "/usr/lib/wsl/lib"
                with (output / "extract.log").open("a") as log:
                    child = subprocess.Popen([FEATURE_PYTHON, str(ROOT / "extract_panderm_context.py"),
                        "--experiment-dir", str(output)], cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
                    try:
                        save_json(output / "status.json", {"status": "running", "phase": "extract_full_frame", "pid": child.pid,
                            "driver_pid": os.getpid(), "updated_at": now()})
                        print(f"Started full-frame GPU extraction: PID {child.pid}", flush=True)
                        code = child.wait()
                    except BaseException:
                        child.terminate()
                        child.wait()
                        raise
                if code:
                    raise RuntimeError(f"Context extraction exited with code {code}")
            verify_context_plan(output)
            save_json(output / "status.json", {"status": "running", "phase": "grouped_cv_and_validation", "pid": os.getpid(), "updated_at": now()})
            with threadpool_limits(limits=2):
                heads(output, plan)
            save_json(output / "complete.json", {"report_sha256": file_hash(output / "validation_report.json"), "completed_at": now()})
            save_json(output / "status.json", {"status": "complete", "updated_at": now(), "automatic_promotion": False})
        except BaseException as error:
            save_json(output / "status.json", {"status": "failed", "updated_at": now(), "error": str(error)})
            raise


if __name__ == "__main__":
    main()
