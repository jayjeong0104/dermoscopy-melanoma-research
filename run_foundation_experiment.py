"""Locked frozen-representation comparison, resumable by content-verified phases."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pandas as pd

from compare_model_runs import file_hash
from foundation_common import (ASSETS, BASELINE, ENCODERS, LIMITATION, REFERENCE, ROOT,
                               error_audit, inner_partition, lesion_metadata, save_json, validate_splits)
from operating_policy import crossfit_policy, paired_decision_intervals, promotion_gate

FEATURE_PYTHON = "/mnt/d/SkinCancerData/venvs/skin-cancer-features/bin/python"
CODE = ("foundation_common.py", "extract_foundation_features.py", "train_foundation_head.py",
        "run_foundation_experiment.py", "launch_foundation_experiment.sh", "operating_policy.py",
        "compare_model_runs.py", "targeted_sampling.py", "train_multitask.py", "run_wsl_python.sh",
        "artifact_io.py")
BASELINE_SHA256 = "62c7603e41ee4d30f43f6b4cb2c129cedb3fe16efd8cd1a9fae9fef967538171"
SOURCE_SHA256 = "63fbb03283bb5d61ebd24e4790b66f3495996cc37f000f7003eec0a277bd1ab1"
WEIGHTS_SHA256 = "be1e0fb108b3bc58721cb5195f136c948160799438f222acf1fd142230ac1ff1"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def prepare(experiment: Path) -> dict:
    if file_hash(BASELINE / "best_model.keras") != BASELINE_SHA256:
        raise ValueError("The preserved model no longer matches the retained reference.")
    if file_hash(ASSETS / "modeling_finetune.py") != SOURCE_SHA256:
        raise ValueError("Official encoder source differs from the inspected pinned revision.")
    if file_hash(ASSETS / "panderm_base.pth") != WEIGHTS_SHA256:
        raise ValueError("PanDerm checkpoint differs from the official download recorded for this study.")
    previous = json.loads((REFERENCE.parent / "seed42_validation_report.json").read_text())
    if file_hash(REFERENCE) != previous["results"]["baseline"]["predictions_sha256"]:
        raise ValueError("Archived baseline predictions changed.")
    splits = validate_splits()
    inputs = [BASELINE / f"{name}_split.csv" for name in splits]
    inputs += [BASELINE / "best_model.keras", REFERENCE, ASSETS / "modeling_finetune.py",
               ASSETS / "panderm_base.pth"]
    plan = {"version": 1, "purpose": "Change pretrained representation rather than repeat local warm-start tweaks.",
        "input_sha256": {str(path): file_hash(path) for path in inputs},
        "code_sha256": {name: file_hash(ROOT / name) for name in CODE},
        "encoders": list(ENCODERS), "target": "melanoma_vs_non_melanoma",
        "representation_control": "Retained EfficientNetV2B1 trained backbone, pooled before existing head.",
        "candidate": "Frozen PanDerm Base ViT-B/16, official checkpoint and LP CLS embedding.",
        "pretraining_provenance": {"commit": "fd7a80748ba7fc3e203fed88f909f4689d0d6f24",
            "checkpoint_url": "https://drive.google.com/file/d/17J4MjsZu3gdBP6xAQi_NMDVvH65a00HB/view",
            "source_url": "https://github.com/SiyuanYan1/PanDerm",
            "paper": "https://www.nature.com/articles/s41591-025-03747-y",
            "exhaustive_pretraining_overlap_verified": False,
            "reason": "Private pretraining images/identities are not accessible. Author describes curated pretraining to reduce benchmark overlap; not independently verified for this checkpoint."},
        "images": {name: len(frame) for name, frame in splits.items()},
        "training_lesions": len(lesion_metadata(splits["train"])),
        "training_melanoma_lesions": int(lesion_metadata(splits["train"]).target.sum()),
        "features": {"batch_size": 4, "chunk_size": 256, "views": 1,
                     "aggregation": "Mean image embeddings per lesion; every training photograph used.",
                     "test_images_read": False, "raw_images_cached_in_memory": False},
        "head": {"framework": "TensorFlow", "type": "standardized linear logistic classifier",
                 "l2_grid": [.0001, .001, .01], "learning_rate": .003, "batch_size": 128,
                 "maximum_epochs": 200, "minimum_epochs": 40, "patience": 30,
                 "selection": "Train-only grouped 80/20 holdout: specificity at fitted 95% sensitivity, AP tie-break.",
                 "seeds": [42, 137], "class_weights": None, "normalization_fit": "head training rows only"},
        "comparison": "Both seeds against retained decisions and matching-seed incumbent feature probe. Fixed five group-separated threshold folds; .95 calibration target.",
        "gate": "Recall >= .95 and baseline; specificity >= baseline + .01; AP and ROC nondecreasing; paired specificity lower CI > 0; same-seed probe metrics nondecreasing.",
        "test_inference": False, "automatic_promotion": False, "limitations": LIMITATION}
    experiment.mkdir(parents=True, exist_ok=True)
    path = experiment / "experiment_plan.json"
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError("Protocol changed; refusing silent resume.")
    save_json(path, plan)
    snapshot = experiment / "code_snapshot"
    snapshot.mkdir(exist_ok=True)
    for name in CODE:
        shutil.copy2(ROOT / name, snapshot / name)
    reference = pd.read_csv(REFERENCE)
    error_audit(reference, splits["val"], experiment)
    meta = lesion_metadata(splits["train"])
    meta.assign(inner_holdout=inner_partition(meta).astype(int)).to_csv(experiment / "training_inner_partition.csv", index=False)
    return plan


def verify_plan(experiment: Path) -> dict:
    plan = json.loads((experiment / "experiment_plan.json").read_text())
    expected_code = dict(plan["code_sha256"])
    repair_path = experiment / "io_repair.json"
    if repair_path.exists():
        repair = json.loads(repair_path.read_text())
        allowed = {"foundation_common.py", "train_foundation_head.py", "run_foundation_experiment.py"}
        if (repair["version"] != 1 or repair["scope"] != "artifact_io_only"
                or repair["original_plan_sha256"] != file_hash(experiment / "experiment_plan.json")
                or set(repair["code_overrides"]) != allowed
                or set(repair["additional_code_sha256"]) != {"artifact_io.py"}):
            raise ValueError("Invalid I/O-only recovery record.")
        for name, change in repair["code_overrides"].items():
            if (change["before_sha256"] != expected_code[name]
                    or file_hash(experiment / "code_snapshot" / name) != change["before_sha256"]
                    or file_hash(experiment / "io_repair_snapshot" / name) != change["after_sha256"]):
                raise ValueError(f"I/O recovery provenance changed: {name}")
            expected_code[name] = change["after_sha256"]
        expected_code.update(repair["additional_code_sha256"])
        if file_hash(experiment / "io_repair_snapshot/artifact_io.py") != expected_code["artifact_io.py"]:
            raise ValueError("I/O recovery helper snapshot changed.")
    for name, expected in expected_code.items():
        if file_hash(ROOT / name) != expected:
            raise ValueError(f"Experiment code changed: {name}")
    for name, expected in plan["input_sha256"].items():
        if file_hash(Path(name)) != expected:
            raise ValueError(f"Experiment input changed: {name}")
    return plan


def phase(experiment: Path, name: str, command: list[str], patterns: list[str]) -> None:
    verify_plan(experiment)
    marker = experiment / "phase_done" / f"{name}.json"
    if marker.exists():
        saved = json.loads(marker.read_text())
        if saved["command"] != command:
            raise ValueError("Completed phase command changed.")
        for path, expected in saved["outputs"].items():
            if file_hash(experiment / path) != expected:
                raise ValueError(f"Completed phase output changed: {path}")
        print(f"Verified completed phase: {name}", flush=True)
        return
    with (experiment / f"{name}.log").open("a") as log:
        environment = os.environ.copy()
        if command[0] == FEATURE_PYTHON:
            # Do not let TensorFlow's CUDA 12 search path override PyTorch's CUDA 11 runtime.
            environment["LD_LIBRARY_PATH"] = "/usr/lib/wsl/lib"
        child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=environment)
        try:
            save_json(experiment / "status.json", {"status": "running", "phase": name, "pid": child.pid,
                                                   "driver_pid": os.getpid(), "updated_at": now()})
            print(f"Started {name}: PID {child.pid}", flush=True)
            code = child.wait()
        except BaseException:
            child.terminate()
            child.wait()
            raise
    if code:
        raise RuntimeError(f"{name} failed with exit code {code}; see {name}.log")
    outputs = {}
    for pattern in patterns:
        paths = list(experiment.glob(pattern))
        if not paths:
            raise ValueError(f"Phase did not produce {pattern}")
        outputs.update({str(path.relative_to(experiment)): file_hash(path) for path in paths})
    save_json(marker, {"command": command, "outputs": outputs, "completed_at": now()})


def evaluate(experiment: Path) -> None:
    plan = verify_plan(experiment)
    reference = pd.read_csv(REFERENCE)
    baseline, reference = crossfit_policy(reference, reference.probability.to_numpy(), folds=reference.policy_fold.to_numpy())
    reports, predictions = {}, {}
    for encoder in ENCODERS:
        directory = experiment / "heads" / encoder
        reports[encoder] = json.loads((directory / "validation_report.json").read_text())
        predictions[encoder] = {}
        for seed in plan["head"]["seeds"]:
            path = directory / f"seed{seed}_validation_predictions.csv"
            if file_hash(path) != reports[encoder]["results"][str(seed)]["predictions_sha256"]:
                raise ValueError("Head predictions changed.")
            predictions[encoder][seed] = pd.read_csv(path)
    comparisons = {}
    for seed in plan["head"]["seeds"]:
        candidate = reports["panderm_base"]["results"][str(seed)]["metrics"]
        control = reports["incumbent"]["results"][str(seed)]["metrics"]
        gate = promotion_gate(baseline, candidate)
        intervals = {"retained": paired_decision_intervals(reference, predictions["panderm_base"][seed]),
            "matched_probe": paired_decision_intervals(predictions["incumbent"][seed], predictions["panderm_base"][seed])}
        guards = {key: candidate[key] >= control[key] for key in ("recall", "specificity", "pr_auc", "roc_auc")}
        passes = gate["passes"] and all(guards.values()) and all(
            item["delta_intervals_95"]["specificity"]["lower"] > 0 for item in intervals.values())
        comparisons[str(seed)] = {"candidate": candidate, "control": control, "baseline_gate": gate,
                                  "control_guards": guards, "intervals": intervals, "passes": bool(passes)}
    passed = all(result["passes"] for result in comparisons.values())
    save_json(experiment / "validation_report.json", {"baseline": baseline, "comparisons": comparisons,
        "both_seeds_pass": passed, "decision": "promising_needs_independent_confirmation" if passed else "retain_baseline",
        "test_inference": False, "automatic_promotion": False, "limitations": LIMITATION})


def assert_idle() -> None:
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            argv = (entry / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        names = [Path(os.fsdecode(arg)).name for arg in argv[1:] if arg]
        if argv and b"python" in argv[0] and any(
            name.endswith(".py") and (name.startswith(("train", "evaluate", "finalize", "run_"))
                                       or name == "extract_foundation_features.py") for name in names):
            raise RuntimeError(f"Another training/evaluation process is active: PID {entry.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    experiment = args.experiment_dir.resolve()
    experiment.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (ROOT / "artifacts/recovery_pilot.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert_idle()
        if (experiment / "experiment_plan.json").exists():
            verify_plan(experiment)
        else:
            prepare(experiment)
        if args.prepare_only:
            print("Plan and training-only selection partition prepared; no test images read.")
            return
        if (experiment / "validation_report.json").exists():
            marker = experiment / "phase_done/report.json"
            if not marker.exists() or json.loads(marker.read_text())["sha256"] != file_hash(experiment / "validation_report.json"):
                raise ValueError("Unverified completed report.")
            print("Experiment already complete; no jobs restarted.")
            return
        try:
            for encoder in ("panderm_base", "incumbent"):
                interpreter = FEATURE_PYTHON if encoder == "panderm_base" else sys.executable
                phase(experiment, f"extract_{encoder}", [interpreter, str(ROOT / "extract_foundation_features.py"),
                    "--experiment-dir", str(experiment), "--encoder", encoder], [f"features/{encoder}/*.npz", f"features/{encoder}/*manifest.json", f"features/{encoder}/complete.json"])
                phase(experiment, f"train_{encoder}", [sys.executable, str(ROOT / "train_foundation_head.py"),
                    "--experiment-dir", str(experiment), "--encoder", encoder], [f"heads/{encoder}/*.keras", f"heads/{encoder}/*.csv", f"heads/{encoder}/*.json"])
            evaluate(experiment)
            save_json(experiment / "phase_done/report.json", {"sha256": file_hash(experiment / "validation_report.json")})
            save_json(experiment / "status.json", {"status": "complete", "updated_at": now(), "automatic_promotion": False})
        except BaseException as error:
            save_json(experiment / "status.json", {"status": "failed", "updated_at": now(), "error": str(error)})
            raise


if __name__ == "__main__":
    main()
