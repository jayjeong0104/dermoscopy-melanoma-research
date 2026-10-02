"""Bounded, resumable old-data versus added-SIIM experiment. No test inference."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd

from compare_model_runs import compare_frames, file_hash, load_validation, write_report

ROOT = Path(__file__).resolve().parent
BASELINE = ROOT / "artifacts/melanoma_v1/run_20260822_111527"
ARMS = {
    "control": ("combined_ham_bcn_derm12345", "grouped_folds.csv"),
    "added_siim": ("combined_ham_bcn_derm12345_siim2020", "training_folds.csv"),
}


def assert_same_holdout(reference: pd.DataFrame, candidate: pd.DataFrame) -> None:
    columns = ["source", "image_id", "lesion_id", "split_group_id", "sha256", "diagnosis"]
    def canonical(frame: pd.DataFrame) -> pd.DataFrame:
        return frame[columns].astype("string").sort_values(["source", "image_id"]).reset_index(drop=True)
    if not canonical(reference).equals(canonical(candidate)):
        raise ValueError("Pilot holdout identities, hashes, groups or diagnoses differ from baseline.")


def prepare_plan(experiment: Path) -> dict:
    _, baseline_info = load_validation(BASELINE)
    plan = {
        "purpose": "Screen the effect of adding SIIM under a fixed training budget.",
        "baseline": baseline_info, "arms": {},
        "head_epochs": 1, "fine_tune_epochs": 2, "steps_per_epoch": 1000,
        "batch_size": 4, "gradient_accumulation_steps": 2, "seed": 42,
        "finalization_mode": "none", "test_inference": False,
        "selection": "Validation only, one view for all checkpoints, minimum recall 0.95.",
        "training_code_sha256": file_hash(ROOT / "train_multitask.py"),
        "pilot_code_sha256": file_hash(Path(__file__)),
        "limitations": "One seed and short training budget. A promising result needs confirmation; no automatic promotion or full training.",
    }
    for arm, (folder, folds_name) in ARMS.items():
        manifest = ROOT / "data" / folder / "combined_manifest.csv"
        folds = ROOT / "data" / folder / folds_name
        frame = pd.read_csv(manifest, dtype="string", low_memory=False)
        assignments = pd.read_csv(folds, dtype={"split_group_id": "string"})
        frame = frame.merge(assignments, on="split_group_id", validate="many_to_one")
        if len(frame) != len(pd.read_csv(manifest, usecols=["sha256"], dtype="string")):
            raise ValueError("Some manifest rows have no fold assignment.")
        for name, fold in (("test", 0), ("val", 1)):
            reference = pd.read_csv(BASELINE / f"{name}_split.csv", dtype="string", low_memory=False)
            assert_same_holdout(reference, frame.loc[frame.fold.eq(fold)])
        held_out = frame.fold.isin([0, 1])
        for key in ("sha256", "split_group_id", "lesion_id"):
            if set(frame.loc[held_out, key]) & set(frame.loc[~held_out, key]):
                raise ValueError(f"Training/holdout overlap detected: {key}")
        plan["arms"][arm] = {
            "manifest": str(manifest), "folds": str(folds),
            "manifest_sha256": file_hash(manifest), "folds_sha256": file_hash(folds),
            "training_images": int((~held_out).sum()),
        }
    experiment.mkdir(parents=True, exist_ok=True)
    path = experiment / "experiment_plan.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != plan:
        raise ValueError("Inputs or code changed since this experiment was prepared. Do not silently resume.")
    write_report(path, plan)
    return plan


def run_child(experiment: Path, phase: str, command: list[str]) -> None:
    log_path = experiment / f"{phase}.log"
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\nStarted {datetime.now(timezone.utc).isoformat()}\n{command!r}\n")
        log.flush()
        child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        write_report(experiment / "status.json", {
            "status": "running", "phase": phase, "pid": child.pid,
            "log_path": str(log_path), "updated_at": datetime.now(timezone.utc).isoformat(),
        })
        print(f"{phase}: PID {child.pid}; log {log_path}", flush=True)
        try:
            code = child.wait()
        except BaseException:
            child.terminate()
            child.wait()
            raise
    if code:
        raise RuntimeError(f"{phase} failed with exit code {code}; see {log_path}")


def completed_report(experiment: Path) -> Path | None:
    status_path = experiment / "status.json"
    if not status_path.exists():
        return None
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") != "complete":
        return None
    report = experiment / "validation_report.json"
    if not report.is_file():
        raise ValueError("Completed experiment is missing its validation report; refusing automatic restart.")
    return report


def run_training(experiment: Path, plan: dict) -> None:
    for arm, inputs in plan["arms"].items():
        root = experiment / arm
        runs = sorted(root.glob("run_*"))
        if len(runs) > 1:
            raise ValueError(f"Ambiguous resume state in {root}")
        command = [sys.executable, str(ROOT / "train_multitask.py"),
            "--manifest", inputs["manifest"], "--fold-assignments", inputs["folds"],
            "--output-root", str(root), "--test-fold", "0", "--validation-fold", "1",
            "--binary-target", "melanoma", "--backbone", "efficientnetv2b1",
            "--image-size", "300", "--batch-size", "4", "--gradient-accumulation-steps", "2",
            "--head-epochs", "1", "--fine-tune-epochs", "2", "--steps-per-epoch", "1000",
            "--fine-tune-layers", "100", "--learning-rate", "1e-5", "--fine-tune-learning-rate", "3e-6",
            "--risk-loss", "focal", "--focal-gamma", "1.5", "--diagnosis-loss-weight", "0.25",
            "--sampling-alpha", "0.5", "--positive-sampling-fraction", "0.45",
            "--risk-features", "image_only", "--metadata-modality-dropout", "0.5",
            "--minimum-operating-recall", "0.95", "--tta-transforms", "1",
            "--warm-start-model", str(BASELINE / "best_model.keras"), "--weights", "none",
            "--data-workers", "1", "--prefetch-batches", "1", "--seed", "42",
            "--mixed-precision", "auto", "--finalization-mode", "none", "--fit-verbose", "2"]
        if runs:
            command += ["--resume-run-dir", str(runs[0])]
        run_child(experiment, arm, command)
    run_child(experiment, "validation", [sys.executable, str(Path(__file__)),
                                        "--experiment-dir", str(experiment), "--evaluate"])


def evaluate(experiment: Path) -> None:
    # TensorFlow only lives in the evaluation child, never alongside a training child.
    import train_multitask as training
    from finalize_multitask_run import load_run_split, predict_checkpoint

    training.configure_runtime("auto")
    plan = json.loads((experiment / "experiment_plan.json").read_text(encoding="utf-8"))
    if file_hash(BASELINE / "best_model.keras") != plan["baseline"]["model_sha256"]:
        raise ValueError("Baseline model changed during the experiment.")
    baseline_config = json.loads((BASELINE / "run_config.json").read_text(encoding="utf-8"))
    reference = load_run_split(BASELINE, "val", "melanoma")
    checkpoints = [("baseline", BASELINE, "best_model.keras")]
    for arm in ARMS:
        runs = list((experiment / arm).glob("run_*"))
        if len(runs) != 1:
            raise ValueError(f"Expected exactly one run for {arm}")
        checkpoints.extend((f"{arm}_{stage}", runs[0], f"best_{stage}_model.keras")
                           for stage in ("head", "finetune"))
    results, predictions = {}, {}
    for name, run_dir, model_name in checkpoints:
        config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
        for field in ("binary_target", "image_size", "binary_inference_metadata"):
            if config[field] != baseline_config[field]:
                raise ValueError(f"Incompatible pilot inference setting: {field}")
        frame = load_run_split(run_dir, "val", "melanoma")
        assert_same_holdout(reference, frame)
        schema = json.loads((run_dir / "metadata_schema.json").read_text(encoding="utf-8"))
        encoder = training.MetadataEncoder(float(schema["age_median"]), tuple(schema["feature_names"]))
        print(f"Evaluating {name} on validation only with one view.", flush=True)
        direct, diagnosis = predict_checkpoint(
            run_dir / model_name, frame, encoder, int(config["image_size"]), 2, 1, 1, 42, 1,
            experiment / "validation_cache" / f"{name}.npz", force=True,
        )
        lesions, lesion_diagnosis = training.aggregate_predictions_by_lesion(frame, direct, diagnosis)
        if lesion_diagnosis is None:
            raise ValueError("Missing diagnosis probabilities.")
        blend, values, _ = training.select_probability_blend(
            lesions.target.to_numpy(), lesions.probability.to_numpy(),
            training.diagnosis_target_probabilities(lesion_diagnosis, "melanoma"),
        )
        lesions["probability"] = values
        lesions = lesions.set_index(["source", "lesion_id"]).sort_index()
        predictions[name] = lesions
        lesions.to_csv(experiment / f"{name}_validation_predictions.csv")
        results[name] = {"model_path": str(run_dir / model_name),
                         "model_sha256": file_hash(run_dir / model_name), "blend": blend}
        if name != "baseline":
            results[name]["comparison"] = compare_frames(predictions["baseline"], lesions)
    passed = [name for name, result in results.items()
              if result.get("comparison", {}).get("candidate_passes")]
    write_report(experiment / "validation_report.json", {
        "results": results, "promising_candidates": passed,
        "decision": "confirm_promising_candidates" if passed else "retain_baseline_stop_expansion",
        "test_inference": False, "tta_transforms": 1, "automatic_promotion": False,
        "limitations": plan["limitations"],
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    experiment = args.experiment_dir.resolve()
    if args.evaluate:
        evaluate(experiment)
        return
    experiment.mkdir(parents=True, exist_ok=True)
    report = completed_report(experiment)
    if report is not None:
        print(f"Experiment already complete. No training or evaluation restarted. Report: {report}")
        return
    # Kernel-owned lock survives stale PID files and is released on process exit.
    import fcntl
    with (ROOT / "artifacts/recovery_pilot.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not args.prepare_only:
            for entry in Path("/proc").iterdir():
                if not entry.name.isdigit() or int(entry.name) == os.getpid():
                    continue
                try:
                    argv = (entry / "cmdline").read_bytes().split(b"\0")
                except (OSError, ProcessLookupError):
                    continue
                scripts = {Path(os.fsdecode(arg)).name for arg in argv[1:] if arg}
                if argv and b"python" in argv[0] and scripts & {
                    "train_multitask.py", "finalize_multitask_run.py", "evaluate_derm12345.py",
                    "evaluate_multitask_ph2.py", "run_recovery_pilot.py",
                }:
                    raise RuntimeError(f"Another training/evaluation process is active: PID {entry.name}")
        plan = prepare_plan(experiment)
        if args.prepare_only:
            print(json.dumps(plan, indent=2))
            return
        try:
            run_training(experiment, plan)
        except BaseException as error:
            status_path = experiment / "status.json"
            status = json.loads(status_path.read_text()) if status_path.exists() else {}
            write_report(status_path, {**status, "status": "failed", "error": str(error)})
            raise
        write_report(experiment / "status.json", {
            "status": "complete", "updated_at": datetime.now(timezone.utc).isoformat(),
            "report": str(experiment / "validation_report.json"), "automatic_promotion": False,
        })


if __name__ == "__main__":
    main()
