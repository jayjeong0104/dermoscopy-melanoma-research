"""Two-seed controlled hard-negative experiment. Validation only; no model promotion."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

import pandas as pd

from compare_model_runs import compare_frames, file_hash, load_validation, operating_metrics, write_report
from analyze_recovery_errors import error_breakdown, paired_bootstrap
from run_recovery_pilot import BASELINE, ROOT, assert_same_holdout, completed_report, run_child
from targeted_sampling import assert_training_only, canonical, mine_weights

SEEDS = (42, 137)
POLICIES = ("control", "hard_negative")
CODE_FILES = ("run_targeted_recovery.py", "targeted_sampling.py", "train_multitask.py",
              "finalize_multitask_run.py", "compare_model_runs.py", "run_recovery_pilot.py",
              "run_wsl_python.sh", "analyze_recovery_errors.py")


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype="string", low_memory=False)


def prepare_plan(experiment: Path) -> dict:
    _, baseline_info = load_validation(BASELINE)
    manifest = ROOT / "data/combined_ham_bcn_derm12345/combined_manifest.csv"
    folds = manifest.parent / "grouped_folds.csv"
    frame = read_table(manifest)
    joined = frame.merge(read_table(folds), on="split_group_id", how="left", validate="many_to_one")
    if joined.fold.isna().any() or len(frame) != len(joined):
        raise ValueError("Incomplete fold assignments.")
    joined["target"] = joined.diagnosis.eq("MEL").astype(int)
    train = joined.loc[~joined.fold.astype(int).isin([0, 1])]
    holdouts = []
    for name, fold in (("val", 1), ("test", 0)):
        holdout = joined.loc[joined.fold.astype(int).eq(fold)]
        assert_same_holdout(read_table(BASELINE / f"{name}_split.csv"), holdout)
        holdouts.append(holdout)
    if not canonical(train).equals(canonical(read_table(BASELINE / "train_split.csv"))):
        raise ValueError("Targeted experiment must use exactly the incumbent's training images.")
    assert_training_only(train, holdouts)
    plan = {
        "version": 1, "purpose": "Test capped training-only hard-negative exposure against matched controls.",
        "baseline": baseline_info, "manifest": str(manifest), "folds": str(folds),
        "input_sha256": {str(path): file_hash(path) for path in
                         (manifest, folds, BASELINE / "train_split.csv", BASELINE / "val_split.csv",
                          BASELINE / "test_split.csv", BASELINE / "metadata_schema.json")},
        "code_sha256": {name: file_hash(ROOT / name) for name in CODE_FILES},
        "training_images": len(train), "seeds": list(SEEDS), "policies": list(POLICIES),
        "head_epochs": 1, "fine_tune_epochs": 2, "steps_per_epoch": 1000,
        "batch_size": 4, "gradient_accumulation_steps": 2,
        "mining": {"fraction_per_negative_lesion_stratum": 0.2, "relative_weight_cap": 2,
                   "direct_weight": 0.35, "transforms": 1, "source": "training_split_only",
                   "unknown_diagnoses": "not_mined", "labels_changed": False, "images_deleted": False},
        "checkpoint_metric": "validation_lesion_blended_average_precision",
        "screening_tta_transforms": 1, "tta_scheme": "d4_v2", "minimum_recall": 0.95,
        "minimum_absolute_gain": 0.005, "test_inference": False, "automatic_promotion": False,
        "uncertainty": "1000 paired split-group bootstrap samples, conditional on validation selection.",
        "confirmation": "Only if hard-negative gains pass in both seeds: evaluate both selected candidates and baseline with the same corrected eight-view protocol on validation.",
        "limitations": "Development experiment. Two seeds are not independent cohorts. Mining scores are in-sample; potentially mislabeled hard cases are not relabeled. Validation has been used repeatedly, and external development datasets are not pristine confirmation sets. No clinical-use claim.",
    }
    experiment.mkdir(parents=True, exist_ok=True)
    path = experiment / "experiment_plan.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != plan:
        raise ValueError("Experiment inputs/code changed. Refusing silent resume; use a reviewed new experiment.")
    write_report(path, plan)
    return plan


def verify_plan(experiment: Path) -> dict:
    plan = json.loads((experiment / "experiment_plan.json").read_text(encoding="utf-8"))
    for name, digest in plan["code_sha256"].items():
        if file_hash(ROOT / name) != digest:
            raise ValueError(f"Code changed during experiment: {name}")
    for name, digest in plan["input_sha256"].items():
        if file_hash(Path(name)) != digest:
            raise ValueError(f"Data changed during experiment: {name}")
    if file_hash(BASELINE / "best_model.keras") != plan["baseline"]["model_sha256"]:
        raise ValueError("Preserved baseline changed.")
    mining_report = experiment / "mining_report.json"
    if mining_report.exists():
        mining = json.loads(mining_report.read_text(encoding="utf-8"))
        if file_hash(experiment / "training_sampling_weights.csv") != mining["weights_sha256"]:
            raise ValueError("Training sampling weights changed after mining.")
    return plan


def phase(experiment: Path, name: str, command: list[str], output_patterns: list[str]) -> None:
    verify_plan(experiment)
    marker = experiment / "phase_done" / f"{name}.json"
    if marker.exists():
        saved = json.loads(marker.read_text(encoding="utf-8"))
        if saved["command"] != command or not saved["outputs"]:
            raise ValueError(f"Incompatible completed phase: {name}")
        for output, digest in saved["outputs"].items():
            if file_hash(experiment / output) != digest:
                raise ValueError(f"Completed phase output changed: {output}")
        print(f"Already completed {name}; not restarting.", flush=True)
        return
    run_child(experiment, name, command)
    outputs = {}
    for pattern in output_patterns:
        paths = list(experiment.glob(pattern))
        if not paths:
            raise FileNotFoundError(f"Phase {name} did not produce {pattern}")
        outputs.update({str(path.relative_to(experiment)): file_hash(path) for path in paths})
    write_report(marker, {"command": command, "outputs": outputs,
                          "completed_at": datetime.now(timezone.utc).isoformat()})


def training_command(experiment: Path, plan: dict, policy: str, seed: int) -> list[str]:
    root = experiment / f"{policy}_seed{seed}"
    runs = list(root.glob("run_*"))
    if len(runs) > 1:
        raise ValueError(f"Ambiguous run state: {root}")
    command = [sys.executable, str(ROOT / "train_multitask.py"),
        "--manifest", plan["manifest"], "--fold-assignments", plan["folds"],
        "--output-root", str(root), "--test-fold", "0", "--validation-fold", "1",
        "--binary-target", "melanoma", "--backbone", "efficientnetv2b1",
        "--image-size", "300", "--batch-size", "4", "--gradient-accumulation-steps", "2",
        "--head-epochs", "1", "--fine-tune-epochs", "2", "--steps-per-epoch", "1000",
        "--fine-tune-layers", "100", "--learning-rate", "1e-5", "--fine-tune-learning-rate", "3e-6",
        "--risk-loss", "focal", "--focal-gamma", "1.5", "--diagnosis-loss-weight", "0.25",
        "--sampling-alpha", "0.5", "--positive-sampling-fraction", "0.45",
        "--risk-features", "image_only", "--metadata-modality-dropout", "0.5",
        "--minimum-operating-recall", "0.95", "--tta-transforms", "1", "--tta-scheme", "d4_v2",
        "--checkpoint-metric", "lesion_pr_auc",
        "--warm-start-model", str(BASELINE / "best_model.keras"), "--weights", "none",
        "--data-workers", "1", "--prefetch-batches", "1", "--seed", str(seed),
        "--mixed-precision", "auto", "--finalization-mode", "none", "--fit-verbose", "2"]
    if policy == "hard_negative":
        command += ["--hard-negative-weights", str(experiment / "training_sampling_weights.csv")]
    if runs:
        command += ["--resume-run-dir", str(runs[0])]
    command.append("--require-gpu")
    return command


def require_gpu(training) -> None:
    runtime = training.configure_runtime("auto")
    if not runtime["gpu_devices"]:
        raise RuntimeError("No TensorFlow GPU detected. Refusing silent CPU fallback.")
    training.set_random_seeds(42)
    print(json.dumps(runtime, indent=2), flush=True)


def predict(experiment: Path, name: str, run: Path, model_name: str, split: str, transforms: int):
    import train_multitask as training
    from finalize_multitask_run import load_run_split, predict_checkpoint

    config = json.loads((run / "run_config.json").read_text(encoding="utf-8"))
    reference_config = json.loads((BASELINE / "run_config.json").read_text(encoding="utf-8"))
    for key in ("binary_target", "image_size", "binary_inference_metadata"):
        if config[key] != reference_config[key]:
            raise ValueError(f"Incompatible candidate protocol: {key}")
    frame = load_run_split(run, split, "melanoma")
    assert_same_holdout(read_table(BASELINE / f"{split}_split.csv"), frame)
    schema = json.loads((run / "metadata_schema.json").read_text(encoding="utf-8"))
    encoder = training.MetadataEncoder(float(schema["age_median"]), tuple(schema["feature_names"]))
    direct, diagnosis = predict_checkpoint(
        run / model_name, frame, encoder, 300, 2, 1, 1, 42, transforms,
        experiment / "prediction_cache" / f"{name}_{split}_d4_v2_tta{transforms}.npz",
        force=False, tta_scheme="d4_v2",
    )
    lesions, lesion_diagnosis = training.aggregate_predictions_by_lesion(frame, direct, diagnosis)
    diagnosis_target = training.diagnosis_target_probabilities(lesion_diagnosis, "melanoma")
    return frame, lesions, diagnosis_target


def mine(experiment: Path) -> None:
    import train_multitask as training

    plan = verify_plan(experiment)
    require_gpu(training)
    frame, lesions, diagnosis = predict(experiment, "baseline_mining", BASELINE, "best_model.keras", "train", 1)
    assert_training_only(frame, [read_table(BASELINE / f"{name}_split.csv") for name in ("val", "test")])
    direct_weight = plan["mining"]["direct_weight"]
    lesions["probability"] = direct_weight * lesions.probability + (1 - direct_weight) * diagnosis
    weights = mine_weights(frame, lesions, plan["mining"]["fraction_per_negative_lesion_stratum"])
    weights.to_csv(experiment / "training_sampling_weights.csv", index=False)
    strata = weights.assign(hard=weights.sampling_weight.eq(2)).groupby(["source", "diagnosis"]).agg(
        images=("image_id", "size"), hard_images=("hard", "sum"), lesions=("lesion_id", "nunique"))
    write_report(experiment / "mining_report.json", {
        "model_sha256": file_hash(BASELINE / "best_model.keras"), "policy": plan["mining"],
        "training_images": len(weights), "hard_images": int(weights.sampling_weight.eq(2).sum()),
        "weights_sha256": file_hash(experiment / "training_sampling_weights.csv"),
        "strata": strata.reset_index().to_dict(orient="records"), "test_inference": False,
    })
    print(f"Mined {int(weights.sampling_weight.eq(2).sum())} hard training images; all {len(frame)} retained.", flush=True)


def evaluate(experiment: Path) -> None:
    import train_multitask as training

    plan = verify_plan(experiment)
    require_gpu(training)
    results, predictions = {}, {}

    def evaluate_one(name: str, run: Path, model_name: str, transforms: int) -> None:
        _, lesions, diagnosis = predict(experiment, name, run, model_name, "val", transforms)
        blend, values, _ = training.select_probability_blend(
            lesions.target.to_numpy(), lesions.probability.to_numpy(), diagnosis)
        lesions["probability"] = values
        lesions = lesions.set_index(["source", "lesion_id"]).sort_index()
        predictions[name] = lesions
        lesions.to_csv(experiment / f"{name}_validation_predictions.csv")
        results[name] = {"run_dir": str(run), "model_name": model_name,
                         "model_sha256": file_hash(run / model_name), "blend": blend,
                         "tta_transforms": transforms, "tta_scheme": "d4_v2",
                         "metrics": operating_metrics(lesions.target.to_numpy(),
                                                      lesions.probability.to_numpy(), 0.95)}
        threshold = results[name]["metrics"]["threshold"]
        results[name]["by_source"] = error_breakdown(lesions.reset_index(), threshold, "source")
        results[name]["by_diagnosis"] = error_breakdown(lesions.reset_index(), threshold, "diagnosis")

    evaluate_one("baseline", BASELINE, "best_model.keras", 1)
    selected = {}
    for seed in SEEDS:
        for policy in POLICIES:
            arm = f"{policy}_seed{seed}"
            runs = list((experiment / arm).glob("run_*"))
            if len(runs) != 1:
                raise ValueError(f"Expected one completed run in {arm}.")
            names = []
            for stage in ("head", "finetune"):
                name = f"{arm}_{stage}"
                evaluate_one(name, runs[0], f"best_{stage}_model.keras", 1)
                results[name]["baseline_comparison"] = compare_frames(predictions["baseline"], predictions[name])
                names.append(name)
            selected[arm] = max(names, key=lambda name: (results[name]["metrics"]["pr_auc"],
                                                        results[name]["metrics"]["roc_auc"]))
    seed_gates = {}
    split = read_table(BASELINE / "val_split.csv")
    groups = split.groupby(["source", "lesion_id"]).split_group_id
    if groups.nunique(dropna=False).ne(1).any():
        raise ValueError("Validation lesions do not have unique split groups.")
    clusters = groups.first().reindex(predictions["baseline"].index)
    if clusters.isna().any():
        raise ValueError("Missing validation split groups for uncertainty analysis.")
    for seed in SEEDS:
        control = selected[f"control_seed{seed}"]
        hard = selected[f"hard_negative_seed{seed}"]
        control_comparison = compare_frames(predictions[control], predictions[hard])
        old, new = control_comparison["baseline_metrics"], control_comparison["candidate_metrics"]
        not_worse = all(new[key] >= old[key] for key in ("pr_auc", "specificity", "roc_auc"))
        seed_gates[str(seed)] = {"selected_control": control, "selected_hard_negative": hard,
                                "control_comparison": control_comparison,
                                "passes": results[hard]["baseline_comparison"]["candidate_passes"] and not_worse}
        for name in (control, hard):
            results[name]["paired_uncertainty_vs_baseline"] = paired_bootstrap(
                predictions["baseline"].target.to_numpy(), predictions["baseline"].probability.to_numpy(),
                predictions[name].probability.to_numpy(), clusters.to_numpy(),
                results["baseline"]["metrics"]["threshold"], results[name]["metrics"]["threshold"],
                iterations=1000, seed=seed,
            )
    screening_passes = all(gate["passes"] for gate in seed_gates.values())
    confirmation_passes = False
    if screening_passes:
        evaluate_one("baseline_d4", BASELINE, "best_model.keras", 8)
        confirmation_passes = True
        for seed in SEEDS:
            chosen = selected[f"hard_negative_seed{seed}"]
            info = results[chosen]
            name = f"{chosen}_d4"
            evaluate_one(name, Path(info["run_dir"]), info["model_name"], 8)
            results[name]["baseline_comparison"] = compare_frames(predictions["baseline_d4"], predictions[name])
            confirmation_passes &= results[name]["baseline_comparison"]["candidate_passes"]
    write_report(experiment / "validation_report.json", {
        "results": results, "selected": selected, "seed_gates": seed_gates,
        "screening_passes_both_seeds": screening_passes, "d4_confirmation_passes": bool(confirmation_passes),
        "decision": "eligible_for_independent_confirmation" if confirmation_passes else "retain_baseline",
        "test_inference": False, "automatic_promotion": False, "limitations": plan["limitations"],
    })


def assert_no_active_job() -> None:
    scripts_to_check = {"train_multitask.py", "finalize_multitask_run.py", "evaluate_derm12345.py",
                        "evaluate_multitask_ph2.py", "run_recovery_pilot.py", "run_targeted_recovery.py"}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            argv = (entry / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        names = {Path(os.fsdecode(arg)).name for arg in argv[1:] if arg}
        if argv and b"python" in argv[0] and names & scripts_to_check:
            raise RuntimeError(f"Training/evaluation already active: PID {entry.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("mine", "evaluate"))
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    experiment = args.experiment_dir.resolve()
    if args.phase:
        {"mine": mine, "evaluate": evaluate}[args.phase](experiment)
        return
    if completed_report(experiment) is not None:
        print("Experiment already complete; nothing restarted.")
        return
    import fcntl
    (ROOT / "artifacts").mkdir(exist_ok=True)
    with (ROOT / "artifacts/recovery_pilot.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert_no_active_job()
        plan = prepare_plan(experiment)
        if args.prepare_only:
            print(json.dumps(plan, indent=2))
            return
        try:
            child_base = [sys.executable, str(Path(__file__)), "--experiment-dir", str(experiment)]
            phase(experiment, "mining", child_base + ["--phase", "mine"],
                  ["training_sampling_weights.csv", "mining_report.json"])
            for seed in SEEDS:
                for policy in POLICIES:
                    name = f"{policy}_seed{seed}"
                    command = training_command(experiment, plan, policy, seed)
                    # Resuming adds a CLI argument; completed phases retain their original command.
                    marker = experiment / "phase_done" / f"{name}.json"
                    if marker.exists():
                        command = json.loads(marker.read_text(encoding="utf-8"))["command"]
                    phase(experiment, name, command,
                          [f"{name}/run_*/best_*_model.keras", f"{name}/run_*/training_*.csv",
                           f"{name}/run_*/run_config.json", f"{name}/run_*/*_split.csv"])
            phase(experiment, "validation", child_base + ["--phase", "evaluate"], ["validation_report.json"])
            verify_plan(experiment)
        except BaseException as error:
            status_path = experiment / "status.json"
            status = json.loads(status_path.read_text()) if status_path.exists() else {}
            write_report(status_path, {**status, "status": "failed", "error": str(error)})
            raise
        write_report(experiment / "status.json", {"status": "complete",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "report": str(experiment / "validation_report.json"), "automatic_promotion": False})


if __name__ == "__main__":
    main()
