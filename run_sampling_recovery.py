"""Bounded lesion-exposure study with archived controls and validation-only evaluation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path, PurePosixPath
import shutil
import sys

import numpy as np
import pandas as pd

from compare_model_runs import file_hash, write_report
from lesion_sampling import exposure_audit
from operating_policy import crossfit_policy, decision_metrics, paired_decision_intervals, promotion_gate
from run_operating_recovery import EPOCHS, choose_checkpoint, training_command as original_command
from run_recovery_pilot import BASELINE, ROOT, assert_same_holdout, completed_report
from run_targeted_recovery import assert_no_active_job, phase, verify_plan
from targeted_sampling import assert_training_only, canonical

PREVIOUS = ROOT / "artifacts/operating_recovery_20260907"
CODE = ("run_sampling_recovery.py", "lesion_sampling.py", "train_multitask.py",
        "operating_policy.py", "run_operating_recovery.py", "run_targeted_recovery.py",
        "run_recovery_pilot.py", "compare_model_runs.py", "targeted_sampling.py",
        "finalize_multitask_run.py", "audit_operating_policy.py", "run_wsl_python.sh",
        "launch_sampling_recovery.sh")
KEYS = ["source", "lesion_id", "split_group_id", "target", "policy_fold"]
LIMITATION = ("Development study with repeatedly inspected validation, not independent confirmation. "
              "Seed-42 image sampling control is reused from the completed frozen-BatchNorm study; "
              "the legacy image-sampler path is unchanged. A passing screen needs a fresh matched "
              "seed-137 comparison and an independent cohort. No automatic model promotion.")


def artifact_path(value: str) -> Path:
    parts = PurePosixPath(value.replace("\\", "/")).parts
    path = (ROOT / Path(*parts[parts.index("artifacts"):])).resolve()
    if not path.is_relative_to((ROOT / "artifacts").resolve()):
        raise ValueError("Checkpoint path escapes artifacts.")
    return path


def recorded_predictions(name: str) -> pd.DataFrame:
    saved = json.loads((PREVIOUS / "seed42_validation_report.json").read_text(encoding="utf-8"))["results"][name]
    path = PREVIOUS / f"{name}_crossfit_predictions.csv"
    if file_hash(path) != saved["predictions_sha256"]:
        raise ValueError("Archived prediction integrity failure.")
    frame = pd.read_csv(path)
    metrics, decisions = crossfit_policy(frame, frame.probability.to_numpy(), folds=frame.policy_fold.to_numpy())
    for key in ("recall", "specificity", "pr_auc", "roc_auc"):
        if not np.isclose(metrics[key], saved["metrics"][key], rtol=0, atol=1e-12):
            raise ValueError("Archived predictions do not reproduce reported metrics.")
    return decisions


def prepare(experiment: Path) -> dict:
    if completed_report(PREVIOUS) is None:
        raise ValueError("Previous experiment must be complete before reusing controls.")
    previous = json.loads((PREVIOUS / "seed42_validation_report.json").read_text(encoding="utf-8"))
    old_plan = json.loads((PREVIOUS / "experiment_plan.json").read_text(encoding="utf-8"))
    names = ["baseline", *[f"frozen_bn_seed42_{Path(name).stem}" for name in EPOCHS]]
    inputs = [PREVIOUS / name for name in ("seed42_validation_report.json", "experiment_plan.json",
              "development_policy_folds.csv", "status.json")]
    inputs += [BASELINE / name for name in ("train_split.csv", "val_split.csv", "test_split.csv", "metadata_schema.json")]
    for name in names:
        saved = previous["results"][name]
        model = artifact_path(saved["model_path"])
        if file_hash(model) != saved["model_sha256"]:
            raise ValueError("Archived control checkpoint changed.")
        inputs.extend([model, model.parent / "run_config.json", PREVIOUS / f"{name}_crossfit_predictions.csv"])
    control = artifact_path(previous["results"][names[1]]["model_path"]).parent
    config = json.loads((control / "run_config.json").read_text(encoding="utf-8"))
    required = {"freeze_batch_norm": True, "sampling_unit": "image", "risk_loss": "focal",
        "focal_gamma": 1.5, "seed": 42, "head_epochs": 1, "fine_tune_epochs": 2,
        "steps_per_epoch": 1000, "batch_size": 4, "gradient_accumulation_steps": 2,
        "sampling_alpha": .5, "positive_sampling_fraction": .45, "diagnosis_loss_weight": .25,
        "learning_rate": 1e-5, "fine_tune_learning_rate": 3e-6,
        "checkpoint_metric": "lesion_operating", "hard_negative_weights": None,
        "binary_target": "melanoma", "image_size": 300, "binary_inference_metadata": "unknown",
        "backbone": "efficientnetv2b1", "archive_epoch_checkpoints": True,
        "risk_features": "image_only", "label_smoothing": .02, "weight_decay": 1e-5,
        "dropout_rate": .35, "metadata_modality_dropout": .5, "fine_tune_layers": 100,
        "reset_risk_head": False, "weights": "none", "minimum_operating_recall": .95,
        "test_fold": 0, "validation_fold": 1, "sample_fraction": 1.0,
        "mixed_precision": "auto", "tta_transforms": 1, "tta_scheme": "d4_v2"}
    if any(config.get(key, "image" if key == "sampling_unit" else None) != value for key, value in required.items()):
        raise ValueError("Archived control has an incompatible training or inference policy.")
    if artifact_path(config["warm_start_model"]) != (BASELINE / "best_model.keras").resolve():
        raise ValueError("Control did not start from the preserved baseline.")
    train = pd.read_csv(BASELINE / "train_split.csv", low_memory=False)
    holds = [pd.read_csv(BASELINE / f"{split}_split.csv", low_memory=False) for split in ("val", "test")]
    for split, reference in zip(("train", "val", "test"), [train, *holds]):
        path = control / f"{split}_split.csv"
        assert_same_holdout(reference, pd.read_csv(path, low_memory=False))
        inputs.append(path)
    assert_training_only(train, holds)
    manifest = ROOT / "data/combined_ham_bcn_derm12345/combined_manifest.csv"
    fold_path = manifest.parent / "grouped_folds.csv"
    full = pd.read_csv(manifest, low_memory=False).merge(pd.read_csv(fold_path), on="split_group_id", how="left", validate="many_to_one")
    if full.fold.isna().any():
        raise ValueError("Missing training fold assignments.")
    full["target"] = full.diagnosis.eq("MEL").astype(int)
    if not canonical(train).equals(canonical(full.loc[~full.fold.isin([0, 1])])):
        raise ValueError("Training identities/labels changed.")
    for fold, held in zip((1, 0), holds):
        assert_same_holdout(held, full.loc[full.fold.eq(fold)])
    baseline = recorded_predictions("baseline")
    locked = pd.read_csv(PREVIOUS / "development_policy_folds.csv")
    if not baseline[KEYS].astype("string").equals(locked[KEYS].astype("string")):
        raise ValueError("Archived calibration folds changed.")
    for name in names[1:]:
        if not recorded_predictions(name)[KEYS].astype("string").equals(baseline[KEYS].astype("string")):
            raise ValueError("Control validation groups or labels differ.")
    inputs.extend([manifest, fold_path])
    plan = {"version": 1, "hypothesis": "Photograph-count exposure biases lesion-level learning; equalize lesions within source/diagnosis without discarding any images.",
        "baseline": old_plan["baseline"], "manifest": str(manifest), "folds": str(fold_path),
        "code_sha256": {name: file_hash(ROOT / name) for name in CODE},
        "input_sha256": {str(path): file_hash(path) for path in set(inputs)},
        "required_control_config": required, "tensorflow_version": config["tensorflow_version"],
        "exposure_audit": exposure_audit(train), "screening_seed": 42,
        "seed42_control": str(control), "conditional_fresh_matched_seed": 137,
        "maximum_new_training_runs": 3, "minimum_recall": .95,
        "sampling_only_intervention": True, "normalization_frozen_in_both_arms": True,
        "evaluation": "All three epochs; batch4, one d4_v2 view, masked metadata, fixed .5 head blend, original group folds.",
        "acceptance": old_plan["acceptance"], "test_inference": False,
        "automatic_promotion": False, "limitations": LIMITATION}
    experiment.mkdir(parents=True, exist_ok=True)
    path = experiment / "experiment_plan.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != plan:
        raise ValueError("Protocol/code/inputs changed; refusing silent resume.")
    write_report(path, plan)
    write_report(experiment / "sampling_exposure_audit.json", plan["exposure_audit"])
    return plan


def training_command(experiment: Path, plan: dict, unit: str, seed: int) -> list[str]:
    if unit not in {"image", "lesion"} or seed not in {42, 137}:
        raise ValueError("Unsupported sampling experiment arm.")
    return original_command(experiment / unit, plan, "frozen_bn", seed) + ["--sampling-unit", unit]


def comparison(results: dict, predictions: dict, controls: list[str], candidates: list[str], seed: int) -> dict:
    control = choose_checkpoint(controls, results, results["baseline"]["metrics"])
    candidate = choose_checkpoint(candidates, results, results["baseline"]["metrics"])
    old, new = results[control]["metrics"], results[candidate]["metrics"]
    guards = {key: new[key] >= old[key] for key in ("recall", "specificity", "pr_auc", "roc_auc")}
    gate = promotion_gate(results["baseline"]["metrics"], new)
    intervals = {name: paired_decision_intervals(predictions[name], predictions[candidate], seed=seed)
                 for name in ("baseline", control)}
    passed = gate["passes"] and all(guards.values()) and all(
        item["delta_intervals_95"]["specificity"]["lower"] > 0 for item in intervals.values())
    return {"selected": {"control": control, "candidate": candidate}, "passes": bool(passed),
        "baseline_gate": gate, "control_guard": guards, "uncertainty": intervals}


def evaluate(experiment: Path, seed: int) -> dict:
    import train_multitask as training
    from finalize_multitask_run import load_run_split, predict_checkpoint

    plan = verify_plan(experiment)
    runtime = training.configure_runtime("auto")
    if not runtime["gpu_devices"] or training.tf.__version__ != plan["tensorflow_version"]:
        raise ValueError("Evaluation needs the same TensorFlow version and GPU protocol as the archived control.")
    reference = recorded_predictions("baseline")
    results, predictions = {}, {}

    def register(name: str, frame: pd.DataFrame):
        if not frame[KEYS].astype("string").equals(reference[KEYS].astype("string")):
            raise ValueError("Evaluation identities, labels or group folds changed.")
        frame = reference[KEYS].assign(probability=frame.probability.to_numpy())
        metrics, decisions = crossfit_policy(frame, frame.probability.to_numpy(), folds=frame.policy_fold.to_numpy())
        path = experiment / f"{name}_crossfit_predictions.csv"
        decisions.to_csv(path, index=False)
        predictions[name] = decisions
        results[name] = {"metrics": metrics, "predictions_sha256": file_hash(path),
            "by_source": {str(source): decision_metrics(group.target.to_numpy(), group.prediction.to_numpy())
                          for source, group in decisions.groupby("source")}}
        print(f"{name}: sensitivity={metrics['recall']:.5f}, specificity={metrics['specificity']:.5f}, AP={metrics['pr_auc']:.5f}", flush=True)

    register("baseline", reference)
    names = {"image": [], "lesion": []}
    for unit in ("image", "lesion"):
        for filename in EPOCHS:
            name = f"{unit}_seed{seed}_{Path(filename).stem}"
            names[unit].append(name)
            if unit == "image" and seed == 42:
                register(name, recorded_predictions(f"frozen_bn_seed42_{Path(filename).stem}"))
                results[name]["reused_archived_control"] = True
                continue
            runs = list((experiment / unit / f"frozen_bn_seed{seed}").glob("run_*"))
            if len(runs) != 1:
                raise ValueError("Expected exactly one run per sampling arm.")
            run = runs[0]
            config = json.loads((run / "run_config.json").read_text(encoding="utf-8"))
            for key, value in {**plan["required_control_config"], "sampling_unit": unit, "seed": seed}.items():
                if config.get(key) != value:
                    raise ValueError(f"Candidate protocol mismatch: {key}")
            frame = load_run_split(run, "val", "melanoma")
            assert_same_holdout(pd.read_csv(BASELINE / "val_split.csv", low_memory=False), frame)
            schema = json.loads((run / "metadata_schema.json").read_text(encoding="utf-8"))
            encoder = training.MetadataEncoder(float(schema["age_median"]), tuple(schema["feature_names"]))
            direct, diagnosis = predict_checkpoint(run / filename, frame, encoder, 300, 4, 1, 1, 42, 1,
                experiment / "prediction_cache" / f"{name}.npz", force=False, tta_scheme="d4_v2")
            lesions, diagnosis = training.aggregate_predictions_by_lesion(frame, direct, diagnosis)
            lesions["probability"] = .5 * lesions.probability + .5 * training.diagnosis_target_probabilities(diagnosis, "melanoma")
            groups = frame.groupby("lesion_id").split_group_id
            if groups.nunique(dropna=False).ne(1).any():
                raise ValueError("A lesion crosses evaluation groups.")
            lesions["split_group_id"] = lesions.lesion_id.map(groups.first())
            lesions = lesions.sort_values(["source", "lesion_id"]).reset_index(drop=True)
            lesions["policy_fold"] = reference.policy_fold.to_numpy()
            register(name, lesions)
            results[name]["model_path"] = str(run / filename)
            results[name]["model_sha256"] = file_hash(run / filename)
    report = {"seed": seed, "results": results,
        **comparison(results, predictions, names["image"], names["lesion"], seed),
        "test_inference": False, "automatic_promotion": False, "limitations": LIMITATION}
    write_report(experiment / f"seed{seed}_validation_report.json", report)
    return report


def run_seed(experiment: Path, plan: dict, seed: int) -> dict:
    for unit in (("lesion",) if seed == 42 else ("image", "lesion")):
        name = f"{unit}_seed{seed}"
        command = training_command(experiment, plan, unit, seed)
        marker = experiment / "phase_done" / f"{name}.json"
        if marker.exists():
            command = json.loads(marker.read_text(encoding="utf-8"))["command"]
        root = f"{unit}/frozen_bn_seed{seed}/run_*"
        phase(experiment, name, command, [*[f"{root}/{name}" for name in EPOCHS],
            f"{root}/run_config.json", f"{root}/*_split.csv", f"{root}/training_*.csv"])
    phase(experiment, f"validation_seed{seed}", [sys.executable, str(Path(__file__)),
        "--experiment-dir", str(experiment), "--evaluate-seed", str(seed)],
        [f"seed{seed}_validation_report.json", f"*_seed{seed}_*_crossfit_predictions.csv"])
    return json.loads((experiment / f"seed{seed}_validation_report.json").read_text(encoding="utf-8"))


def run_study(experiment: Path, plan: dict) -> dict:
    screen = run_seed(experiment, plan, 42)
    confirmation = run_seed(experiment, plan, 137) if screen["passes"] else None
    passes = bool(confirmation and confirmation["passes"])
    return {"decision": "eligible_for_independent_confirmation" if passes else "retain_baseline",
        "both_seeds_pass": passes, "screening_report": "seed42_validation_report.json",
        "confirmation_report": "seed137_validation_report.json" if confirmation else None,
        "test_inference": False, "automatic_promotion": False, "limitations": LIMITATION}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--evaluate-seed", type=int, choices=(42, 137))
    args = parser.parse_args()
    experiment = args.experiment_dir.resolve()
    if args.evaluate_seed is not None:
        evaluate(experiment, args.evaluate_seed)
        return
    if completed_report(experiment) is not None:
        print("Sampling experiment already complete; nothing restarted.")
        return
    import fcntl
    with (ROOT / "artifacts/recovery_pilot.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert_no_active_job()
        plan = prepare(experiment)
        if args.prepare_only:
            print(json.dumps({"prepared": str(experiment), "max_new_runs": 3, "training_images": plan["exposure_audit"]["training_images"]}))
            return
        if shutil.disk_usage(experiment).free < 4 * 1024**3:
            raise RuntimeError("Less than 4 GiB free for checkpoint archives.")
        try:
            report = run_study(experiment, plan)
            verify_plan(experiment)
            write_report(experiment / "validation_report.json", report)
        except BaseException as error:
            path = experiment / "status.json"
            status = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
            write_report(path, {**status, "status": "failed", "error": str(error)})
            raise
        write_report(experiment / "status.json", {"status": "complete",
            "updated_at": datetime.now(timezone.utc).isoformat(), "automatic_promotion": False})


if __name__ == "__main__":
    main()
