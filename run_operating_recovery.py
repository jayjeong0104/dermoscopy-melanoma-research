"""Operating-point aligned small-batch normalization study. No test inference."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

from audit_operating_policy import EXPERIMENT as PREVIOUS, cached_lesions
from compare_model_runs import file_hash, load_validation, write_report
from operating_policy import (DIRECT_WEIGHT, POLICY_VERSION, crossfit_policy, decision_metrics,
                              paired_decision_intervals, policy_folds, promotion_gate, selection_score)
from run_recovery_pilot import BASELINE, ROOT, assert_same_holdout, completed_report
from run_targeted_recovery import assert_no_active_job, phase, verify_plan
from targeted_sampling import assert_training_only, canonical

CODE = ("run_operating_recovery.py", "operating_policy.py", "audit_operating_policy.py",
        "train_multitask.py", "finalize_multitask_run.py", "compare_model_runs.py",
        "run_recovery_pilot.py", "run_targeted_recovery.py", "targeted_sampling.py",
        "run_wsl_python.sh", "analyze_recovery_errors.py", "launch_operating_recovery.sh")
POLICIES = ("control", "frozen_bn")
EPOCHS = ("epoch_head_01.keras", "epoch_finetune_01.keras", "epoch_finetune_02.keras")


def prepare(experiment: Path) -> dict:
    _, baseline = load_validation(BASELINE)
    manifest = ROOT / "data/combined_ham_bcn_derm12345/combined_manifest.csv"
    folds_path = manifest.parent / "grouped_folds.csv"
    frame = pd.read_csv(manifest, dtype="string", low_memory=False)
    assignments = pd.read_csv(folds_path, dtype="string", low_memory=False)
    joined = frame.merge(assignments, on="split_group_id", how="left", validate="many_to_one")
    if joined.fold.isna().any() or len(joined) != len(frame):
        raise ValueError("Incomplete training-fold assignments.")
    joined["target"] = joined.diagnosis.eq("MEL").astype(int)
    train = joined.loc[~joined.fold.astype(int).isin([0, 1])]
    holdouts = []
    for name, number in (("val", 1), ("test", 0)):
        held = joined.loc[joined.fold.astype(int).eq(number)]
        reference = pd.read_csv(BASELINE / f"{name}_split.csv", dtype="string", low_memory=False)
        assert_same_holdout(reference, held)
        holdouts.append(held)
    original_train = pd.read_csv(BASELINE / "train_split.csv", dtype="string", low_memory=False)
    if not canonical(train).equals(canonical(original_train)):
        raise ValueError("Training identity or label drift from the preserved baseline.")
    assert_training_only(train, holdouts)
    lesions, _ = cached_lesions(PREVIOUS, "baseline")
    fold_table = lesions[["source", "lesion_id", "split_group_id", "target"]].assign(policy_fold=policy_folds(lesions))
    experiment.mkdir(parents=True, exist_ok=True)
    policy_path = experiment / "development_policy_folds.csv"
    csv = fold_table.to_csv(index=False, lineterminator="\n")
    if policy_path.exists() and policy_path.read_text(encoding="utf-8") != csv:
        raise ValueError("Operating-policy folds changed; refusing silent redefinition.")
    if not policy_path.exists():
        policy_path.write_text(csv, encoding="utf-8", newline="")
    inputs = [manifest, folds_path, policy_path, BASELINE / "metadata_schema.json",
              *[BASELINE / f"{name}_split.csv" for name in ("train", "val", "test")]]
    plan = {
        "version": 1, "hypothesis": "Freezing classifier BatchNorm avoids unstable updates from oversampled batches of four.",
        "baseline": baseline, "manifest": str(manifest), "folds": str(folds_path),
        "input_sha256": {str(path): file_hash(path) for path in inputs},
        "code_sha256": {name: file_hash(ROOT / name) for name in CODE},
        "policies": list(POLICIES), "screening_seed": 42, "conditional_confirmation_seed": 137,
        "head_epochs": 1, "fine_tune_epochs": 2, "steps_per_epoch": 1000,
        "training_images": len(train), "batch_size": 4, "gradient_accumulation_steps": 2,
        "normalization_only_intervention": True, "hard_negative_sampling": False,
        "epoch_retention": "All three epochs per run retained separately; no source image deletion.",
        "policy_version": POLICY_VERSION, "fixed_direct_weight": DIRECT_WEIGHT,
        "tta_transforms": 1, "tta_scheme": "d4_v2", "evaluation_batch_size": 4,
        "threshold_fit": "Five fixed group-separated folds within the existing development validation split; no scoring-fold labels used to choose its threshold.",
        "acceptance": "Cross-fitted sensitivity >=0.95 and no lower than baseline; specificity >= baseline +0.01; AP and ROC AUC no lower than baseline. No decreases on these four metrics versus the matched control. Conditional specificity-delta bootstrap lower bounds >0 versus baseline and control. Require repeat success in seed 137 before independent confirmation.",
        "maximum_training_runs": 4, "automatic_promotion": False, "test_inference": False,
        "limitations": "Repeatedly inspected development validation, not independent testing. Cross-fitting protects threshold fitting only, not adaptive model/epoch selection. A new operating-point objective, not retroactive success under the earlier AP-improvement gate. No guaranteed clinical performance.",
    }
    path = experiment / "experiment_plan.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != plan:
        raise ValueError("Code, data, or protocol changed. Refusing silent resume.")
    write_report(path, plan)
    return plan


def training_command(experiment: Path, plan: dict, policy: str, seed: int) -> list[str]:
    if policy not in POLICIES:
        raise ValueError("Unknown normalization policy.")
    root = experiment / f"{policy}_seed{seed}"
    runs = list(root.glob("run_*"))
    if len(runs) > 1:
        raise ValueError("Ambiguous training resume state.")
    command = [sys.executable, str(ROOT / "train_multitask.py"),
        "--manifest", plan["manifest"], "--fold-assignments", plan["folds"], "--output-root", str(root),
        "--test-fold", "0", "--validation-fold", "1", "--binary-target", "melanoma",
        "--backbone", "efficientnetv2b1", "--image-size", "300", "--batch-size", "4",
        "--gradient-accumulation-steps", "2", "--head-epochs", "1", "--fine-tune-epochs", "2",
        "--steps-per-epoch", "1000", "--fine-tune-layers", "100", "--learning-rate", "1e-5",
        "--fine-tune-learning-rate", "3e-6", "--risk-loss", "focal", "--focal-gamma", "1.5",
        "--diagnosis-loss-weight", "0.25", "--sampling-alpha", "0.5", "--positive-sampling-fraction", "0.45",
        "--risk-features", "image_only", "--metadata-modality-dropout", "0.5",
        "--minimum-operating-recall", "0.95", "--tta-transforms", "1", "--tta-scheme", "d4_v2",
        "--checkpoint-metric", "lesion_operating", "--warm-start-model", str(BASELINE / "best_model.keras"),
        "--weights", "none", "--data-workers", "1", "--prefetch-batches", "1", "--seed", str(seed),
        "--mixed-precision", "auto", "--finalization-mode", "none", "--fit-verbose", "2",
        "--archive-epoch-checkpoints", "--require-gpu"]
    if policy == "frozen_bn":
        command.append("--freeze-batch-norm")
    if runs:
        command += ["--resume-run-dir", str(runs[0])]
    return command


def choose_checkpoint(names: list[str], results: dict, baseline: dict) -> str:
    passing = [name for name in names if promotion_gate(baseline, results[name]["metrics"])["passes"]]
    return max(passing or names, key=lambda name: (selection_score(results[name]["metrics"]),
                                                  results[name]["metrics"]["pr_auc"]))


def evaluate(experiment: Path, seed: int) -> dict:
    import train_multitask as training
    from finalize_multitask_run import load_run_split, predict_checkpoint

    plan = verify_plan(experiment)
    runtime = training.configure_runtime("auto")
    if not runtime["gpu_devices"]:
        raise RuntimeError("No GPU available; evaluation CPU fallback is disabled.")
    reference = load_run_split(BASELINE, "val", "melanoma")
    locked_folds = pd.read_csv(experiment / "development_policy_folds.csv", dtype="string")
    results, predictions = {}, {}

    def one(name: str, run: Path, filename: str):
        config = json.loads((run / "run_config.json").read_text(encoding="utf-8"))
        if config["binary_target"] != "melanoma" or config["image_size"] != 300:
            raise ValueError("Incompatible model target or image resolution.")
        if config["binary_inference_metadata"] != "unknown":
            raise ValueError("Incompatible metadata masking.")
        if name != "baseline":
            if bool(config.get("freeze_batch_norm")) != name.startswith("frozen_bn_"):
                raise ValueError("Checkpoint normalization policy does not match its experiment arm.")
            if config.get("checkpoint_metric") != "lesion_operating" or not config.get("archive_epoch_checkpoints"):
                raise ValueError("Candidate did not use the prescribed operating checkpoint protocol.")
        frame = load_run_split(run, "val", "melanoma")
        assert_same_holdout(reference, frame)
        schema = json.loads((run / "metadata_schema.json").read_text(encoding="utf-8"))
        encoder = training.MetadataEncoder(float(schema["age_median"]), tuple(schema["feature_names"]))
        direct, diagnosis = predict_checkpoint(run / filename, frame, encoder, 300, 4, 1, 1, 42, 1,
            experiment / "prediction_cache" / f"{name}_batch4_tta1.npz", force=False, tta_scheme="d4_v2")
        lesions, diagnosis = training.aggregate_predictions_by_lesion(frame, direct, diagnosis)
        group_map = frame.groupby("lesion_id").split_group_id
        if group_map.nunique(dropna=False).ne(1).any():
            raise ValueError("A lesion crosses split groups.")
        lesions["split_group_id"] = lesions.lesion_id.map(group_map.first())
        lesions["diagnosis_probability"] = training.diagnosis_target_probabilities(diagnosis, "melanoma")
        lesions = lesions.sort_values(["source", "lesion_id"]).reset_index(drop=True)
        identity = lesions[["source", "lesion_id", "split_group_id", "target"]].astype("string")
        if not identity.equals(locked_folds[identity.columns]):
            raise ValueError("Candidate lesion identities differ from locked development folds.")
        scores = (DIRECT_WEIGHT * lesions.probability + (1 - DIRECT_WEIGHT) * lesions.diagnosis_probability).to_numpy()
        metrics, decisions = crossfit_policy(lesions, scores, folds=locked_folds.policy_fold.to_numpy(dtype=int))
        destination = experiment / f"{name}_crossfit_predictions.csv"
        decisions.to_csv(destination, index=False)
        predictions[name] = decisions
        results[name] = {"metrics": metrics, "model_path": str(run / filename),
                         "model_sha256": file_hash(run / filename), "predictions_sha256": file_hash(destination),
                         "by_source": {str(source): decision_metrics(group.target.to_numpy(), group.prediction.to_numpy())
                                       for source, group in decisions.groupby("source")}}
        print(f"{name}: sensitivity={metrics['recall']:.5f}, specificity={metrics['specificity']:.5f}, "
              f"AP={metrics['pr_auc']:.5f}", flush=True)

    one("baseline", BASELINE, "best_model.keras")
    selected = {}
    for policy in POLICIES:
        root = experiment / f"{policy}_seed{seed}"
        runs = list(root.glob("run_*"))
        if len(runs) != 1:
            raise ValueError("Expected exactly one run per policy/seed.")
        names = []
        for filename in EPOCHS:
            name = f"{policy}_seed{seed}_{Path(filename).stem}"
            one(name, runs[0], filename)
            results[name]["baseline_gate"] = promotion_gate(results["baseline"]["metrics"], results[name]["metrics"])
            names.append(name)
        selected[policy] = choose_checkpoint(names, results, results["baseline"]["metrics"])
    control, candidate = selected["control"], selected["frozen_bn"]
    old, new = results[control]["metrics"], results[candidate]["metrics"]
    control_guard = {key: new[key] >= old[key] for key in ("recall", "specificity", "pr_auc", "roc_auc")}
    uncertainties = {name: paired_decision_intervals(predictions[name], predictions[candidate], seed=seed)
                     for name in ("baseline", control)}
    positive_intervals = all(value["delta_intervals_95"]["specificity"]["lower"] > 0
                             for value in uncertainties.values())
    passed = (results[candidate]["baseline_gate"]["passes"] and all(control_guard.values()) and positive_intervals)
    report = {"seed": seed, "results": results, "selected": selected, "control_guard": control_guard,
              "uncertainty_vs_reference": uncertainties, "passes": passed,
              "decision": "repeat_in_second_seed" if passed and seed == 42 else
                          "eligible_for_independent_confirmation" if passed else "retain_baseline",
              "test_inference": False, "automatic_promotion": False, "limitations": plan["limitations"]}
    write_report(experiment / f"seed{seed}_validation_report.json", report)
    return report


def run_seed(experiment: Path, plan: dict, seed: int) -> dict:
    for policy in POLICIES:
        name = f"{policy}_seed{seed}"
        command = training_command(experiment, plan, policy, seed)
        marker = experiment / "phase_done" / f"{name}.json"
        if marker.exists():
            command = json.loads(marker.read_text(encoding="utf-8"))["command"]
        outputs = [f"{name}/run_*/{filename}" for filename in EPOCHS]
        outputs += [f"{name}/run_*/best_*_model.keras", f"{name}/run_*/training_*.csv",
                    f"{name}/run_*/run_config.json", f"{name}/run_*/*_split.csv"]
        phase(experiment, name, command, outputs)
    phase(experiment, f"validation_seed{seed}", [sys.executable, str(Path(__file__)),
        "--experiment-dir", str(experiment), "--evaluate-seed", str(seed)],
        [f"seed{seed}_validation_report.json", f"*_seed{seed}_*_crossfit_predictions.csv"])
    return json.loads((experiment / f"seed{seed}_validation_report.json").read_text(encoding="utf-8"))


def run_study(experiment: Path, plan: dict) -> dict:
    screening = run_seed(experiment, plan, 42)
    confirmation = run_seed(experiment, plan, 137) if screening["passes"] else None
    passed = bool(confirmation and confirmation["passes"])
    return {"screening_report": "seed42_validation_report.json",
            "confirmation_report": "seed137_validation_report.json" if confirmation else None,
            "decision": "eligible_for_independent_confirmation" if passed else "retain_baseline",
            "both_seeds_pass": passed, "automatic_promotion": False, "test_inference": False,
            "limitations": plan["limitations"]}


def main() -> None:
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
        print("Operating study already complete; nothing restarted.")
        return
    import fcntl
    with (ROOT / "artifacts/recovery_pilot.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert_no_active_job()
        plan = prepare(experiment)
        if args.prepare_only:
            print(json.dumps(plan, indent=2))
            return
        if shutil.disk_usage(experiment).free < 4 * 1024**3:
            raise RuntimeError("Less than 4 GiB free for checkpoint archives; refusing launch.")
        try:
            report = run_study(experiment, plan)
            verify_plan(experiment)
            write_report(experiment / "validation_report.json", report)
        except BaseException as error:
            path = experiment / "status.json"
            status = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
            write_report(path, {**status, "status": "failed", "error": str(error)})
            raise
        write_report(experiment / "status.json", {"status": "complete", "updated_at": datetime.now(timezone.utc).isoformat(),
                     "report": str(experiment / "validation_report.json"), "automatic_promotion": False})


if __name__ == "__main__":
    main()
