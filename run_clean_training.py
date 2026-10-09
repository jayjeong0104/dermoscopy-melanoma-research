"""Train a new melanoma reference without inherited project weights or test inference."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from compare_model_runs import file_hash
from foundation_common import ROOT, save_json

DEFAULT_OUTPUT = ROOT / "artifacts/clean_melanoma_20261005"
NATURAL_LESION_OUTPUT = ROOT / "artifacts/natural_lesion_20261006"
RECIPES = ("reference", "natural_lesion")
MANIFEST = ROOT / "artifacts/pipeline_review_20261005/clean_melanoma_manifest.csv"
FOLDS = ROOT / "data/combined_ham_bcn_derm12345/grouped_folds.csv"
WEIGHTS = ROOT / ".keras/models/efficientnetv2-b1_notop.h5"
CODE = ("run_clean_training.py", "train_multitask.py", "training_provenance.py",
        "training_callback_state.py",
        "lesion_sampling.py", "targeted_sampling.py", "operating_policy.py",
        "compare_model_runs.py", "artifact_io.py", "foundation_common.py", "run_wsl_python.sh",
        "launch_clean_training.sh")


def now():
    return datetime.now(timezone.utc).isoformat()


def resolve_output(output=None, recipe="reference"):
    if recipe not in RECIPES:
        raise ValueError(f"Unsupported clean training recipe: {recipe}")
    if output is None:
        output = DEFAULT_OUTPUT if recipe == "reference" else NATURAL_LESION_OUTPUT
    output = Path(output).resolve()
    reference = DEFAULT_OUTPUT.resolve()
    if recipe == "natural_lesion" and (
        output == reference or reference in output.parents or output in reference.parents
    ):
        raise ValueError("Natural-lesion recipe must use a separate experiment directory from the reference.")
    return output


def training_arguments(output, recipe="reference"):
    if recipe not in RECIPES:
        raise ValueError(f"Unsupported clean training recipe: {recipe}")
    # This is a new reference recipe, not an ablation proving individual component gains.
    arguments = ["--manifest", str(MANIFEST), "--fold-assignments", str(FOLDS),
        "--output-root", str(output / "training"), "--test-fold", "0", "--validation-fold", "1",
        "--binary-target", "melanoma", "--backbone", "efficientnetv2b1",
        "--weights", "imagenet", "--image-size", "300", "--batch-size", "4",
        "--gradient-accumulation-steps", "4", "--head-epochs", "4", "--fine-tune-epochs", "12",
        "--fine-tune-layers", "100", "--learning-rate", "0.0003", "--fine-tune-learning-rate", "0.00001",
        "--head-normalization", "layernorm", "--training-metadata-policy", "unknown",
        "--risk-features", "image_only", "--risk-loss", "bce", "--label-smoothing", "0",
        "--diagnosis-loss-weight", "0.25", "--metadata-modality-dropout", "0",
        "--sampling-unit", "lesion", "--sampling-alpha", "0.5", "--positive-sampling-fraction", "0.25",
        "--checkpoint-metric", "lesion_pr_auc", "--minimum-operating-recall", "0.95",
        "--tta-scheme", "d4_v2", "--tta-transforms", "1", "--require-gpu",
        "--data-workers", "2", "--prefetch-batches", "1", "--seed", "42",
        "--mixed-precision", "auto", "--finalization-mode", "none", "--fit-verbose", "2"]
    if recipe == "natural_lesion":
        arguments[arguments.index("--sampling-alpha") + 1] = "1"
        arguments += ["--sampling-mass-unit", "lesion"]
    return arguments


def prepare(output, recipe="reference"):
    output = resolve_output(output, recipe)
    output.mkdir(parents=True, exist_ok=True)
    inputs = [MANIFEST, FOLDS, WEIGHTS]
    plan = {"version": 1, "recipe": recipe, "input_sha256": {str(p): file_hash(p) for p in inputs},
        "code_sha256": {name: file_hash(ROOT / name) for name in CODE},
        "runtime": {"python": sys.version, "packages": {name: importlib.metadata.version(name)
            for name in ("tensorflow", "keras", "numpy", "pandas", "scipy", "scikit-learn", "Pillow", "h5py")}},
        "arguments": training_arguments(output, recipe), "initialization": "ImageNet only; no project checkpoint",
        "purpose": "Replace a reference whose ancestors trained on its purported holdouts.",
        "test_inference": False, "automatic_promotion": False,
        "selection": "Lesion validation average precision; validation remains development data.",
        "maximum_epochs": 16, "steps_per_epoch": "ceil(training images / 4), no short-budget truncation",
        "rationale": {"layernorm": "Avoid classifier BatchNorm estimates from physical batches of four.",
            "unknown_metadata": "Match inference and remove source-dependent metadata availability.",
            "lesion_sampling": "Give lesions equal exposure within each source/diagnosis stratum.",
            "bce": "Start a simple reference without focal loss and label smoothing stacked together.",
            "positive_sampling_fraction": "Predeclared 25% exposure; less oversampling than historical 45%, not a claimed optimum."},
        "limitations": ["Historical validation and test have already been inspected; no independent final claim.",
            "HAM10000 and BCN20000 patient IDs are unavailable; lesion separation is not full patient separation.",
            "Changing several components establishes a new reference, not causal attribution.",
            "ImageNet initialization avoids identified local checkpoint leakage; higher accuracy is not guaranteed."]}
    if recipe == "natural_lesion":
        plan["purpose"] = "Controlled sampling-distribution comparison against the clean ImageNet reference; no improvement has been established."
        plan["comparison"] = {
            "reference_experiment": str(DEFAULT_OUTPUT),
            "reference_sampling": {"sampling_mass_unit": "image", "sampling_alpha": .5},
            "candidate_sampling": {"sampling_mass_unit": "lesion", "sampling_alpha": 1.0},
            "fixed_positive_sampling_fraction": .25,
            "changed_training_arguments": ["--sampling-mass-unit", "--sampling-alpha"],
            "preserved": "Same source images, labels, folds, ImageNet initialization, architecture, losses, metadata policy, learning rates, seed, 4 head/12 fine-tuning epochs, and no test finalization.",
            "interpretation": "One combined sampling-distribution intervention; it does not isolate exponent and mass-unit effects or establish gains before evaluation.",
        }
        plan["rationale"]["lesion_sampling"] = "Allocate source/diagnosis mass by independent lesion count with exponent one; sample one photo per selected lesion, preserving 25% melanoma draws."
        plan["limitations"][2] = "The comparison changes the sampling distribution as a whole, and remains a single-seed development experiment."
    path = output / "experiment_plan.json"
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError("Clean training inputs or code changed (including recipe); use a new experiment directory.")
    save_json(path, plan)
    snapshot = output / "code_snapshot"
    snapshot.mkdir(exist_ok=True)
    for name in CODE:
        target = snapshot / name
        if target.exists() and file_hash(target) != plan["code_sha256"][name]:
            raise ValueError(f"Clean training snapshot changed: {name}")
        if not target.exists():
            shutil.copy2(ROOT / name, target)
    return plan


def run(output, plan):
    complete = output / "complete.json"
    if complete.exists():
        for name, expected in json.loads(complete.read_text())["outputs_sha256"].items():
            if file_hash(output / name) != expected:
                raise ValueError("Completed clean training artifact changed.")
        print("Clean training already complete.", flush=True)
        return
    runs = sorted((output / "training").glob("run_*"))
    if len(runs) > 1:
        raise ValueError("Ambiguous clean training resume directory.")
    command = [sys.executable, str(ROOT / "train_multitask.py"), *plan["arguments"]]
    if runs:
        command += ["--resume-run-dir", str(runs[0])]
    with (output / "training.log").open("a", encoding="utf-8") as log:
        log.write(f"\nStarted {now()}\n")
        log.flush()
        child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        save_json(output / "status.json", {"status": "running", "phase": "clean_training",
            "driver_pid": os.getpid(), "pid": child.pid, "updated_at": now(), "test_inference": False})
        print(f"Clean training PID {child.pid}; log {output / 'training.log'}", flush=True)
        try:
            code = child.wait()
        except BaseException:
            child.terminate()
            child.wait()
            raise
    if code:
        raise RuntimeError(f"Clean training exited {code}; see training.log.")
    runs = sorted((output / "training").glob("run_*"))
    if len(runs) != 1 or not (runs[0] / "best_finetune_model.keras").exists():
        raise ValueError("Training returned without its expected checkpoint.")
    names = [p for p in runs[0].iterdir() if p.suffix in {".keras", ".csv", ".json"}]
    save_json(complete, {"completed_at": now(), "test_inference": False,
        "outputs_sha256": {str(p.relative_to(output)): file_hash(p) for p in names}})
    save_json(output / "status.json", {"status": "complete", "phase": "checkpoints_ready",
        "run_dir": str(runs[0]), "updated_at": now(), "test_inference": False,
        "automatic_promotion": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", choices=RECIPES, default="reference")
    parser.add_argument("--experiment-dir", type=Path,
                        help="Defaults to the recipe's isolated experiment directory.")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    output = resolve_output(args.experiment_dir, args.recipe)
    if args.prepare_only:
        prepare(output, args.recipe)
        return
    import fcntl
    from run_foundation_experiment import assert_idle
    with (ROOT / "artifacts/recovery_pilot.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert_idle()
        plan = prepare(output, args.recipe)
        try:
            run(output, plan)
        except BaseException as error:
            save_json(output / "status.json", {"status": "failed", "error": str(error), "updated_at": now()})
            raise


if __name__ == "__main__":
    main()
