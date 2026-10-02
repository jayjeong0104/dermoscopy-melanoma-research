from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import math
import os
import platform
import random
import re
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
KERAS_HOME = PROJECT_ROOT / ".keras"
KERAS_HOME.mkdir(exist_ok=True)
os.environ.setdefault("KERAS_HOME", str(KERAS_HOME))
MATPLOTLIB_CONFIG = Path(tempfile.gettempdir()) / "skin_cancer_multitask_mplconfig"
MATPLOTLIB_CONFIG.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CONFIG))

import matplotlib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


matplotlib.use("Agg")
import matplotlib.pyplot as plt


DIAGNOSES = ("AK", "BCC", "BKL", "DF", "MEL", "NV", "SCC", "VASC")
DIAGNOSIS_TO_INDEX = {name: index for index, name in enumerate(DIAGNOSES)}
UNSUPERVISED_DIAGNOSIS = "UNKNOWN"
HIGH_RISK_DIAGNOSES = frozenset({"AK", "BCC", "MEL", "SCC"})
MALIGNANT_DIAGNOSES = frozenset({"BCC", "MEL", "SCC"})
SITE_CATEGORIES = (
    "head_neck",
    "torso",
    "upper_extremity",
    "lower_extremity",
    "acral",
    "genital",
    "unknown",
)
SEX_CATEGORIES = ("female", "male", "unknown")
REQUIRED_COLUMNS = {
    "source",
    "image_id",
    "lesion_id",
    "image_path",
    "diagnosis",
    "target",
    "sha256",
    "age_approx",
    "sex",
    "anatom_site_general",
}
WINDOWS_PATH = re.compile(r"^([A-Za-z]):[\\/](.*)$")


@dataclass(frozen=True)
class SplitFrames:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


@dataclass(frozen=True)
class MetadataEncoder:
    age_median: float
    feature_names: tuple[str, ...]

    @classmethod
    def fit(cls, frame: pd.DataFrame) -> "MetadataEncoder":
        ages = pd.to_numeric(frame["age_approx"], errors="coerce")
        median = float(ages.median()) if ages.notna().any() else 50.0
        feature_names = (
            "age_scaled",
            "age_missing",
            *(f"sex_{name}" for name in SEX_CATEGORIES),
            *(f"site_{name}" for name in SITE_CATEGORIES),
        )
        return cls(age_median=median, feature_names=tuple(feature_names))

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        ages = pd.to_numeric(frame["age_approx"], errors="coerce")
        age_missing = ages.isna().astype("float32").to_numpy()
        age_scaled = ages.fillna(self.age_median).clip(0, 100).astype("float32").to_numpy() / 100.0

        sexes = frame["sex"].map(normalize_sex)
        sites = frame["anatom_site_general"].map(normalize_site)
        columns: list[np.ndarray] = [age_scaled, age_missing]
        columns.extend((sexes == name).astype("float32").to_numpy() for name in SEX_CATEGORIES)
        columns.extend((sites == name).astype("float32").to_numpy() for name in SITE_CATEGORIES)
        matrix = np.column_stack(columns).astype("float32")
        if matrix.shape[1] != len(self.feature_names):
            raise RuntimeError("Metadata feature count does not match the saved schema.")
        return matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a source-balanced image-and-metadata skin-lesion multitask model."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data") / "combined_ham_bcn" / "combined_manifest.csv",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts") / "multitask",
    )
    parser.add_argument(
        "--resume-run-dir",
        type=Path,
        help="Existing interrupted run directory to resume without repeating completed epochs.",
    )
    parser.add_argument(
        "--backbone",
        choices=("efficientnetv2b0", "efficientnetv2b1", "efficientnetv2s"),
        default="efficientnetv2b1",
    )
    parser.add_argument("--image-size", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--steps-per-epoch", type=int,
        help="Fixed batch budget for controlled experiments; default uses training row count.",
    )
    parser.add_argument("--head-epochs", type=int, default=6)
    parser.add_argument("--fine-tune-epochs", type=int, default=12)
    parser.add_argument("--fine-tune-layers", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--fine-tune-learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--dropout-rate", type=float, default=0.35)
    parser.add_argument(
        "--metadata-modality-dropout",
        type=float,
        default=0.50,
        help="Probability of dropping the entire metadata branch for a training sample.",
    )
    parser.add_argument(
        "--risk-features",
        choices=("image_only", "fused"),
        default="image_only",
        help="Features used by the primary binary head. Image-only is safer across datasets.",
    )
    parser.add_argument("--risk-loss", choices=("bce", "focal"), default="focal")
    parser.add_argument("--focal-gamma", type=float, default=1.5)
    parser.add_argument(
        "--binary-target",
        choices=("high_risk", "malignant", "melanoma"),
        default="high_risk",
        help=(
            "Binary objective. Malignant uses BCC/MEL/SCC; melanoma treats every "
            "non-MEL diagnosis as negative."
        ),
    )
    parser.add_argument("--label-smoothing", type=float, default=0.02)
    parser.add_argument("--diagnosis-loss-weight", type=float, default=0.60)
    parser.add_argument(
        "--sampling-alpha",
        type=float,
        default=0.50,
        help="Source/diagnosis sampling exponent: 1 is natural, 0 is uniform.",
    )
    parser.add_argument(
        "--positive-sampling-fraction",
        type=float,
        help=(
            "Optional positive-class probability for training batches. Validation and test "
            "always retain their natural prevalence."
        ),
    )
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--sampling-unit", choices=("image", "lesion"), default="image",
                        help="Exposure unit within each source/diagnosis stratum; preserves source/class mass.")
    parser.add_argument("--hard-negative-weights", type=Path,
                        help="Exact training-only identity table with capped sampling weights.")
    parser.add_argument("--checkpoint-metric", choices=("image_pr_auc", "lesion_pr_auc", "lesion_operating"),
                        default="image_pr_auc", help="Legacy default preserves old resume behavior.")
    parser.add_argument("--freeze-batch-norm", action="store_true",
                        help="Freeze all BatchNorm layers, including classifier heads, for small-batch warm starts.")
    parser.add_argument("--archive-epoch-checkpoints", action="store_true",
                        help="Keep each epoch separately so AP selection cannot discard operating-point tradeoffs.")
    parser.add_argument("--tta-scheme", choices=("legacy_v1", "d4_v2"), default="legacy_v1")
    parser.add_argument("--require-gpu", action="store_true",
                        help="Fail instead of silently falling back to CPU training.")
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--fold-assignments", type=Path)
    parser.add_argument("--test-fold", type=int)
    parser.add_argument("--validation-fold", type=int)
    parser.add_argument("--sample-fraction", type=float, default=1.0)
    parser.add_argument("--minimum-operating-recall", type=float, default=0.80)
    parser.add_argument(
        "--tta-transforms",
        type=int,
        choices=(1, 4, 8),
        default=1,
        help="Number of deterministic dihedral views averaged during final inference.",
    )
    parser.add_argument("--data-workers", type=int, default=2)
    parser.add_argument("--prefetch-batches", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fit-verbose", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument(
        "--finalization-mode",
        choices=("fresh_process", "in_process", "none"),
        default="fresh_process",
        help=(
            "Run checkpoint selection and held-out evaluation in a fresh process by "
            "default so training allocator memory is fully released."
        ),
    )
    parser.add_argument(
        "--evaluation-batch-size",
        type=int,
        default=2,
        help="Inference batch size used by fresh-process finalization.",
    )
    parser.add_argument(
        "--mixed-precision",
        choices=("off", "on", "auto"),
        default="auto",
    )
    parser.add_argument(
        "--weights",
        choices=("imagenet", "none"),
        default="imagenet",
    )
    parser.add_argument(
        "--warm-start-model",
        type=Path,
        help="Optional prior Keras model whose compatible layers initialize this run.",
    )
    parser.add_argument(
        "--reset-risk-head",
        action="store_true",
        help="Do not copy risk_dense/high_risk weights from a warm-start model.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")
    if args.image_size < 96:
        raise ValueError("--image-size must be at least 96.")
    if (
        args.batch_size < 1
        or args.evaluation_batch_size < 1
        or args.head_epochs < 1
        or args.fine_tune_epochs < 0
    ):
        raise ValueError("Batch size and head epochs must be positive; fine-tune epochs cannot be negative.")
    if args.fine_tune_layers < 1:
        raise ValueError("--fine-tune-layers must be at least 1.")
    if args.steps_per_epoch is not None and args.steps_per_epoch < 1:
        raise ValueError("--steps-per-epoch must be positive.")
    if args.hard_negative_weights is not None and not args.hard_negative_weights.is_file():
        raise FileNotFoundError(f"Sampling weights not found: {args.hard_negative_weights}")
    if args.freeze_batch_norm and args.warm_start_model is None:
        raise ValueError("--freeze-batch-norm requires a trained --warm-start-model.")
    if args.checkpoint_metric == "lesion_operating" and args.finalization_mode != "none":
        raise ValueError("lesion_operating requires validation-only finalization-mode none; use the operating-policy evaluator.")
    if args.sampling_unit == "lesion" and args.hard_negative_weights is not None:
        raise ValueError("Lesion sampling cannot be combined with hard-negative weights in this protocol.")
    if args.learning_rate <= 0 or args.fine_tune_learning_rate <= 0:
        raise ValueError("Learning rates must be positive.")
    if not 0 <= args.label_smoothing < 0.5:
        raise ValueError("--label-smoothing must be in [0, 0.5).")
    if not 0 <= args.metadata_modality_dropout < 1:
        raise ValueError("--metadata-modality-dropout must be in [0, 1).")
    if args.focal_gamma <= 0:
        raise ValueError("--focal-gamma must be positive.")
    if not 0 < args.diagnosis_loss_weight <= 2:
        raise ValueError("--diagnosis-loss-weight must be in (0, 2].")
    if not 0 <= args.sampling_alpha <= 1:
        raise ValueError("--sampling-alpha must be in [0, 1].")
    if args.positive_sampling_fraction is not None and not (
        0 < args.positive_sampling_fraction < 1
    ):
        raise ValueError("--positive-sampling-fraction must be in (0, 1).")
    if not 0 < args.val_size < 1 or not 0 < args.test_size < 1:
        raise ValueError("Validation and test fractions must be between 0 and 1.")
    if args.val_size + args.test_size >= 1:
        raise ValueError("Validation and test fractions must sum to less than 1.")
    fold_values = (args.test_fold, args.validation_fold)
    if args.fold_assignments is None and any(value is not None for value in fold_values):
        raise ValueError("--test-fold/--validation-fold require --fold-assignments.")
    if args.fold_assignments is not None:
        if not args.fold_assignments.exists():
            raise FileNotFoundError(f"Fold assignments not found: {args.fold_assignments}")
        if any(value is None or value < 0 for value in fold_values):
            raise ValueError("Fold runs require non-negative --test-fold and --validation-fold.")
        if args.test_fold == args.validation_fold:
            raise ValueError("Test and validation folds must differ.")
    if not 0 < args.sample_fraction <= 1:
        raise ValueError("--sample-fraction must be in (0, 1].")
    if not 0 < args.minimum_operating_recall <= 1:
        raise ValueError("--minimum-operating-recall must be in (0, 1].")
    if args.data_workers < 1:
        raise ValueError("--data-workers must be at least 1.")
    if args.prefetch_batches < 1:
        raise ValueError("--prefetch-batches must be at least 1.")
    if args.gradient_accumulation_steps < 1:
        raise ValueError("--gradient-accumulation-steps must be at least 1.")
    if args.warm_start_model is not None and not args.warm_start_model.exists():
        raise FileNotFoundError(f"Warm-start model not found: {args.warm_start_model}")
    if args.resume_run_dir is not None:
        if not args.resume_run_dir.is_dir():
            raise FileNotFoundError(f"Resume run directory not found: {args.resume_run_dir}")
        if args.resume_run_dir.resolve().parent != args.output_root.resolve():
            raise ValueError("--resume-run-dir must be a direct child of --output-root.")


def configure_runtime(mode: str) -> dict[str, object]:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

    enable_mixed = mode == "on" or (mode == "auto" and bool(gpus))
    tf.keras.mixed_precision.set_global_policy("mixed_float16" if enable_mixed else "float32")
    return {
        "platform": platform.platform(),
        "tensorflow_version": tf.__version__,
        "physical_devices": [str(device) for device in tf.config.list_physical_devices()],
        "gpu_devices": [str(device) for device in gpus],
        "mixed_precision_policy": tf.keras.mixed_precision.global_policy().name,
    }


def set_random_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def runtime_path(path_text: str) -> str:
    if os.name == "nt":
        return path_text
    match = WINDOWS_PATH.match(path_text)
    if not match:
        return path_text
    drive, remainder = match.groups()
    return f"/mnt/{drive.lower()}/{remainder.replace(chr(92), '/')}"


def resize_and_normalize_image(image: tf.Tensor, image_size: int) -> tf.Tensor:
    """Resize compact decoded pixels before creating the normalized float tensor."""
    image = tf.image.resize(image, [image_size, image_size], antialias=True)
    return tf.cast(image, tf.float32) / 255.0


def load_manifest(path: Path, binary_target: str | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")

    frame = frame.copy()
    if "split_group_id" not in frame:
        frame["split_group_id"] = frame["lesion_id"]
    frame["diagnosis"] = frame["diagnosis"].astype(str).str.upper()
    if "diagnosis_label_available" not in frame:
        frame["diagnosis_label_available"] = 1
    frame["diagnosis_label_available"] = pd.to_numeric(
        frame["diagnosis_label_available"], errors="raise"
    ).astype("int32")
    if not frame["diagnosis_label_available"].isin([0, 1]).all():
        raise ValueError("diagnosis_label_available must contain only 0 or 1.")

    supervised = frame["diagnosis_label_available"].eq(1)
    unknown = sorted(set(frame.loc[supervised, "diagnosis"]) - set(DIAGNOSES))
    if unknown:
        raise ValueError(f"Manifest contains unsupported diagnoses: {unknown}")
    unsupported_unsupervised = frame.loc[
        ~supervised & ~frame["diagnosis"].eq(UNSUPERVISED_DIAGNOSIS)
    ]
    if not unsupported_unsupervised.empty:
        raise ValueError(
            f"Rows without diagnosis supervision must use {UNSUPERVISED_DIAGNOSIS}."
        )
    frame["diagnosis_index"] = (
        frame["diagnosis"].map(DIAGNOSIS_TO_INDEX).fillna(0).astype("int32")
    )
    frame["target"] = frame["target"].astype("int32")
    if binary_target is not None:
        if binary_target == "high_risk":
            expected_target = frame["diagnosis"].isin(HIGH_RISK_DIAGNOSES).astype("int32")
        elif binary_target == "malignant":
            expected_target = frame["diagnosis"].isin(MALIGNANT_DIAGNOSES).astype("int32")
        elif binary_target == "melanoma":
            expected_target = frame["diagnosis"].eq("MEL").astype("int32")
        else:
            raise ValueError(f"Unsupported binary target: {binary_target}")
        if not np.array_equal(frame["target"].to_numpy(), expected_target.to_numpy()):
            raise ValueError(
                "Manifest binary targets disagree with the requested diagnosis-to-target mapping."
            )

    diagnosis_conflicts = frame.groupby("lesion_id")["diagnosis"].nunique()
    if (diagnosis_conflicts > 1).any():
        raise ValueError("At least one lesion has conflicting diagnosis labels.")

    frame["runtime_image_path"] = frame["image_path"].astype(str).map(runtime_path)
    missing_paths = [value for value in frame["runtime_image_path"] if not Path(value).exists()]
    if missing_paths:
        raise FileNotFoundError(
            f"Manifest references {len(missing_paths)} missing runtime paths. Sample: {missing_paths[:3]}"
        )
    return frame


def maybe_downsample(frame: pd.DataFrame, fraction: float, seed: int) -> pd.DataFrame:
    if fraction >= 1.0:
        return frame
    lesion_level = frame[["lesion_id", "source", "diagnosis"]].drop_duplicates()
    sampled = []
    for index, (_, group) in enumerate(lesion_level.groupby(["source", "diagnosis"])):
        take = min(len(group), max(10, int(round(len(group) * fraction))))
        sampled.append(group.sample(n=take, random_state=seed + index))
    selected = pd.concat(sampled, ignore_index=True)["lesion_id"]
    return frame[frame["lesion_id"].isin(selected)].copy()


def configure_binary_target(frame: pd.DataFrame, target_name: str) -> pd.DataFrame:
    configured = frame.copy()
    supervised = configured.get(
        "diagnosis_label_available",
        pd.Series(1, index=configured.index, dtype="int32"),
    ).eq(1)
    if (~supervised).any() and target_name != "melanoma":
        raise ValueError(
            "Rows without diagnosis labels are supported only for melanoma-vs-rest training."
        )
    if target_name == "high_risk":
        configured["target_name"] = np.where(configured["target"] == 1, "high_risk", "benign")
    elif target_name == "malignant":
        configured["target"] = configured["diagnosis"].isin(MALIGNANT_DIAGNOSES).astype("int32")
        configured["target_name"] = np.where(
            configured["target"] == 1, "malignant", "benign"
        )
    elif target_name == "melanoma":
        configured.loc[supervised, "target"] = (
            configured.loc[supervised, "diagnosis"].eq("MEL").astype("int32")
        )
        if not configured.loc[~supervised, "target"].eq(0).all():
            raise ValueError("Unsupervised diagnosis rows must be confirmed non-melanoma.")
        configured["target_name"] = np.where(
            configured["target"] == 1, "melanoma", "not_melanoma"
        )
    else:
        raise ValueError(f"Unsupported binary target: {target_name}")
    return configured


def binary_class_names(target_name: str) -> list[str]:
    if target_name == "malignant":
        return ["benign", "malignant"]
    if target_name == "melanoma":
        return ["not_melanoma", "melanoma"]
    return ["benign", "high_risk"]


def assert_split_integrity(splits: SplitFrames) -> None:
    lesion_sets = [set(split["lesion_id"]) for split in (splits.train, splits.val, splits.test)]
    group_sets = [
        set(split["split_group_id"]) for split in (splits.train, splits.val, splits.test)
    ]
    hash_sets = [set(split["sha256"]) for split in (splits.train, splits.val, splits.test)]
    for left in range(3):
        for right in range(left + 1, 3):
            if lesion_sets[left] & lesion_sets[right]:
                raise RuntimeError("Lesion leakage detected after splitting.")
            if group_sets[left] & group_sets[right]:
                raise RuntimeError("Patient/split-group leakage detected after splitting.")
            if hash_sets[left] & hash_sets[right]:
                raise RuntimeError("Exact image leakage detected after splitting.")


def split_by_lesion_source_and_diagnosis(
    frame: pd.DataFrame,
    val_size: float,
    test_size: float,
    seed: int,
) -> SplitFrames:
    group_level = frame[
        ["split_group_id", "source", "diagnosis", "target"]
    ].drop_duplicates()
    group_conflicts = group_level.groupby("split_group_id")[["source", "diagnosis", "target"]].nunique()
    if (group_conflicts > 1).any().any():
        raise ValueError(
            "Random splitting requires one diagnosis per split group; use predefined grouped "
            "folds for multi-lesion patients."
        )
    lesion_level = group_level.rename(columns={"split_group_id": "lesion_id"})
    lesion_level["stratum"] = lesion_level["source"] + "::" + lesion_level["diagnosis"]
    counts = lesion_level["stratum"].value_counts()
    if counts.min() < 7:
        raise ValueError(
            "Each source/diagnosis stratum needs at least 7 lesions for a three-way split. "
            f"Smallest strata: {counts.nsmallest(5).to_dict()}"
        )

    train_lesions, temporary = train_test_split(
        lesion_level,
        test_size=val_size + test_size,
        stratify=lesion_level["stratum"],
        random_state=seed,
    )
    val_lesions, test_lesions = train_test_split(
        temporary,
        test_size=test_size / (val_size + test_size),
        stratify=temporary["stratum"],
        random_state=seed,
    )
    result = SplitFrames(
        train=frame[frame["split_group_id"].isin(train_lesions["lesion_id"])].copy(),
        val=frame[frame["split_group_id"].isin(val_lesions["lesion_id"])].copy(),
        test=frame[frame["split_group_id"].isin(test_lesions["lesion_id"])].copy(),
    )
    assert_split_integrity(result)
    return result


def split_from_fold_assignments(
    frame: pd.DataFrame,
    assignments_path: Path,
    test_fold: int,
    validation_fold: int,
) -> SplitFrames:
    assignments = pd.read_csv(assignments_path)
    assignment_key = "split_group_id" if "split_group_id" in assignments else "lesion_id"
    required = {assignment_key, "fold"}
    if not required.issubset(assignments.columns):
        raise ValueError(f"Fold assignments must contain columns: {sorted(required)}")
    if assignments[assignment_key].duplicated().any():
        raise ValueError("Fold assignments contain duplicate split-group IDs.")

    fold_table = assignments[[assignment_key, "fold"]].rename(
        columns={assignment_key: "split_group_id"}
    )
    merged = frame.merge(
        fold_table,
        on="split_group_id",
        how="left",
        validate="many_to_one",
    )
    if merged["fold"].isna().any():
        missing = merged.loc[merged["fold"].isna(), "split_group_id"].drop_duplicates().head(5)
        raise ValueError(f"Fold assignments are missing split-group IDs: {missing.tolist()}")
    merged["fold"] = merged["fold"].astype("int32")
    available = set(merged["fold"].unique())
    if test_fold not in available or validation_fold not in available:
        raise ValueError(f"Requested folds are unavailable; found {sorted(available)}")

    result = SplitFrames(
        train=merged[~merged["fold"].isin([test_fold, validation_fold])].copy(),
        val=merged[merged["fold"] == validation_fold].copy(),
        test=merged[merged["fold"] == test_fold].copy(),
    )
    for split in (result.train, result.val, result.test):
        split.drop(columns=["fold"], inplace=True)
    if min(len(result.train), len(result.val), len(result.test)) == 0:
        raise ValueError("A predefined fold produced an empty split.")
    assert_split_integrity(result)
    return result


def normalize_sex(value: object) -> str:
    if pd.isna(value):
        return "unknown"
    normalized = str(value).strip().lower()
    return normalized if normalized in {"female", "male"} else "unknown"


def normalize_site(value: object) -> str:
    if pd.isna(value):
        return "unknown"
    normalized = re.sub(r"[^a-z]+", " ", str(value).strip().lower()).strip()
    if not normalized or normalized in {"unknown", "nan"}:
        return "unknown"
    if any(token in normalized for token in ("scalp", "face", "ear", "neck", "head")):
        return "head_neck"
    if any(token in normalized for token in ("back", "chest", "abdomen", "torso", "trunk")):
        return "torso"
    if any(token in normalized for token in ("palm", "sole", "acral")):
        return "acral"
    if any(token in normalized for token in ("hand", "upper extremity", "arm")):
        return "upper_extremity"
    if any(token in normalized for token in ("foot", "lower extremity", "leg")):
        return "lower_extremity"
    if "genital" in normalized:
        return "genital"
    return "unknown"


def _tensor_slices(
    frame: pd.DataFrame, encoder: MetadataEncoder, mask_metadata: bool = False
) -> tuple[np.ndarray, ...]:
    metadata_frame = frame
    if mask_metadata:
        metadata_frame = frame.copy()
        metadata_frame["age_approx"] = pd.NA
        metadata_frame["sex"] = "unknown"
        metadata_frame["anatom_site_general"] = "unknown"
    return (
        frame["runtime_image_path"].astype(str).to_numpy(),
        encoder.transform(metadata_frame),
        frame["target"].astype("float32").to_numpy(),
        frame["diagnosis_index"].astype("int32").to_numpy(),
        frame.get(
            "diagnosis_label_available",
            pd.Series(1, index=frame.index, dtype="int32"),
        ).astype("float32").to_numpy(),
    )


def build_dataset(
    frame: pd.DataFrame,
    encoder: MetadataEncoder,
    image_size: int,
    batch_size: int,
    training: bool,
    data_workers: int,
    seed: int,
    sampling_alpha: float = 1.0,
    positive_sampling_fraction: float | None = None,
    prefetch_batches: int = 1,
    mask_metadata: bool = False,
    sampling_unit: str = "image",
) -> tf.data.Dataset:
    if sampling_unit not in {"image", "lesion"}:
        raise ValueError("Unsupported sampling unit.")
    if training and sampling_unit == "lesion" and "sampling_weight" in frame:
        raise ValueError("Lesion sampling cannot silently ignore hard-negative weights.")
    if training:
        datasets: list[tf.data.Dataset] = []
        weights: list[float] = []
        group_targets: list[int] = []
        grouped = frame.groupby(["source", "diagnosis"], sort=True)
        for index, (_, group) in enumerate(grouped):
            if group.target.nunique() != 1:
                raise ValueError("A source/diagnosis stratum has conflicting binary targets.")
            if sampling_unit == "lesion":
                from lesion_sampling import row_dataset
                tensors = tuple(tf.convert_to_tensor(value) for value in _tensor_slices(group, encoder, mask_metadata))

                def gather_row(row, tensors=tensors):
                    return tuple(tf.gather(value, row) for value in tensors)

                datasets.append(row_dataset(group, seed + index).map(gather_row))
                weights.append(float(len(group) ** sampling_alpha))
                group_targets.append(int(group.target.iloc[0]))
                continue
            if "sampling_weight" in group:
                values = group.sampling_weight.to_numpy(dtype=float)
                if not np.isfinite(values).all() or not np.isin(values, [1.0, 2.0]).all():
                    raise ValueError("Training sampling weights must be 1 or 2.")
                partitions = list(group.groupby("sampling_weight", sort=True))
            else:
                partitions = [(1.0, group)]
            inner_datasets, inner_weights = [], []
            for part_index, (weight, part) in enumerate(partitions):
                subset = tf.data.Dataset.from_tensor_slices(_tensor_slices(part, encoder, mask_metadata))
                inner_datasets.append(subset.shuffle(
                    max(16, len(part)), seed=seed + index + 1000 * part_index,
                    reshuffle_each_iteration=True,
                ).repeat())
                inner_weights.append(float(weight * len(part)))
            # Keep the original source/diagnosis mass; change only exposure within it.
            dataset = inner_datasets[0] if len(inner_datasets) == 1 else tf.data.Dataset.sample_from_datasets(
                inner_datasets, weights=np.asarray(inner_weights) / sum(inner_weights),
                seed=seed + index, stop_on_empty_dataset=False,
            )
            datasets.append(dataset)
            weights.append(float(len(group) ** sampling_alpha))
            group_targets.append(int(group["target"].iloc[0]))
        if positive_sampling_fraction is None:
            total = sum(weights)
            normalized_weights = [weight / total for weight in weights]
        else:
            class_totals = {
                target: sum(
                    weight
                    for weight, group_target in zip(weights, group_targets)
                    if group_target == target
                )
                for target in (0, 1)
            }
            if min(class_totals.values()) <= 0:
                raise ValueError("Positive-class sampling requires both binary classes.")
            class_fractions = {0: 1.0 - positive_sampling_fraction, 1: positive_sampling_fraction}
            normalized_weights = [
                class_fractions[target] * weight / class_totals[target]
                for weight, target in zip(weights, group_targets)
            ]
        dataset = tf.data.Dataset.sample_from_datasets(
            datasets,
            weights=normalized_weights,
            seed=seed,
            stop_on_empty_dataset=False,
        )
    else:
        dataset = tf.data.Dataset.from_tensor_slices(
            _tensor_slices(frame, encoder, mask_metadata)
        )

    def decode(
        path: tf.Tensor,
        metadata: tf.Tensor,
        risk: tf.Tensor,
        diagnosis: tf.Tensor,
        diagnosis_weight: tf.Tensor,
    ) -> tuple[
        dict[str, tf.Tensor],
        dict[str, tf.Tensor],
        dict[str, tf.Tensor],
    ]:
        image = tf.io.read_file(path)
        image = tf.io.decode_jpeg(image, channels=3)
        image = resize_and_normalize_image(image, image_size)
        image.set_shape([image_size, image_size, 3])
        metadata.set_shape([len(encoder.feature_names)])
        labels = {
            "high_risk": tf.reshape(risk, [1]),
            "diagnosis": tf.one_hot(diagnosis, depth=len(DIAGNOSES), dtype=tf.float32),
        }
        sample_weights = {
            "high_risk": tf.constant(1.0, dtype=tf.float32),
            "diagnosis": tf.cast(diagnosis_weight, tf.float32),
        }
        return {"image": image, "metadata": metadata}, labels, sample_weights

    options = tf.data.Options()
    options.deterministic = not training
    options.autotune.enabled = False
    options.threading.private_threadpool_size = data_workers
    options.threading.max_intra_op_parallelism = 1
    dataset = dataset.with_options(options)
    dataset = dataset.map(decode, num_parallel_calls=data_workers)
    dataset = dataset.batch(batch_size)
    return dataset.prefetch(prefetch_batches)


def backbone_builder(name: str):
    return {
        "efficientnetv2b0": tf.keras.applications.EfficientNetV2B0,
        "efficientnetv2b1": tf.keras.applications.EfficientNetV2B1,
        "efficientnetv2s": tf.keras.applications.EfficientNetV2S,
    }[name]


def build_model(
    image_size: int,
    metadata_size: int,
    dropout_rate: float,
    metadata_modality_dropout: float,
    risk_features: str,
    backbone_name: str,
    weights: str | None,
) -> tf.keras.Model:
    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip(
                "horizontal_and_vertical", dtype="float32"
            ),
            tf.keras.layers.RandomRotation(
                0.20, fill_mode="reflect", dtype="float32"
            ),
            tf.keras.layers.RandomTranslation(
                0.06, 0.06, fill_mode="reflect", dtype="float32"
            ),
            tf.keras.layers.RandomZoom(
                0.15, fill_mode="reflect", dtype="float32"
            ),
            tf.keras.layers.RandomContrast(0.18, dtype="float32"),
            tf.keras.layers.RandomBrightness(
                0.10, value_range=(0.0, 1.0), dtype="float32"
            ),
            tf.keras.layers.RandomHue(
                0.04, value_range=(0.0, 1.0), dtype="float32"
            ),
            tf.keras.layers.RandomSaturation(
                (0.40, 0.60), value_range=(0.0, 1.0), dtype="float32"
            ),
            tf.keras.layers.RandomErasing(
                factor=0.15,
                scale=(0.02, 0.12),
                value_range=(0.0, 1.0),
                dtype="float32",
            ),
            tf.keras.layers.GaussianNoise(0.01, dtype="float32"),
        ],
        name="augmentation",
    )
    application_model = backbone_builder(backbone_name)(
        include_top=False,
        weights=weights,
        input_shape=(image_size, image_size, 3),
        include_preprocessing=True,
    )
    backbone = tf.keras.Model(
        application_model.input, application_model.output, name="backbone"
    )
    backbone.trainable = False

    image_input = tf.keras.Input((image_size, image_size, 3), name="image")
    metadata_input = tf.keras.Input((metadata_size,), name="metadata")

    image_features = augmentation(image_input)
    image_features = tf.keras.layers.Rescaling(255.0, name="backbone_input_scale")(image_features)
    image_features = backbone(image_features, training=False)
    image_features = tf.keras.layers.GlobalAveragePooling2D(name="image_pool")(image_features)
    image_features = tf.keras.layers.BatchNormalization(name="image_batch_norm")(image_features)
    image_features = tf.keras.layers.Dropout(dropout_rate, name="image_dropout")(image_features)
    image_features = tf.keras.layers.Dense(384, activation="swish", name="image_projection")(
        image_features
    )

    metadata_features = tf.keras.layers.BatchNormalization(name="metadata_batch_norm")(
        metadata_input
    )
    metadata_features = tf.keras.layers.Dense(64, activation="swish", name="metadata_dense")(
        metadata_features
    )
    metadata_features = tf.keras.layers.Dropout(dropout_rate * 0.5, name="metadata_dropout")(
        metadata_features
    )
    metadata_features = tf.keras.layers.Dropout(
        metadata_modality_dropout,
        noise_shape=(None, 1),
        name="metadata_modality_dropout",
    )(metadata_features)

    shared = tf.keras.layers.Concatenate(name="feature_fusion")(
        [image_features, metadata_features]
    )
    shared = tf.keras.layers.Dense(
        256,
        activation="swish",
        kernel_regularizer=tf.keras.regularizers.l2(1e-5),
        name="shared_dense",
    )(shared)
    shared = tf.keras.layers.Dropout(dropout_rate, name="shared_dropout")(shared)

    diagnosis = tf.keras.layers.Dense(
        len(DIAGNOSES), activation="softmax", dtype="float32", name="diagnosis"
    )(shared)

    # Keep the primary binary decision independent of the auxiliary diagnosis probabilities.
    # Image-only risk features prevent missing metadata from becoming a dataset shortcut.
    if risk_features == "image_only":
        risk_input = image_features
        risk_input = tf.keras.layers.Dense(
            256,
            activation="swish",
            kernel_regularizer=tf.keras.regularizers.l2(1e-5),
            name="risk_dense",
        )(risk_input)
        risk_input = tf.keras.layers.Dropout(dropout_rate, name="risk_dropout")(risk_input)
    elif risk_features == "fused":
        risk_input = shared
    else:
        raise ValueError(f"Unsupported risk feature mode: {risk_features}")
    high_risk = tf.keras.layers.Dense(
        1, activation="sigmoid", dtype="float32", name="high_risk"
    )(risk_input)

    return tf.keras.Model(
        inputs={"image": image_input, "metadata": metadata_input},
        outputs={"high_risk": high_risk, "diagnosis": diagnosis},
        name=f"skin_lesion_multitask_{backbone_name}",
    )


def freeze_batch_norm_layers(model: tf.keras.Model) -> list[str]:
    frozen = []
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
            frozen.append(layer.name)
        elif isinstance(layer, tf.keras.Model):
            frozen.extend(f"{layer.name}/{name}" for name in freeze_batch_norm_layers(layer))
    return frozen


def compile_model(
    model: tf.keras.Model,
    learning_rate: float,
    weight_decay: float,
    label_smoothing: float,
    diagnosis_loss_weight: float,
    risk_loss_name: str,
    focal_gamma: float,
    gradient_accumulation_steps: int = 1,
    freeze_batch_norm: bool = False,
) -> None:
    if freeze_batch_norm:
        print(f"Frozen BatchNorm layers: {len(freeze_batch_norm_layers(model))}", flush=True)
    if risk_loss_name == "focal":
        risk_loss = tf.keras.losses.BinaryFocalCrossentropy(
            gamma=focal_gamma,
            label_smoothing=label_smoothing,
        )
    elif risk_loss_name == "bce":
        risk_loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=label_smoothing)
    else:
        raise ValueError(f"Unsupported risk loss: {risk_loss_name}")
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            global_clipnorm=1.0,
            gradient_accumulation_steps=(
                None if gradient_accumulation_steps == 1 else gradient_accumulation_steps
            ),
        ),
        loss={
            "high_risk": risk_loss,
            "diagnosis": tf.keras.losses.CategoricalCrossentropy(
                label_smoothing=label_smoothing
            ),
        },
        loss_weights={"high_risk": 1.0, "diagnosis": diagnosis_loss_weight},
        metrics={
            "high_risk": [
                tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                tf.keras.metrics.AUC(name="auc", curve="ROC"),
                tf.keras.metrics.AUC(name="pr_auc", curve="PR"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
            ]
        },
        weighted_metrics={
            "diagnosis": [
                tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
                tf.keras.metrics.TopKCategoricalAccuracy(k=2, name="top2_accuracy"),
            ],
        },
    )


def warm_start_compatible_layers(
    model: tf.keras.Model,
    source_path: Path,
    risk_features: str,
    reset_risk_head: bool = False,
) -> dict[str, object]:
    source_model = tf.keras.models.load_model(source_path, compile=False)
    copied: list[dict[str, str]] = []
    skipped: dict[str, str] = {}
    for target_layer in model.layers:
        if not target_layer.weights:
            continue
        if reset_risk_head and target_layer.name in {"risk_dense", "high_risk"}:
            skipped[target_layer.name] = "reset requested"
            continue
        source_names = [target_layer.name]
        if target_layer.name == "high_risk" and risk_features == "fused":
            source_names.append("direct_risk")
        copied_layer = False
        mismatch_found = False
        for source_name in source_names:
            try:
                source_layer = source_model.get_layer(source_name)
            except ValueError:
                continue
            source_weights = source_layer.get_weights()
            target_weights = target_layer.get_weights()
            if len(source_weights) != len(target_weights) or any(
                source.shape != target.shape
                for source, target in zip(source_weights, target_weights)
            ):
                mismatch_found = True
                continue
            target_layer.set_weights(source_weights)
            copied.append({"target": target_layer.name, "source": source_name})
            copied_layer = True
            break
        if not copied_layer:
            skipped[target_layer.name] = (
                "incompatible weight shapes" if mismatch_found else "missing in source model"
            )
    del source_model
    return {
        "source_model": str(source_path.resolve()),
        "copied_layers": copied,
        "skipped_layers": skipped,
    }


def make_backbone_trainable(model: tf.keras.Model, trainable_layers: int) -> int:
    backbone = model.get_layer("backbone")
    backbone.trainable = True
    for layer in backbone.layers:
        layer.trainable = False
    selected = backbone.layers[-min(trainable_layers, len(backbone.layers)) :]
    trainable_count = 0
    for layer in selected:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            continue
        layer.trainable = True
        trainable_count += 1
    return trainable_count


def trim_host_memory() -> None:
    if os.name != "posix":
        return
    try:
        libc = ctypes.CDLL(None)
        malloc_trim = libc.malloc_trim
        malloc_trim.argtypes = [ctypes.c_size_t]
        malloc_trim.restype = ctypes.c_int
        malloc_trim(0)
    except (AttributeError, OSError):
        pass


class HostMemoryCleanup(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch: int, logs: dict[str, float] | None = None) -> None:
        del logs
        gc.collect()
        trim_host_memory()

        rss_mb = None
        status_path = Path("/proc/self/status")
        if status_path.exists():
            for line in status_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("VmRSS:"):
                    rss_mb = int(line.split()[1]) / 1024.0
                    break
        if rss_mb is not None:
            print(f"Host memory after epoch {epoch + 1}: {rss_mb:.1f} MiB RSS")


def stage_history(run_dir: Path, stage: str) -> pd.DataFrame:
    path = run_dir / f"training_{stage}.csv"
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def stage_resume_epoch(run_dir: Path, stage: str) -> int:
    history = stage_history(run_dir, stage)
    history_epoch = 0
    if not history.empty:
        if "epoch" not in history:
            raise ValueError(f"Stage history is missing its epoch column: {stage}")
        history_epoch = int(pd.to_numeric(history["epoch"], errors="raise").max()) + 1

    metadata_path = run_dir / f"backup_{stage}" / "training_metadata.json"
    backup_epoch = 0
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        backup_epoch = int(metadata["epoch"])
        if backup_epoch < 0:
            raise ValueError(f"Invalid backup epoch for stage {stage}: {backup_epoch}")
    return max(history_epoch, backup_epoch)


def training_callbacks(
    run_dir: Path,
    stage: str,
    patience: int,
    append_history: bool = False,
    initial_best: float | None = None,
    lesion_validation: "LesionValidation | None" = None,
    archive_epochs: bool = False,
) -> list[tf.keras.callbacks.Callback]:
    monitor = lesion_validation.monitor if lesion_validation is not None else "val_high_risk_pr_auc"
    prefix = [lesion_validation] if lesion_validation is not None else []
    if archive_epochs:
        prefix.append(tf.keras.callbacks.ModelCheckpoint(
            filepath=str(run_dir / ("epoch_" + stage + "_{epoch:02d}.keras")), save_best_only=False, verbose=0))
    return prefix + [
        tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.BackupAndRestore(
            backup_dir=str(run_dir / f"backup_{stage}"), delete_checkpoint=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(run_dir / f"best_{stage}_model.keras"),
            monitor=monitor,
            mode="max",
            save_best_only=True,
            verbose=1,
            initial_value_threshold=initial_best,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            mode="max",
            factor=0.4,
            patience=max(1, patience // 2),
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            mode="max",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(
            str(run_dir / f"training_{stage}.csv"), append=append_history
        ),
        HostMemoryCleanup(),
    ]


def create_run_dir(output_root: Path, resume_run_dir: Path | None = None) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    if resume_run_dir is not None:
        return resume_run_dir.resolve()
    run_dir = output_root / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(exist_ok=False)
    return run_dir


def save_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


RESUME_CONFIG_KEYS = (
    "manifest",
    "output_root",
    "backbone",
    "image_size",
    "batch_size",
    "steps_per_epoch",
    "head_epochs",
    "fine_tune_epochs",
    "fine_tune_layers",
    "learning_rate",
    "fine_tune_learning_rate",
    "weight_decay",
    "dropout_rate",
    "metadata_modality_dropout",
    "risk_features",
    "risk_loss",
    "focal_gamma",
    "binary_target",
    "label_smoothing",
    "diagnosis_loss_weight",
    "sampling_alpha",
    "positive_sampling_fraction",
    "sampling_unit",
    "val_size",
    "test_size",
    "fold_assignments",
    "test_fold",
    "validation_fold",
    "sample_fraction",
    "data_workers",
    "prefetch_batches",
    "gradient_accumulation_steps",
    "seed",
    "mixed_precision",
    "weights",
    "warm_start_model",
    "reset_risk_head",
    "hard_negative_weights",
    "hard_negative_weights_sha256",
    "checkpoint_metric",
    "tta_scheme",
    "minimum_operating_recall",
    "freeze_batch_norm",
    "archive_epoch_checkpoints",
)


def validate_resume_state(
    run_dir: Path,
    config: dict[str, object],
    encoder: MetadataEncoder,
    summary: dict[str, object],
) -> None:
    required = ("run_config.json", "metadata_schema.json", "split_summary.json")
    missing = [name for name in required if not (run_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Resume run is missing state files: {missing}")

    saved_config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    saved_config.setdefault("checkpoint_metric", "image_pr_auc")
    saved_config.setdefault("tta_scheme", "legacy_v1")
    saved_config.setdefault("freeze_batch_norm", False)
    saved_config.setdefault("archive_epoch_checkpoints", False)
    saved_config.setdefault("sampling_unit", "image")
    mismatches = {
        key: {"saved": saved_config.get(key), "current": config.get(key)}
        for key in RESUME_CONFIG_KEYS
        if saved_config.get(key) != config.get(key)
    }
    if mismatches:
        raise ValueError(
            "Resume arguments do not match the saved run configuration: "
            + json.dumps(mismatches, indent=2)
        )

    saved_encoder = json.loads(
        (run_dir / "metadata_schema.json").read_text(encoding="utf-8")
    )
    current_encoder = json.loads(json.dumps(asdict(encoder)))
    if saved_encoder != current_encoder:
        raise ValueError("Resume metadata encoder does not match the saved run.")

    saved_summary = json.loads(
        (run_dir / "split_summary.json").read_text(encoding="utf-8")
    )
    if saved_summary != summary:
        raise ValueError("Resume data splits do not match the saved run.")


def split_summary(frame: pd.DataFrame) -> dict[str, object]:
    return {
        "images": int(len(frame)),
        "lesions": int(frame["lesion_id"].nunique()),
        "split_groups": int(frame["split_group_id"].nunique()),
        "sources": {str(k): int(v) for k, v in frame["source"].value_counts().items()},
        "targets": {str(k): int(v) for k, v in frame["target_name"].value_counts().items()},
        "diagnoses": {
            str(k): int(v) for k, v in frame["diagnosis"].value_counts().sort_index().items()
        },
        "source_diagnoses": [
            {"source": str(key[0]), "diagnosis": str(key[1]), "count": int(value)}
            for key, value in frame.groupby(["source", "diagnosis"]).size().items()
        ],
    }


def threshold_metrics(
    truth: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, object]:
    predictions = (probabilities >= threshold).astype("int32")
    matrix = confusion_matrix(truth, predictions, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(truth, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predictions)),
        "precision": float(precision_score(truth, predictions, zero_division=0)),
        "recall": float(recall_score(truth, predictions, zero_division=0)),
        "specificity": float(specificity),
        "f1": float(f1_score(truth, predictions, zero_division=0)),
        "matthews_correlation_coefficient": float(matthews_corrcoef(truth, predictions)),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "support": int(len(truth)),
        "confusion_matrix": matrix.tolist(),
    }


def wilson_interval(successes: int, total: int, z: float = 1.96) -> dict[str, float]:
    if total <= 0:
        return {"lower": 0.0, "upper": 0.0}
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return {"lower": float(max(0.0, center - margin)), "upper": float(min(1.0, center + margin))}


def operating_confidence_intervals(metrics: dict[str, object]) -> dict[str, dict[str, float]]:
    tn = int(metrics["true_negatives"])
    fp = int(metrics["false_positives"])
    fn = int(metrics["false_negatives"])
    tp = int(metrics["true_positives"])
    return {
        "accuracy": wilson_interval(tn + tp, tn + fp + fn + tp),
        "precision": wilson_interval(tp, tp + fp),
        "recall": wilson_interval(tp, tp + fn),
        "specificity": wilson_interval(tn, tn + fp),
    }


def select_thresholds(
    truth: np.ndarray,
    probabilities: np.ndarray,
    minimum_recall: float,
) -> tuple[dict[str, object], dict[str, object], list[dict[str, object]]]:
    rows = [
        threshold_metrics(truth, probabilities, float(threshold))
        for threshold in np.round(np.arange(0.001, 1.0, 0.001), 3)
    ]
    balanced = max(rows, key=lambda row: (row["balanced_accuracy"], row["f1"], row["recall"]))
    eligible = [row for row in rows if row["recall"] >= minimum_recall]
    if eligible:
        operating = max(
            eligible,
            key=lambda row: (row["accuracy"], row["precision"], row["specificity"]),
        )
    else:
        operating = max(rows, key=lambda row: (row["recall"], row["accuracy"]))
    return balanced, operating, rows


def fit_temperature(truth: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    clipped = np.clip(probabilities.astype("float64"), 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped))
    candidates = np.linspace(0.50, 3.00, 251)
    losses = []
    for temperature in candidates:
        calibrated = 1.0 / (1.0 + np.exp(-logits / temperature))
        losses.append(log_loss(truth, calibrated, labels=[0, 1]))
    index = int(np.argmin(losses))
    return float(candidates[index]), float(losses[index])


def apply_temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    clipped = np.clip(probabilities.astype("float64"), 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped))
    return (1.0 / (1.0 + np.exp(-logits / temperature))).astype("float32")


def unpack_predictions(
    model: tf.keras.Model, predictions: object
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(predictions, dict):
        output = predictions
    elif isinstance(predictions, (list, tuple)):
        output = dict(zip(model.output_names, predictions))
    else:
        raise TypeError("Expected a dictionary or sequence from the multitask model.")
    return np.asarray(output["high_risk"]).reshape(-1), np.asarray(output["diagnosis"])


def transform_tta_dataset(dataset: tf.data.Dataset, transform_index: int,
                          scheme: str = "legacy_v1") -> tf.data.Dataset:
    if scheme not in {"legacy_v1", "d4_v2"} or transform_index not in range(8):
        raise ValueError("Unsupported TTA scheme or transform index.")
    if transform_index == 0:
        return dataset

    def transform(
        inputs: dict[str, tf.Tensor],
        labels: dict[str, tf.Tensor],
        sample_weights: dict[str, tf.Tensor] | None = None,
    ) -> tuple[dict[str, tf.Tensor], ...]:
        image = inputs["image"]
        if transform_index == 1:
            image = tf.image.flip_left_right(image)
        elif transform_index == 2:
            image = tf.image.flip_up_down(image)
        elif transform_index == 3:
            image = tf.image.rot90(image, k=2)
        elif transform_index == 4:
            image = tf.image.rot90(image, k=1)
        elif transform_index == 5:
            image = tf.image.rot90(image, k=3)
        elif transform_index == 6:
            image = tf.transpose(image, perm=[0, 2, 1, 3])
        elif transform_index == 7:
            transposed = tf.transpose(image, perm=[0, 2, 1, 3])
            image = (tf.image.rot90(transposed, k=2) if scheme == "d4_v2"
                     else tf.image.flip_left_right(transposed))
        else:
            raise ValueError(f"Unsupported TTA transform: {transform_index}")
        transformed_inputs = {"image": image, "metadata": inputs["metadata"]}
        if sample_weights is None:
            return transformed_inputs, labels
        return transformed_inputs, labels, sample_weights

    return dataset.map(transform, num_parallel_calls=1)


def predict_with_tta(
    model: tf.keras.Model, dataset: tf.data.Dataset, transforms: int, scheme: str = "legacy_v1"
) -> tuple[np.ndarray, np.ndarray]:
    if transforms not in {1, 4, 8}:
        raise ValueError("TTA transforms must be one of 1, 4, or 8.")
    risk_sum: np.ndarray | None = None
    diagnosis_sum: np.ndarray | None = None
    for transform_index in range(transforms):
        risk, diagnosis = unpack_predictions(
            model,
            model.predict(transform_tta_dataset(dataset, transform_index, scheme), verbose=0),
        )
        risk = risk.astype("float64")
        diagnosis = diagnosis.astype("float64")
        if risk_sum is None or diagnosis_sum is None:
            risk_sum = risk
            diagnosis_sum = diagnosis
        else:
            risk_sum += risk
            diagnosis_sum += diagnosis
    if risk_sum is None or diagnosis_sum is None:
        raise RuntimeError("TTA prediction produced no views.")
    return (
        (risk_sum / transforms).astype("float32"),
        (diagnosis_sum / transforms).astype("float32"),
    )


def aggregate_predictions_by_lesion(
    frame: pd.DataFrame,
    risk_probabilities: np.ndarray,
    diagnosis_probabilities: np.ndarray | None = None,
) -> tuple[pd.DataFrame, np.ndarray | None]:
    if len(frame) != len(risk_probabilities):
        raise ValueError("Prediction count does not match the image manifest.")
    identity_columns = [
        "lesion_id",
        "source",
        "target",
        "target_name",
        "diagnosis",
        "diagnosis_index",
    ]
    working = frame[identity_columns].reset_index(drop=True).copy()
    clipped = np.clip(risk_probabilities.astype("float64"), 1e-6, 1.0 - 1e-6)
    working["risk_logit"] = np.log(clipped / (1.0 - clipped))
    grouped = working.groupby("lesion_id", sort=False)
    if (grouped["target"].nunique() > 1).any() or (grouped["diagnosis"].nunique() > 1).any():
        raise ValueError("A lesion has conflicting labels during prediction aggregation.")

    lesions = grouped[identity_columns[1:]].first().reset_index()
    lesions["image_count"] = grouped.size().to_numpy()
    mean_logits = grouped["risk_logit"].mean().to_numpy()
    lesions["probability"] = (1.0 / (1.0 + np.exp(-mean_logits))).astype("float32")

    aggregated_diagnosis = None
    if diagnosis_probabilities is not None:
        if diagnosis_probabilities.shape != (len(frame), len(DIAGNOSES)):
            raise ValueError("Diagnosis prediction shape does not match the manifest.")
        probability_columns = [f"diagnosis_probability_{name}" for name in DIAGNOSES]
        diagnosis_frame = pd.DataFrame(diagnosis_probabilities, columns=probability_columns)
        diagnosis_frame["lesion_id"] = working["lesion_id"]
        diagnosis_frame = diagnosis_frame.groupby("lesion_id", sort=False)[probability_columns].mean()
        diagnosis_frame = diagnosis_frame.reindex(lesions["lesion_id"])
        aggregated_diagnosis = diagnosis_frame.to_numpy(dtype="float32")
    return lesions, aggregated_diagnosis


def probability_metrics(truth: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
    return {
        "roc_auc": float(roc_auc_score(truth, probabilities)),
        "pr_auc": float(average_precision_score(truth, probabilities)),
        "brier_score": float(brier_score_loss(truth, probabilities)),
        "log_loss": float(log_loss(truth, clipped, labels=[0, 1])),
    }


def diagnosis_target_probabilities(
    diagnosis_probabilities: np.ndarray, binary_target: str
) -> np.ndarray:
    if binary_target == "melanoma":
        target_diagnoses = {"MEL"}
    elif binary_target == "malignant":
        target_diagnoses = set(MALIGNANT_DIAGNOSES)
    elif binary_target == "high_risk":
        target_diagnoses = set(HIGH_RISK_DIAGNOSES)
    else:
        raise ValueError(f"Unsupported binary target: {binary_target}")
    indices = [DIAGNOSIS_TO_INDEX[name] for name in sorted(target_diagnoses)]
    return diagnosis_probabilities[:, indices].sum(axis=1).astype("float32")


def select_probability_blend(
    truth: np.ndarray,
    direct_probabilities: np.ndarray,
    diagnosis_probabilities: np.ndarray,
) -> tuple[dict[str, float], np.ndarray, list[dict[str, float]]]:
    rows: list[dict[str, float]] = []
    candidates: list[np.ndarray] = []
    for direct_weight in np.linspace(0.0, 1.0, 21):
        blended = (
            direct_weight * direct_probabilities
            + (1.0 - direct_weight) * diagnosis_probabilities
        ).astype("float32")
        metrics = probability_metrics(truth, blended)
        rows.append({"direct_weight": float(direct_weight), **metrics})
        candidates.append(blended)
    selected_index = max(
        range(len(rows)),
        key=lambda index: (
            rows[index]["pr_auc"],
            rows[index]["roc_auc"],
            -rows[index]["brier_score"],
        ),
    )
    return rows[selected_index], candidates[selected_index], rows


class LesionValidation(tf.keras.callbacks.Callback):
    """Publish validation-only lesion metrics before checkpoint/LR callbacks run."""

    def __init__(self, frame: pd.DataFrame, dataset: tf.data.Dataset,
                 binary_target: str, minimum_recall: float, operating: bool = False):
        super().__init__()
        self.frame = frame
        self.dataset = dataset
        self.binary_target = binary_target
        self.minimum_recall = minimum_recall
        self.operating = operating
        self.monitor = "val_operating_score" if operating else "val_lesion_pr_auc"

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        from compare_model_runs import operating_metrics

        if logs is None:
            raise ValueError("Lesion validation needs the shared epoch logs.")
        direct, diagnosis = unpack_predictions(self.model, self.model.predict(self.dataset, verbose=0))
        if not np.isfinite(direct).all() or not np.isfinite(diagnosis).all():
            raise ValueError("Non-finite lesion validation predictions; refusing to save a checkpoint.")
        lesions, lesion_diagnosis = aggregate_predictions_by_lesion(self.frame, direct, diagnosis)
        blend, probabilities, _ = select_probability_blend(
            lesions.target.to_numpy(), lesions.probability.to_numpy(),
            diagnosis_target_probabilities(lesion_diagnosis, self.binary_target),
        )
        metrics = operating_metrics(lesions.target.to_numpy(), probabilities, self.minimum_recall)
        logs.update({f"val_lesion_{name}": metrics[name] for name in
                     ("pr_auc", "roc_auc", "specificity", "recall", "precision", "threshold")})
        logs["val_lesion_direct_weight"] = blend["direct_weight"]
        print(f"\nLesion validation: AP={metrics['pr_auc']:.6f}, ROC={metrics['roc_auc']:.6f}, "
              f"specificity={metrics['specificity']:.6f} at sensitivity={metrics['recall']:.6f}", flush=True)
        if self.operating:
            from operating_policy import DIRECT_WEIGHT, crossfit_policy, selection_score
            group_map = self.frame.groupby("lesion_id").split_group_id
            if group_map.nunique(dropna=False).ne(1).any():
                raise ValueError("A validation lesion crosses split groups.")
            lesions["split_group_id"] = lesions.lesion_id.map(group_map.first())
            fixed = (DIRECT_WEIGHT * lesions.probability.to_numpy() + (1 - DIRECT_WEIGHT) *
                     diagnosis_target_probabilities(lesion_diagnosis, self.binary_target))
            crossfit_metrics, _ = crossfit_policy(lesions, fixed, self.minimum_recall)
            logs["val_operating_score"] = selection_score(crossfit_metrics, self.minimum_recall)
            logs.update({f"val_crossfit_{name}": crossfit_metrics[name] for name in
                         ("recall", "specificity", "precision", "accuracy", "pr_auc", "roc_auc")})
            print(f"Group-crossfit operating metrics: sensitivity={crossfit_metrics['recall']:.6f}, "
                  f"specificity={crossfit_metrics['specificity']:.6f}, AP={crossfit_metrics['pr_auc']:.6f}", flush=True)


def diagnosis_metrics(truth: np.ndarray, probabilities: np.ndarray) -> dict[str, object]:
    predictions = probabilities.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(truth, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predictions)),
        "macro_precision": float(
            precision_score(truth, predictions, average="macro", zero_division=0)
        ),
        "macro_recall": float(recall_score(truth, predictions, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(truth, predictions, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(
            truth, predictions, labels=np.arange(len(DIAGNOSES))
        ).tolist(),
        "classification_report": classification_report(
            truth,
            predictions,
            labels=np.arange(len(DIAGNOSES)),
            target_names=list(DIAGNOSES),
            output_dict=True,
            zero_division=0,
        ),
    }


def plot_history(history_frames: list[pd.DataFrame], path: Path) -> None:
    history = pd.concat(history_frames, ignore_index=True)
    metrics = (
        ("loss", "val_loss"),
        ("high_risk_pr_auc", "val_high_risk_pr_auc"),
        ("diagnosis_accuracy", "val_diagnosis_accuracy"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for axis, (train_name, val_name) in zip(axes, metrics):
        if train_name in history and val_name in history:
            axis.plot(history.index, history[train_name], label=train_name)
            axis.plot(history.index, history[val_name], label=val_name)
        axis.set_xlabel("Epoch")
        axis.set_title(train_name.replace("_", " ").title())
        axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_confusion(
    matrix: list[list[int]], labels: list[str], path: Path, title: str
) -> None:
    values = np.asarray(matrix)
    side = max(5, len(labels) * 0.8)
    fig, axis = plt.subplots(figsize=(side, side * 0.85))
    axis.imshow(values, cmap="Blues")
    axis.set_title(title)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    axis.set_xticks(np.arange(len(labels)), labels=labels, rotation=45, ha="right")
    axis.set_yticks(np.arange(len(labels)), labels=labels)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            axis.text(column, row, int(values[row, column]), ha="center", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def evaluate_candidate(
    path: Path, dataset: tf.data.Dataset, frame: pd.DataFrame, binary_target: str
) -> dict[str, float]:
    tf.keras.backend.clear_session()
    model = tf.keras.models.load_model(path, compile=False)
    risk, diagnosis = unpack_predictions(model, model.predict(dataset, verbose=0))
    metrics = {
        f"image_{key}": value
        for key, value in probability_metrics(
            frame["target"].to_numpy(dtype="int32"), risk
        ).items()
    }
    lesion_predictions, lesion_diagnosis = aggregate_predictions_by_lesion(
        frame, risk, diagnosis
    )
    if lesion_diagnosis is None:
        raise RuntimeError("Candidate diagnosis probabilities were not aggregated.")
    lesion_probability_metrics = probability_metrics(
        lesion_predictions["target"].to_numpy(),
        lesion_predictions["probability"].to_numpy(),
    )
    metrics.update(
        {f"lesion_{name}": value for name, value in lesion_probability_metrics.items()}
    )
    blend, _, _ = select_probability_blend(
        lesion_predictions["target"].to_numpy(dtype="int32"),
        lesion_predictions["probability"].to_numpy(dtype="float32"),
        diagnosis_target_probabilities(lesion_diagnosis, binary_target),
    )
    metrics.update({f"lesion_blended_{name}": value for name, value in blend.items()})
    del model
    tf.keras.backend.clear_session()
    gc.collect()
    return metrics


def main() -> None:
    args = parse_args()
    validate_args(args)
    runtime = configure_runtime(args.mixed_precision)
    if args.require_gpu and not runtime["gpu_devices"]:
        raise RuntimeError("GPU training was required, but TensorFlow found no GPU.")
    set_random_seeds(args.seed)

    frame = load_manifest(args.manifest)
    frame = configure_binary_target(frame, args.binary_target)
    frame = maybe_downsample(frame, args.sample_fraction, args.seed)
    if args.fold_assignments is not None:
        splits = split_from_fold_assignments(
            frame,
            args.fold_assignments,
            args.test_fold,
            args.validation_fold,
        )
    else:
        splits = split_by_lesion_source_and_diagnosis(
            frame, args.val_size, args.test_size, args.seed
        )
    encoder = MetadataEncoder.fit(splits.train)
    run_dir = create_run_dir(args.output_root, args.resume_run_dir)

    config = vars(args).copy()
    if args.hard_negative_weights is not None:
        with args.hard_negative_weights.open("rb") as handle:
            config["hard_negative_weights_sha256"] = hashlib.file_digest(handle, "sha256").hexdigest()
    config = {
        key: str(value.resolve()) if isinstance(value, Path) else value
        for key, value in config.items()
    }
    config.update(runtime)
    config["model_version"] = 3
    config["binary_class_names"] = binary_class_names(args.binary_target)
    config["diagnoses"] = list(DIAGNOSES)
    config["high_risk_diagnoses"] = sorted(HIGH_RISK_DIAGNOSES)
    config["malignant_diagnoses"] = sorted(MALIGNANT_DIAGNOSES)
    config["primary_evaluation_unit"] = "lesion"
    config["binary_inference_metadata"] = "unknown"

    summary = {
        "full_dataset": split_summary(frame),
        "train": split_summary(splits.train),
        "validation": split_summary(splits.val),
        "test": split_summary(splits.test),
    }
    if args.resume_run_dir is not None:
        validate_resume_state(run_dir, config, encoder, summary)
        print(f"Resuming interrupted run: {run_dir}")
    else:
        save_json(run_dir / "run_config.json", config)
        save_json(run_dir / "metadata_schema.json", asdict(encoder))
        save_json(run_dir / "split_summary.json", summary)
        for name, split in (
            ("train", splits.train),
            ("val", splits.val),
            ("test", splits.test),
        ):
            split.drop(columns=["runtime_image_path"], errors="ignore").to_csv(
                run_dir / f"{name}_split.csv", index=False
            )

    print("Runtime:")
    print(json.dumps(runtime, indent=2))
    print("Split summary:")
    print(json.dumps(summary, indent=2))

    training_frame = splits.train
    if args.sampling_unit == "lesion":
        from lesion_sampling import exposure_audit
        if args.positive_sampling_fraction is not None:
            save_json(run_dir / "sampling_exposure.json", exposure_audit(
                training_frame, args.sampling_alpha, args.positive_sampling_fraction))
        print("Lesion-uniform exposure within source/diagnosis; all original images retained.", flush=True)
    if args.hard_negative_weights is not None:
        from targeted_sampling import assert_training_only, attach_weights
        assert_training_only(splits.train, [splits.val, splits.test])
        training_frame = attach_weights(splits.train, args.hard_negative_weights)
        print(f"Capped hard-negative sampling: {int(training_frame.sampling_weight.eq(2).sum())} "
              f"of {len(training_frame)} training images have 2x relative within-stratum exposure.")
    train_ds = build_dataset(
        training_frame,
        encoder,
        args.image_size,
        args.batch_size,
        True,
        args.data_workers,
        args.seed,
        args.sampling_alpha,
        args.positive_sampling_fraction,
        args.prefetch_batches,
        sampling_unit=args.sampling_unit,
    )
    val_ds = build_dataset(
        splits.val,
        encoder,
        args.image_size,
        args.batch_size,
        False,
        args.data_workers,
        args.seed,
        prefetch_batches=args.prefetch_batches,
        mask_metadata=True,
    )
    test_ds = build_dataset(
        splits.test,
        encoder,
        args.image_size,
        args.batch_size,
        False,
        args.data_workers,
        args.seed,
        prefetch_batches=args.prefetch_batches,
        mask_metadata=True,
    )
    steps_per_epoch = args.steps_per_epoch or math.ceil(len(splits.train) / args.batch_size)
    print(f"Training budget: {steps_per_epoch} batches per epoch, batch size {args.batch_size}.")
    monitor = {"image_pr_auc": "val_high_risk_pr_auc", "lesion_pr_auc": "val_lesion_pr_auc",
               "lesion_operating": "val_operating_score"}[args.checkpoint_metric]

    def lesion_callback() -> LesionValidation | None:
        if args.checkpoint_metric in {"lesion_pr_auc", "lesion_operating"}:
            return LesionValidation(splits.val, val_ds, args.binary_target, args.minimum_operating_recall,
                                    operating=args.checkpoint_metric == "lesion_operating")
        return None

    pretrained_weights = None if args.weights == "none" else args.weights
    model = build_model(
        args.image_size,
        len(encoder.feature_names),
        args.dropout_rate,
        args.metadata_modality_dropout,
        args.risk_features,
        args.backbone,
        pretrained_weights,
    )
    if args.warm_start_model is not None:
        warm_start = warm_start_compatible_layers(
            model,
            args.warm_start_model,
            args.risk_features,
            args.reset_risk_head,
        )
        save_json(run_dir / "warm_start.json", warm_start)
        print("Warm-start summary:")
        print(json.dumps(warm_start, indent=2))
    compile_model(
        model,
        args.learning_rate,
        args.weight_decay,
        args.label_smoothing,
        args.diagnosis_loss_weight,
        args.risk_loss,
        args.focal_gamma,
        args.gradient_accumulation_steps,
        freeze_batch_norm=args.freeze_batch_norm,
    )
    model.summary()
    head_path = run_dir / "best_head_model.keras"
    head_completed = stage_resume_epoch(run_dir, "head")
    if head_completed > args.head_epochs:
        raise ValueError(
            f"Head history has {head_completed} epochs, expected at most {args.head_epochs}."
        )
    head_before = stage_history(run_dir, "head")
    head_best = None
    if not head_before.empty and monitor in head_before:
        head_best = float(head_before[monitor].max())
    if head_completed < args.head_epochs:
        backup_weights = run_dir / "backup_head" / "latest.weights.h5"
        if head_completed and not backup_weights.exists():
            if not head_path.exists():
                raise FileNotFoundError(
                    "Partial head history has neither a recovery backup nor a best model."
                )
            print("Head backup is absent; resuming from the best completed head epoch.")
            model = tf.keras.models.load_model(head_path)
            compile_model(
                model,
                args.learning_rate,
                args.weight_decay,
                args.label_smoothing,
                args.diagnosis_loss_weight,
                args.risk_loss,
                args.focal_gamma,
                args.gradient_accumulation_steps,
                freeze_batch_norm=args.freeze_batch_norm,
            )
        if head_completed:
            print(f"Resuming head training at epoch {head_completed + 1}/{args.head_epochs}.")
        model.fit(
            train_ds,
            validation_data=val_ds,
            steps_per_epoch=steps_per_epoch,
            initial_epoch=head_completed,
            epochs=args.head_epochs,
            callbacks=training_callbacks(
                run_dir,
                "head",
                patience=3,
                append_history=head_completed > 0,
                initial_best=head_best,
                lesion_validation=lesion_callback(),
                archive_epochs=args.archive_epoch_checkpoints,
            ),
            verbose=args.fit_verbose,
        )
    else:
        print(f"Head training already completed ({head_completed}/{args.head_epochs} epochs).")
    if not head_path.exists():
        raise FileNotFoundError(f"Head checkpoint was not produced: {head_path}")
    history_frames = [stage_history(run_dir, "head").assign(stage="head")]

    candidate_paths = [head_path]
    if args.fine_tune_epochs:
        model = tf.keras.models.load_model(candidate_paths[0])
        trainable_count = make_backbone_trainable(model, args.fine_tune_layers)
        print(f"Fine-tuning {trainable_count} non-BatchNorm backbone layers.")
        compile_model(
            model,
            args.fine_tune_learning_rate,
            args.weight_decay,
            args.label_smoothing,
            args.diagnosis_loss_weight,
            args.risk_loss,
            args.focal_gamma,
            args.gradient_accumulation_steps,
            freeze_batch_norm=args.freeze_batch_norm,
        )
        fine_path = run_dir / "best_finetune_model.keras"
        fine_completed = stage_resume_epoch(run_dir, "finetune")
        if fine_completed > args.fine_tune_epochs:
            raise ValueError(
                f"Fine-tune history has {fine_completed} epochs, expected at most "
                f"{args.fine_tune_epochs}."
            )
        fine_before = stage_history(run_dir, "finetune")
        fine_best = None
        if not fine_before.empty and monitor in fine_before:
            fine_best = float(fine_before[monitor].max())
        if fine_completed < args.fine_tune_epochs:
            backup_weights = run_dir / "backup_finetune" / "latest.weights.h5"
            if fine_completed and not backup_weights.exists():
                if not fine_path.exists():
                    raise FileNotFoundError(
                        "Partial fine-tune history has neither a recovery backup nor a best model."
                    )
                print("Fine-tune backup is absent; resuming from its best completed epoch.")
                model = tf.keras.models.load_model(fine_path)
                make_backbone_trainable(model, args.fine_tune_layers)
                compile_model(
                    model,
                    args.fine_tune_learning_rate,
                    args.weight_decay,
                    args.label_smoothing,
                    args.diagnosis_loss_weight,
                    args.risk_loss,
                    args.focal_gamma,
                    args.gradient_accumulation_steps,
                    freeze_batch_norm=args.freeze_batch_norm,
                )
            if fine_completed:
                print(
                    f"Resuming fine-tuning at epoch {fine_completed + 1}/"
                    f"{args.fine_tune_epochs}."
                )
            model.fit(
                train_ds,
                validation_data=val_ds,
                steps_per_epoch=steps_per_epoch,
                initial_epoch=fine_completed,
                epochs=args.fine_tune_epochs,
                callbacks=training_callbacks(
                    run_dir,
                    "finetune",
                    patience=5,
                    append_history=fine_completed > 0,
                    initial_best=fine_best,
                    lesion_validation=lesion_callback(),
                    archive_epochs=args.archive_epoch_checkpoints,
                ),
                verbose=args.fit_verbose,
            )
            model.save(run_dir / "final_finetuned_model.keras")
        else:
            print(
                f"Fine-tuning already completed ({fine_completed}/"
                f"{args.fine_tune_epochs} epochs)."
            )
        if not fine_path.exists():
            raise FileNotFoundError(f"Fine-tune checkpoint was not produced: {fine_path}")
        history_frames.append(stage_history(run_dir, "finetune").assign(stage="finetune"))
        candidate_paths.append(fine_path)

    del model
    gc.collect()
    history = pd.concat(history_frames, ignore_index=True)
    history.to_csv(run_dir / "training_history.csv", index=False)
    plot_history(history_frames, run_dir / "training_curves.png")

    if args.finalization_mode == "none":
        print("Training checkpoints saved; finalization was explicitly skipped.")
        print("Artifacts saved to:", run_dir.resolve())
        return
    if args.finalization_mode == "fresh_process":
        finalizer = PROJECT_ROOT / "finalize_multitask_run.py"
        command = [
            sys.executable,
            str(finalizer),
            "--run-dir",
            str(run_dir),
            "--batch-size",
            str(args.evaluation_batch_size),
            "--data-workers",
            str(args.data_workers),
            "--prefetch-batches",
            str(args.prefetch_batches),
        ]
        print("Replacing the training process with low-memory finalization:")
        print(" ".join(command))
        sys.stdout.flush()
        sys.stderr.flush()
        os.execv(sys.executable, command)

    candidate_metrics: dict[str, dict[str, float]] = {}
    for path in candidate_paths:
        candidate_metrics[path.name] = evaluate_candidate(
            path, val_ds, splits.val, args.binary_target
        )
    selected_path = max(
        candidate_paths,
        key=lambda path: (
            candidate_metrics[path.name].get("lesion_blended_pr_auc", -1.0),
            candidate_metrics[path.name].get("lesion_blended_roc_auc", -1.0),
        ),
    )
    model = tf.keras.models.load_model(selected_path)
    model.save(run_dir / "best_model.keras")
    save_json(
        run_dir / "model_selection.json",
        {"selected_candidate": selected_path.name, "validation_metrics": candidate_metrics},
    )

    history = pd.concat(history_frames, ignore_index=True)
    history.to_csv(run_dir / "training_history.csv", index=False)
    plot_history(history_frames, run_dir / "training_curves.png")

    val_image_direct_raw, val_image_diagnosis_probabilities = predict_with_tta(
        model, val_ds, args.tta_transforms, args.tta_scheme
    )
    test_image_direct_raw, test_image_diagnosis_probabilities = predict_with_tta(
        model, test_ds, args.tta_transforms, args.tta_scheme
    )
    val_lesions, val_lesion_diagnosis_probabilities = aggregate_predictions_by_lesion(
        splits.val, val_image_direct_raw, val_image_diagnosis_probabilities
    )
    test_lesions, test_lesion_diagnosis_probabilities = aggregate_predictions_by_lesion(
        splits.test, test_image_direct_raw, test_image_diagnosis_probabilities
    )
    if val_lesion_diagnosis_probabilities is None or test_lesion_diagnosis_probabilities is None:
        raise RuntimeError("Diagnosis probabilities were not aggregated.")

    val_truth = val_lesions["target"].to_numpy(dtype="int32")
    test_truth = test_lesions["target"].to_numpy(dtype="int32")
    val_direct_raw = val_lesions["probability"].to_numpy(dtype="float32")
    test_direct_raw = test_lesions["probability"].to_numpy(dtype="float32")
    val_diagnosis_target_raw = diagnosis_target_probabilities(
        val_lesion_diagnosis_probabilities, args.binary_target
    )
    test_diagnosis_target_raw = diagnosis_target_probabilities(
        test_lesion_diagnosis_probabilities, args.binary_target
    )
    selected_blend, val_risk_raw, blend_rows = select_probability_blend(
        val_truth, val_direct_raw, val_diagnosis_target_raw
    )
    direct_weight = float(selected_blend["direct_weight"])
    test_risk_raw = (
        direct_weight * test_direct_raw
        + (1.0 - direct_weight) * test_diagnosis_target_raw
    ).astype("float32")
    val_image_diagnosis_target_raw = diagnosis_target_probabilities(
        val_image_diagnosis_probabilities, args.binary_target
    )
    test_image_diagnosis_target_raw = diagnosis_target_probabilities(
        test_image_diagnosis_probabilities, args.binary_target
    )
    val_image_risk_raw = (
        direct_weight * val_image_direct_raw
        + (1.0 - direct_weight) * val_image_diagnosis_target_raw
    ).astype("float32")
    test_image_risk_raw = (
        direct_weight * test_image_direct_raw
        + (1.0 - direct_weight) * test_image_diagnosis_target_raw
    ).astype("float32")
    pd.DataFrame(blend_rows).to_csv(run_dir / "probability_blend_sweep.csv", index=False)
    save_json(
        run_dir / "probability_blend.json",
        {
            "selection_unit": "lesion",
            "selection_split": "validation",
            "direct_weight": direct_weight,
            "diagnosis_target_weight": 1.0 - direct_weight,
            "selected_validation_metrics": selected_blend,
            "direct_validation_metrics": probability_metrics(val_truth, val_direct_raw),
            "diagnosis_target_validation_metrics": probability_metrics(
                val_truth, val_diagnosis_target_raw
            ),
        },
    )
    temperature, calibrated_validation_loss = fit_temperature(val_truth, val_risk_raw)
    val_risk = apply_temperature(val_risk_raw, temperature)
    test_risk = apply_temperature(test_risk_raw, temperature)
    val_image_risk = apply_temperature(val_image_risk_raw, temperature)
    test_image_risk = apply_temperature(test_image_risk_raw, temperature)
    balanced_threshold, operating_threshold, threshold_rows = select_thresholds(
        val_truth, val_risk, args.minimum_operating_recall
    )
    pd.DataFrame(threshold_rows).drop(columns=["confusion_matrix"]).to_csv(
        run_dir / "threshold_sweep.csv", index=False
    )
    save_json(
        run_dir / "calibration.json",
        {
            "method": "temperature_scaling",
            "evaluation_unit": "lesion",
            "tta_transforms": args.tta_transforms,
            "tta_scheme": args.tta_scheme,
            "temperature": temperature,
            "validation_log_loss": calibrated_validation_loss,
            "raw_validation_metrics": probability_metrics(val_truth, val_risk_raw),
            "calibrated_validation_metrics": probability_metrics(val_truth, val_risk),
        },
    )
    save_json(
        run_dir / "selected_threshold.json",
        {
            "balanced_accuracy_threshold": balanced_threshold["threshold"],
            "balanced_accuracy_validation_metrics": balanced_threshold,
            "operating_threshold": operating_threshold["threshold"],
            "minimum_operating_recall": args.minimum_operating_recall,
            "operating_validation_metrics": operating_threshold,
        },
    )

    operating_value = float(operating_threshold["threshold"])
    selected_predictions = (test_risk >= operating_value).astype("int32")
    operating_metrics = threshold_metrics(test_truth, test_risk, operating_value)
    operating_metrics["confidence_intervals_95"] = operating_confidence_intervals(
        operating_metrics
    )
    risk_results = {
        "evaluation_unit": "lesion",
        "tta_transforms": args.tta_transforms,
        "tta_scheme": args.tta_scheme,
        "direct_weight": direct_weight,
        "component_probability_metrics": {
            "direct_head": probability_metrics(test_truth, test_direct_raw),
            "diagnosis_target": probability_metrics(test_truth, test_diagnosis_target_raw),
        },
        "raw_probability_metrics": probability_metrics(test_truth, test_risk_raw),
        "calibrated_probability_metrics": probability_metrics(test_truth, test_risk),
        "threshold_metrics": {
            "default_0_5": threshold_metrics(test_truth, test_risk, 0.5),
            "balanced_accuracy": threshold_metrics(
                test_truth, test_risk, float(balanced_threshold["threshold"])
            ),
            "operating": operating_metrics,
        },
        "classification_report": classification_report(
            test_truth,
            selected_predictions,
            labels=[0, 1],
            target_names=binary_class_names(args.binary_target),
            output_dict=True,
            zero_division=0,
        ),
    }
    diagnosis_results = diagnosis_metrics(
        test_lesions["diagnosis_index"].to_numpy(dtype="int32"),
        test_lesion_diagnosis_probabilities,
    )
    image_truth = splits.test["target"].to_numpy(dtype="int32")
    image_selected_predictions = (test_image_risk >= operating_value).astype("int32")
    image_operating_metrics = threshold_metrics(
        image_truth, test_image_risk, operating_value
    )
    image_operating_metrics["confidence_intervals_95"] = operating_confidence_intervals(
        image_operating_metrics
    )
    image_risk_results = {
        "evaluation_unit": "image",
        "tta_transforms": args.tta_transforms,
        "tta_scheme": args.tta_scheme,
        "direct_weight": direct_weight,
        "component_probability_metrics": {
            "direct_head": probability_metrics(image_truth, test_image_direct_raw),
            "diagnosis_target": probability_metrics(
                image_truth, test_image_diagnosis_target_raw
            ),
        },
        "raw_probability_metrics": probability_metrics(image_truth, test_image_risk_raw),
        "calibrated_probability_metrics": probability_metrics(image_truth, test_image_risk),
        "threshold_metrics": {
            "default_0_5": threshold_metrics(image_truth, test_image_risk, 0.5),
            "balanced_accuracy": threshold_metrics(
                image_truth, test_image_risk, float(balanced_threshold["threshold"])
            ),
            "operating": image_operating_metrics,
        },
        "classification_report": classification_report(
            image_truth,
            image_selected_predictions,
            labels=[0, 1],
            target_names=binary_class_names(args.binary_target),
            output_dict=True,
            zero_division=0,
        ),
    }
    image_diagnosis_results = diagnosis_metrics(
        splits.test["diagnosis_index"].to_numpy(dtype="int32"),
        test_image_diagnosis_probabilities,
    )
    save_json(run_dir / "test_risk_metrics.json", risk_results)
    save_json(run_dir / "test_diagnosis_metrics.json", diagnosis_results)
    save_json(run_dir / "test_image_risk_metrics.json", image_risk_results)
    save_json(run_dir / "test_image_diagnosis_metrics.json", image_diagnosis_results)

    validation_predictions = splits.val.drop(
        columns=["runtime_image_path"], errors="ignore"
    ).copy()
    validation_predictions["probability_direct_raw"] = val_image_direct_raw
    validation_predictions["probability_diagnosis_target_raw"] = (
        val_image_diagnosis_target_raw
    )
    validation_predictions[f"probability_{args.binary_target}_raw"] = val_image_risk_raw
    validation_predictions[f"probability_{args.binary_target}"] = val_image_risk
    validation_predictions[f"{args.binary_target}_prediction"] = (
        val_image_risk >= operating_value
    ).astype("int32")
    if args.binary_target == "high_risk":
        validation_predictions["probability_high_risk_raw"] = val_image_risk_raw
        validation_predictions["probability_high_risk"] = val_image_risk
    validation_diagnosis_indices = val_image_diagnosis_probabilities.argmax(axis=1)
    validation_predictions["diagnosis_prediction"] = [
        DIAGNOSES[index] for index in validation_diagnosis_indices
    ]
    validation_predictions["diagnosis_confidence"] = val_image_diagnosis_probabilities.max(axis=1)
    validation_predictions.to_csv(run_dir / "val_predictions.csv", index=False)

    predictions = splits.test.drop(columns=["runtime_image_path"], errors="ignore").copy()
    predictions["probability_direct_raw"] = test_image_direct_raw
    predictions["probability_diagnosis_target_raw"] = test_image_diagnosis_target_raw
    predictions[f"probability_{args.binary_target}_raw"] = test_image_risk_raw
    predictions[f"probability_{args.binary_target}"] = test_image_risk
    predictions[f"{args.binary_target}_prediction"] = image_selected_predictions
    if args.binary_target == "high_risk":
        predictions["probability_high_risk_raw"] = test_image_risk_raw
        predictions["probability_high_risk"] = test_image_risk
        predictions["risk_prediction"] = image_selected_predictions
    diagnosis_predictions = test_image_diagnosis_probabilities.argmax(axis=1)
    predictions["diagnosis_prediction"] = [DIAGNOSES[index] for index in diagnosis_predictions]
    predictions["diagnosis_confidence"] = test_image_diagnosis_probabilities.max(axis=1)
    predictions.to_csv(run_dir / "test_predictions.csv", index=False)

    val_lesions["probability_direct_raw"] = val_direct_raw
    val_lesions["probability_diagnosis_target_raw"] = val_diagnosis_target_raw
    val_lesions["probability_raw"] = val_risk_raw
    val_lesions["probability"] = val_risk
    val_lesions["prediction"] = (val_risk >= operating_value).astype("int32")
    val_lesions["diagnosis_prediction"] = [
        DIAGNOSES[index] for index in val_lesion_diagnosis_probabilities.argmax(axis=1)
    ]
    val_lesions["diagnosis_confidence"] = val_lesion_diagnosis_probabilities.max(axis=1)
    val_lesions.to_csv(run_dir / "val_lesion_predictions.csv", index=False)

    test_lesions["probability_direct_raw"] = test_direct_raw
    test_lesions["probability_diagnosis_target_raw"] = test_diagnosis_target_raw
    test_lesions["probability_raw"] = test_risk_raw
    test_lesions["probability"] = test_risk
    test_lesions["prediction"] = selected_predictions
    test_lesions["diagnosis_prediction"] = [
        DIAGNOSES[index] for index in test_lesion_diagnosis_probabilities.argmax(axis=1)
    ]
    test_lesions["diagnosis_confidence"] = test_lesion_diagnosis_probabilities.max(axis=1)
    test_lesions.to_csv(run_dir / "test_lesion_predictions.csv", index=False)

    per_source = {
        str(source): threshold_metrics(
            group["target"].to_numpy(),
            group["probability"].to_numpy(),
            operating_value,
        )
        for source, group in test_lesions.groupby("source")
    }
    # Single-diagnosis groups contain only one binary class, so report direct error counts instead.
    diagnosis_error_summary = []
    for diagnosis, group in test_lesions.groupby("diagnosis"):
        diagnosis_error_summary.append(
            {
                "diagnosis": str(diagnosis),
                "lesion_support": int(len(group)),
                "image_support": int(group["image_count"].sum()),
                "correct_binary": int((group["target"] == group["prediction"]).sum()),
                "binary_error": int((group["target"] != group["prediction"]).sum()),
                "mean_probability": float(group["probability"].mean()),
            }
        )
    save_json(run_dir / "test_risk_metrics_by_source.json", per_source)
    save_json(run_dir / "test_errors_by_diagnosis.json", diagnosis_error_summary)

    plot_confusion(
        risk_results["threshold_metrics"]["operating"]["confusion_matrix"],
        binary_class_names(args.binary_target),
        run_dir / "risk_confusion_matrix.png",
        f"Risk Confusion Matrix @ {operating_value:.2f}",
    )
    plot_confusion(
        diagnosis_results["confusion_matrix"],
        list(DIAGNOSES),
        run_dir / "diagnosis_confusion_matrix.png",
        "Diagnosis Confusion Matrix",
    )

    print("Selected model:", selected_path.name)
    print("Risk test metrics:")
    print(json.dumps(risk_results, indent=2))
    print("Diagnosis test metrics:")
    print(json.dumps({key: value for key, value in diagnosis_results.items() if key not in {"classification_report", "confusion_matrix"}}, indent=2))
    print("Artifacts saved to:", run_dir.resolve())


if __name__ == "__main__":
    main()
