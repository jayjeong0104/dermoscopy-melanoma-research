from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
MATPLOTLIB_CONFIG = Path(tempfile.gettempdir()) / "skin_cancer_mplconfig"
MATPLOTLIB_CONFIG.mkdir(exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(MATPLOTLIB_CONFIG)

import matplotlib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


matplotlib.use("Agg")
import matplotlib.pyplot as plt


DX_LABELS = {
    "akiec": "Actinic keratoses / intraepithelial carcinoma",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesion",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevus",
    "vasc": "Vascular lesion",
}
DEFAULT_MALIGNANT_CLASSES = ("mel", "bcc", "akiec")
THRESHOLD_METRICS = ("accuracy", "balanced_accuracy", "f1", "precision", "recall")


@dataclass(frozen=True)
class SplitFrames:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a TensorFlow CNN for early skin cancer detection with HAM10000."
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=Path("HAM10000_metadata.csv"),
        help="Path to the HAM10000 metadata CSV.",
    )
    parser.add_argument(
        "--image-dirs",
        nargs="+",
        type=Path,
        default=[Path("HAM10000_images_part_1"), Path("HAM10000_images_part_2")],
        help="One or more directories containing the dermatoscopic images.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts"),
        help="Parent directory where each training run will be saved.",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Square image size used for training and evaluation.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Initial Adam learning rate.",
    )
    parser.add_argument(
        "--dropout-rate",
        type=float,
        default=0.35,
        help="Dropout applied in the classifier head.",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.15,
        help="Fraction of lesions used for validation.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.15,
        help="Fraction of lesions used for final testing.",
    )
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=1.0,
        help="Optional lesion-level downsampling fraction for quick experiments.",
    )
    parser.add_argument(
        "--malignant-classes",
        nargs="+",
        default=list(DEFAULT_MALIGNANT_CLASSES),
        help="Diagnosis codes treated as malignant in the binary task.",
    )
    parser.add_argument(
        "--threshold-metric",
        choices=THRESHOLD_METRICS,
        default="accuracy",
        help="Validation metric used to select the classification threshold.",
    )
    parser.add_argument(
        "--min-threshold",
        type=float,
        default=0.10,
        help="Minimum threshold considered during validation-based threshold tuning.",
    )
    parser.add_argument(
        "--max-threshold",
        type=float,
        default=0.90,
        help="Maximum threshold considered during validation-based threshold tuning.",
    )
    parser.add_argument(
        "--threshold-step",
        type=float,
        default=0.01,
        help="Step size used during validation-based threshold tuning.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def set_random_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def validate_args(args: argparse.Namespace) -> None:
    if not args.metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {args.metadata_path}")
    for image_dir in args.image_dirs:
        if not image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    if args.image_size < 32:
        raise ValueError("--image-size must be at least 32.")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive.")
    if not 0 < args.val_size < 1:
        raise ValueError("--val-size must be between 0 and 1.")
    if not 0 < args.test_size < 1:
        raise ValueError("--test-size must be between 0 and 1.")
    if args.val_size + args.test_size >= 1:
        raise ValueError("--val-size + --test-size must be less than 1.")
    if not 0 < args.sample_fraction <= 1:
        raise ValueError("--sample-fraction must be between 0 and 1.")
    if not 0 < args.min_threshold < 1:
        raise ValueError("--min-threshold must be between 0 and 1.")
    if not 0 < args.max_threshold < 1:
        raise ValueError("--max-threshold must be between 0 and 1.")
    if args.min_threshold >= args.max_threshold:
        raise ValueError("--min-threshold must be less than --max-threshold.")
    if not 0 < args.threshold_step < 1:
        raise ValueError("--threshold-step must be between 0 and 1.")


def build_image_index(image_dirs: list[Path]) -> tuple[dict[str, str], dict[str, list[str]]]:
    image_index: dict[str, str] = {}
    duplicate_index: dict[str, list[str]] = {}

    for image_dir in image_dirs:
        for image_path in sorted(image_dir.glob("*.jpg")):
            stem = image_path.stem
            if stem.endswith(" (2)"):
                base_stem = stem[:-4]
                duplicate_index.setdefault(base_stem, []).append(str(image_path.resolve()))
                continue
            if stem in image_index:
                raise ValueError(f"Found multiple primary image files for {stem}.")
            image_index[stem] = str(image_path.resolve())

    return image_index, duplicate_index


def load_metadata(
    metadata_path: Path,
    image_dirs: list[Path],
    malignant_classes: set[str],
) -> pd.DataFrame:
    df = pd.read_csv(metadata_path)
    required_columns = {"lesion_id", "image_id", "dx"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Metadata is missing required columns: {sorted(missing_columns)}")

    image_index, duplicate_index = build_image_index(image_dirs)
    image_paths: list[str] = []
    missing_images: list[str] = []
    fallback_duplicates = 0

    for image_id in df["image_id"]:
        image_path = image_index.get(image_id)
        if image_path is None:
            duplicate_matches = duplicate_index.get(image_id, [])
            if duplicate_matches:
                image_path = duplicate_matches[0]
                fallback_duplicates += 1
            else:
                missing_images.append(image_id)
                continue
        image_paths.append(image_path)

    if missing_images:
        sample = ", ".join(missing_images[:5])
        raise FileNotFoundError(
            f"Missing {len(missing_images)} images referenced by metadata. Sample: {sample}"
        )

    df = df.copy()
    df["image_path"] = image_paths
    df["diagnosis_name"] = df["dx"].map(DX_LABELS).fillna(df["dx"])
    df["target"] = df["dx"].isin(malignant_classes).astype(np.int32)
    df["target_name"] = np.where(df["target"] == 1, "malignant", "benign")
    df["is_duplicate_file_only"] = False

    if fallback_duplicates:
        print(f"Used duplicate-suffixed image files for {fallback_duplicates} metadata rows.")

    return df


def maybe_downsample_lesions(
    df: pd.DataFrame,
    sample_fraction: float,
    seed: int,
) -> pd.DataFrame:
    if sample_fraction >= 1.0:
        return df

    lesion_level = df[["lesion_id", "target"]].drop_duplicates().copy()
    sampled_lesions = []

    for target, group in lesion_level.groupby("target"):
        take = max(1, int(round(len(group) * sample_fraction)))
        sampled_group = group.sample(n=take, random_state=seed + int(target))
        sampled_lesions.append(sampled_group)

    sampled_lesion_ids = pd.concat(sampled_lesions)["lesion_id"]
    return df[df["lesion_id"].isin(sampled_lesion_ids)].copy()


def split_by_lesion(
    df: pd.DataFrame,
    val_size: float,
    test_size: float,
    seed: int,
) -> SplitFrames:
    lesion_level = df[["lesion_id", "target"]].drop_duplicates().copy()
    class_counts = lesion_level["target"].value_counts()

    if len(class_counts) < 2:
        raise ValueError("The dataset must contain both benign and malignant lesions.")
    if class_counts.min() < 3:
        raise ValueError("Each class needs at least 3 lesions to create train/val/test splits.")

    train_lesions, temp_lesions = train_test_split(
        lesion_level,
        test_size=val_size + test_size,
        stratify=lesion_level["target"],
        random_state=seed,
    )
    test_ratio_within_temp = test_size / (val_size + test_size)
    val_lesions, test_lesions = train_test_split(
        temp_lesions,
        test_size=test_ratio_within_temp,
        stratify=temp_lesions["target"],
        random_state=seed,
    )

    train_df = df[df["lesion_id"].isin(train_lesions["lesion_id"])].copy()
    val_df = df[df["lesion_id"].isin(val_lesions["lesion_id"])].copy()
    test_df = df[df["lesion_id"].isin(test_lesions["lesion_id"])].copy()

    return SplitFrames(train=train_df, val=val_df, test=test_df)


def build_dataset(
    frame: pd.DataFrame,
    image_size: int,
    batch_size: int,
    training: bool,
) -> tf.data.Dataset:
    paths = frame["image_path"].astype(str).to_numpy()
    labels = frame["target"].astype(np.float32).to_numpy()

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        dataset = dataset.shuffle(buffer_size=len(frame), reshuffle_each_iteration=True)

    def preprocess(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        image = tf.io.read_file(path)
        image = tf.io.decode_image(image, channels=3, expand_animations=False)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, [image_size, image_size])
        return image, label

    dataset = dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def build_model(image_size: int, dropout_rate: float) -> tf.keras.Model:
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomContrast(0.1),
        ],
        name="augmentation",
    )

    inputs = tf.keras.Input(shape=(image_size, image_size, 3), name="image")
    x = data_augmentation(inputs)

    for filters in (32, 64, 128, 256):
        x = tf.keras.layers.Conv2D(filters, 3, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv2D(filters, 3, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.Dropout(min(dropout_rate, 0.25))(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="malignancy")(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="ham10000_cnn")


def compute_balanced_class_weights(labels: pd.Series) -> dict[int, float]:
    classes = np.array(sorted(labels.unique()))
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels.to_numpy())
    return {int(label): float(weight) for label, weight in zip(classes, weights)}


def create_run_dir(output_root: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def predict_probabilities(model: tf.keras.Model, dataset: tf.data.Dataset, verbose: int = 0) -> np.ndarray:
    return model.predict(dataset, verbose=verbose).ravel()


def compute_threshold_metrics(
    truth: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, object]:
    predictions = (probabilities >= threshold).astype(np.int32)
    matrix = confusion_matrix(truth, predictions, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    metrics = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(truth, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predictions)),
        "precision": float(precision_score(truth, predictions, zero_division=0)),
        "recall": float(recall_score(truth, predictions, zero_division=0)),
        "f1": float(f1_score(truth, predictions, zero_division=0)),
        "specificity": float(specificity),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "support": int(len(truth)),
    }
    report = classification_report(
        truth,
        predictions,
        target_names=["benign", "malignant"],
        output_dict=True,
        zero_division=0,
    )
    return {
        "metrics": metrics,
        "report": report,
        "confusion_matrix": matrix.tolist(),
    }


def sweep_thresholds(
    truth: np.ndarray,
    probabilities: np.ndarray,
    metric_name: str,
    min_threshold: float,
    max_threshold: float,
    step: float,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    thresholds = np.arange(min_threshold, max_threshold + (step / 2.0), step)
    rows: list[dict[str, float]] = []
    best_row: dict[str, float] | None = None

    for threshold in thresholds:
        summary = compute_threshold_metrics(truth, probabilities, float(np.round(threshold, 6)))
        row = summary["metrics"]
        rows.append(row)

        if best_row is None:
            best_row = row
            continue

        candidate_key = (
            row[metric_name],
            row["recall"],
            row["precision"],
            row["balanced_accuracy"],
        )
        best_key = (
            best_row[metric_name],
            best_row["recall"],
            best_row["precision"],
            best_row["balanced_accuracy"],
        )
        if candidate_key > best_key:
            best_row = row

    if best_row is None:
        raise ValueError("Threshold sweep produced no candidates.")

    return best_row, rows


def save_threshold_sweep(path: Path, rows: list[dict[str, float]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def plot_history(history: tf.keras.callbacks.History, output_path: Path) -> None:
    history_frame = pd.DataFrame(history.history)
    metrics_to_plot = [
        ("loss", "val_loss"),
        ("accuracy", "val_accuracy"),
        ("auc", "val_auc"),
    ]

    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(15, 4))
    for axis, (train_metric, val_metric) in zip(axes, metrics_to_plot):
        axis.plot(history_frame[train_metric], label=train_metric)
        axis.plot(history_frame[val_metric], label=val_metric)
        axis.set_title(train_metric.replace("_", " ").title())
        axis.set_xlabel("Epoch")
        axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_confusion_matrix(matrix: np.ndarray, output_path: Path, title: str) -> None:
    fig, axis = plt.subplots(figsize=(5, 4))
    axis.imshow(matrix, cmap="Blues")
    axis.set_title(title)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    axis.set_xticks([0, 1], labels=["benign", "malignant"])
    axis.set_yticks([0, 1], labels=["benign", "malignant"])

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            axis.text(col, row, int(matrix[row, col]), ha="center", va="center")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def split_summary(frame: pd.DataFrame) -> dict[str, object]:
    return {
        "rows": int(len(frame)),
        "lesions": int(frame["lesion_id"].nunique()),
        "class_counts": {str(k): int(v) for k, v in frame["target"].value_counts().sort_index().items()},
        "dx_counts": {str(k): int(v) for k, v in frame["dx"].value_counts().sort_index().items()},
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_random_seeds(args.seed)

    malignant_classes = {label.strip() for label in args.malignant_classes if label.strip()}
    metadata = load_metadata(args.metadata_path, args.image_dirs, malignant_classes)
    metadata = maybe_downsample_lesions(metadata, args.sample_fraction, args.seed)
    splits = split_by_lesion(metadata, args.val_size, args.test_size, args.seed)

    run_dir = create_run_dir(args.output_root)
    save_json(
        run_dir / "run_config.json",
        {
            "metadata_path": str(args.metadata_path.resolve()),
            "image_dirs": [str(path.resolve()) for path in args.image_dirs],
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "image_size": args.image_size,
            "learning_rate": args.learning_rate,
            "dropout_rate": args.dropout_rate,
            "val_size": args.val_size,
            "test_size": args.test_size,
            "sample_fraction": args.sample_fraction,
            "malignant_classes": sorted(malignant_classes),
            "threshold_metric": args.threshold_metric,
            "min_threshold": args.min_threshold,
            "max_threshold": args.max_threshold,
            "threshold_step": args.threshold_step,
            "seed": args.seed,
            "tensorflow_version": tf.__version__,
        },
    )

    split_payload = {
        "full_dataset": split_summary(metadata),
        "train": split_summary(splits.train),
        "validation": split_summary(splits.val),
        "test": split_summary(splits.test),
        "duplicate_primary_files_ignored": int(
            sum(1 for path in Path(args.image_dirs[0]).parent.glob("HAM10000_images_part_*/*.jpg") if " (2)" in path.stem)
        ),
    }
    save_json(run_dir / "split_summary.json", split_payload)

    for split_name, frame in (("train", splits.train), ("val", splits.val), ("test", splits.test)):
        frame.to_csv(run_dir / f"{split_name}_split.csv", index=False)

    print("Split summary:")
    print(json.dumps(split_payload, indent=2))
    print("Training label balance:", Counter(splits.train["target"]))

    train_ds = build_dataset(splits.train, args.image_size, args.batch_size, training=True)
    val_ds = build_dataset(splits.val, args.image_size, args.batch_size, training=False)
    test_ds = build_dataset(splits.test, args.image_size, args.batch_size, training=False)

    class_weight = compute_balanced_class_weights(splits.train["target"])
    print("Class weights:", class_weight)

    model = build_model(args.image_size, args.dropout_rate)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

    callbacks: list[tf.keras.callbacks.Callback] = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max", patience=6, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(run_dir / "best_model.keras"),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
        ),
        tf.keras.callbacks.CSVLogger(str(run_dir / "training_log.csv")),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    model.save(run_dir / "final_model.keras")
    plot_history(history, run_dir / "training_curves.png")

    keras_evaluation = model.evaluate(test_ds, return_dict=True, verbose=1)
    val_probabilities = predict_probabilities(model, val_ds, verbose=1)
    test_probabilities = predict_probabilities(model, test_ds, verbose=1)
    val_truth = splits.val["target"].to_numpy()
    truth = splits.test["target"].to_numpy()

    selected_threshold, threshold_rows = sweep_thresholds(
        truth=val_truth,
        probabilities=val_probabilities,
        metric_name=args.threshold_metric,
        min_threshold=args.min_threshold,
        max_threshold=args.max_threshold,
        step=args.threshold_step,
    )
    save_threshold_sweep(run_dir / "threshold_sweep.csv", threshold_rows)
    save_json(
        run_dir / "selected_threshold.json",
        {
            "threshold_metric": args.threshold_metric,
            "selected_threshold": float(selected_threshold["threshold"]),
            "validation_metrics_at_selected_threshold": selected_threshold,
        },
    )

    default_threshold_summary = compute_threshold_metrics(truth, test_probabilities, threshold=0.5)
    tuned_threshold_summary = compute_threshold_metrics(
        truth,
        test_probabilities,
        threshold=float(selected_threshold["threshold"]),
    )
    shared_test_metrics = {
        "loss": float(keras_evaluation["loss"]),
        "keras_auc": float(keras_evaluation["auc"]),
        "roc_auc_score": float(roc_auc_score(truth, test_probabilities)),
    }
    metrics_payload = {
        "threshold_metric": args.threshold_metric,
        "selected_threshold": float(selected_threshold["threshold"]),
        "validation_metrics_at_selected_threshold": selected_threshold,
        "shared_test_metrics": shared_test_metrics,
        "default_threshold_metrics": {
            **default_threshold_summary["metrics"],
            **shared_test_metrics,
        },
        "selected_threshold_metrics": {
            **tuned_threshold_summary["metrics"],
            **shared_test_metrics,
        },
    }
    save_json(run_dir / "test_metrics.json", metrics_payload)
    save_json(
        run_dir / "classification_report.json",
        {
            "default_threshold": {
                "threshold": 0.5,
                "report": default_threshold_summary["report"],
            },
            "selected_threshold": {
                "threshold": float(selected_threshold["threshold"]),
                "report": tuned_threshold_summary["report"],
            },
        },
    )
    save_json(
        run_dir / "confusion_matrix.json",
        {
            "default_threshold": {
                "threshold": 0.5,
                "matrix": default_threshold_summary["confusion_matrix"],
            },
            "selected_threshold": {
                "threshold": float(selected_threshold["threshold"]),
                "matrix": tuned_threshold_summary["confusion_matrix"],
            },
        },
    )
    plot_confusion_matrix(
        np.array(default_threshold_summary["confusion_matrix"]),
        run_dir / "confusion_matrix_default_threshold.png",
        title="Test Confusion Matrix @ 0.50",
    )
    plot_confusion_matrix(
        np.array(tuned_threshold_summary["confusion_matrix"]),
        run_dir / "confusion_matrix.png",
        title=f"Test Confusion Matrix @ {selected_threshold['threshold']:.2f}",
    )

    print("Selected threshold:")
    print(json.dumps(selected_threshold, indent=2))
    print("Test metrics:")
    print(json.dumps(metrics_payload, indent=2))
    print("Classification report at selected threshold:")
    print(json.dumps(tuned_threshold_summary["report"], indent=2))
    print(f"Artifacts saved to: {run_dir.resolve()}")


if __name__ == "__main__":
    main()
