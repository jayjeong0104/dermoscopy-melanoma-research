from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
KERAS_HOME = PROJECT_ROOT / ".keras"
KERAS_HOME.mkdir(exist_ok=True)
os.environ.setdefault("KERAS_HOME", str(KERAS_HOME))
MATPLOTLIB_CONFIG = Path(tempfile.gettempdir()) / "skin_cancer_transfer_mplconfig"
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


REQUIRED_COLUMNS = {
    "source",
    "image_id",
    "lesion_id",
    "image_path",
    "diagnosis",
    "target",
}


@dataclass(frozen=True)
class SplitFrames:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a leakage-safe EfficientNet model on a combined lesion manifest."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data") / "combined_ham_bcn" / "combined_manifest.csv",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts") / "transfer_learning",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--head-epochs", type=int, default=6)
    parser.add_argument("--fine-tune-epochs", type=int, default=8)
    parser.add_argument("--fine-tune-layers", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--fine-tune-learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--dropout-rate", type=float, default=0.40)
    parser.add_argument("--label-smoothing", type=float, default=0.02)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--sample-fraction", type=float, default=1.0)
    parser.add_argument("--minimum-screening-recall", type=float, default=0.80)
    parser.add_argument("--data-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--weights",
        choices=("imagenet", "none"),
        default="imagenet",
        help="Use ImageNet transfer weights or random initialization.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")
    if args.image_size < 96:
        raise ValueError("--image-size must be at least 96 for EfficientNetB0.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    if args.head_epochs < 1:
        raise ValueError("--head-epochs must be at least 1.")
    if args.fine_tune_epochs < 0:
        raise ValueError("--fine-tune-epochs cannot be negative.")
    if args.fine_tune_layers < 1:
        raise ValueError("--fine-tune-layers must be at least 1.")
    if args.learning_rate <= 0 or args.fine_tune_learning_rate <= 0:
        raise ValueError("Learning rates must be positive.")
    if not 0 <= args.label_smoothing < 0.5:
        raise ValueError("--label-smoothing must be in [0, 0.5).")
    if not 0 < args.val_size < 1 or not 0 < args.test_size < 1:
        raise ValueError("Validation and test fractions must be between 0 and 1.")
    if args.val_size + args.test_size >= 1:
        raise ValueError("Validation and test fractions must sum to less than 1.")
    if not 0 < args.sample_fraction <= 1:
        raise ValueError("--sample-fraction must be between 0 and 1.")
    if not 0 < args.minimum_screening_recall <= 1:
        raise ValueError("--minimum-screening-recall must be between 0 and 1.")
    if args.data_workers < 1:
        raise ValueError("--data-workers must be at least 1.")


def set_random_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def load_manifest(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    frame = frame.copy()
    frame["target"] = frame["target"].astype("int32")
    if set(frame["target"].unique()) != {0, 1}:
        raise ValueError("Manifest target must contain both binary values 0 and 1.")
    missing_paths = [path for path in frame["image_path"].astype(str) if not Path(path).exists()]
    if missing_paths:
        raise FileNotFoundError(
            f"Manifest references {len(missing_paths)} missing images. Sample: {missing_paths[:3]}"
        )
    conflicts = frame.groupby("lesion_id")["target"].nunique()
    if (conflicts > 1).any():
        raise ValueError("At least one lesion has conflicting target labels.")
    return frame


def maybe_downsample(frame: pd.DataFrame, fraction: float, seed: int) -> pd.DataFrame:
    if fraction >= 1.0:
        return frame
    lesion_level = frame[["lesion_id", "source", "target"]].drop_duplicates()
    sampled = []
    for index, (_, group) in enumerate(lesion_level.groupby(["source", "target"])):
        take = max(3, int(round(len(group) * fraction)))
        take = min(take, len(group))
        sampled.append(group.sample(n=take, random_state=seed + index))
    selected = pd.concat(sampled, ignore_index=True)["lesion_id"]
    return frame[frame["lesion_id"].isin(selected)].copy()


def split_by_lesion_and_source(
    frame: pd.DataFrame,
    val_size: float,
    test_size: float,
    seed: int,
) -> SplitFrames:
    lesion_level = frame[["lesion_id", "source", "target"]].drop_duplicates().copy()
    lesion_level["stratum"] = (
        lesion_level["source"].astype(str) + "::" + lesion_level["target"].astype(str)
    )
    stratum_counts = lesion_level["stratum"].value_counts()
    if stratum_counts.min() < 7:
        raise ValueError(
            "Each source/target stratum needs at least 7 lesions for train/val/test splitting."
        )

    train_lesions, temporary = train_test_split(
        lesion_level,
        test_size=val_size + test_size,
        stratify=lesion_level["stratum"],
        random_state=seed,
    )
    test_ratio = test_size / (val_size + test_size)
    val_lesions, test_lesions = train_test_split(
        temporary,
        test_size=test_ratio,
        stratify=temporary["stratum"],
        random_state=seed,
    )

    return SplitFrames(
        train=frame[frame["lesion_id"].isin(train_lesions["lesion_id"])].copy(),
        val=frame[frame["lesion_id"].isin(val_lesions["lesion_id"])].copy(),
        test=frame[frame["lesion_id"].isin(test_lesions["lesion_id"])].copy(),
    )


def build_dataset(
    frame: pd.DataFrame,
    image_size: int,
    batch_size: int,
    training: bool,
    data_workers: int,
    seed: int,
) -> tf.data.Dataset:
    paths = frame["image_path"].astype(str).to_numpy()
    labels = frame["target"].astype("float32").to_numpy()
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        dataset = dataset.shuffle(
            buffer_size=len(frame), seed=seed, reshuffle_each_iteration=True
        )

    def decode(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        image = tf.io.read_file(path)
        image = tf.io.decode_jpeg(image, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, [image_size, image_size], antialias=True)
        image.set_shape([image_size, image_size, 3])
        return image, label

    options = tf.data.Options()
    options.deterministic = not training
    dataset = dataset.with_options(options)
    dataset = dataset.map(decode, num_parallel_calls=data_workers)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(2)
    return dataset


def build_model(
    image_size: int,
    dropout_rate: float,
    weights: str | None,
) -> tf.keras.Model:
    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.RandomRotation(0.15, fill_mode="reflect"),
            tf.keras.layers.RandomTranslation(0.05, 0.05, fill_mode="reflect"),
            tf.keras.layers.RandomZoom(0.15, fill_mode="reflect"),
            tf.keras.layers.RandomContrast(0.15),
        ],
        name="augmentation",
    )
    backbone = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights=weights,
        input_shape=(image_size, image_size, 3),
    )
    backbone.trainable = False

    inputs = tf.keras.Input(shape=(image_size, image_size, 3), name="image")
    x = augmentation(inputs)
    # The shared loaders emit [0, 1]; EfficientNet's built-in preprocessing expects [0, 255].
    x = tf.keras.layers.Rescaling(255.0, name="efficientnet_input_scale")(x)
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_pool")(x)
    x = tf.keras.layers.BatchNormalization(name="head_batch_norm")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="head_dropout_1")(x)
    x = tf.keras.layers.Dense(
        256,
        activation="swish",
        kernel_regularizer=tf.keras.regularizers.l2(1e-5),
        name="head_dense",
    )(x)
    x = tf.keras.layers.Dropout(dropout_rate * 0.6, name="head_dropout_2")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="high_risk")(x)
    return tf.keras.Model(inputs, outputs, name="skin_lesion_efficientnetb0")


def compile_model(
    model: tf.keras.Model,
    learning_rate: float,
    weight_decay: float,
    label_smoothing: float,
) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        ),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=label_smoothing),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc", curve="ROC"),
            tf.keras.metrics.AUC(name="pr_auc", curve="PR"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )


def class_weights(labels: pd.Series) -> dict[int, float]:
    classes = np.array([0, 1])
    weights = compute_class_weight("balanced", classes=classes, y=labels.to_numpy())
    return {int(label): float(weight) for label, weight in zip(classes, weights)}


def training_callbacks(run_dir: Path, stage: str, patience: int) -> list[tf.keras.callbacks.Callback]:
    return [
        tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.BackupAndRestore(
            backup_dir=str(run_dir / f"backup_{stage}"),
            delete_checkpoint=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(run_dir / f"best_{stage}_model.keras"),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc",
            mode="max",
            factor=0.4,
            patience=max(1, patience // 2),
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(run_dir / f"training_{stage}.csv")),
    ]


def make_backbone_trainable(model: tf.keras.Model, trainable_layers: int) -> int:
    backbone = model.get_layer("efficientnetb0")
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


def create_run_dir(output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(exist_ok=False)
    return run_dir


def save_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def split_summary(frame: pd.DataFrame) -> dict[str, object]:
    source_target = frame.groupby(["source", "target_name"]).size()
    return {
        "images": int(len(frame)),
        "lesions": int(frame["lesion_id"].nunique()),
        "source_counts": {
            str(key): int(value) for key, value in frame["source"].value_counts().items()
        },
        "target_counts": {
            str(key): int(value) for key, value in frame["target_name"].value_counts().items()
        },
        "source_target_counts": [
            {"source": str(key[0]), "target": str(key[1]), "count": int(value)}
            for key, value in source_target.items()
        ],
        "diagnosis_counts": {
            str(key): int(value)
            for key, value in frame["diagnosis"].value_counts().sort_index().items()
        },
    }


def threshold_metrics(
    truth: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, object]:
    predictions = (probabilities >= threshold).astype("int32")
    matrix = confusion_matrix(truth, predictions, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(truth, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predictions)),
        "precision": float(precision_score(truth, predictions, zero_division=0)),
        "recall": float(recall_score(truth, predictions, zero_division=0)),
        "specificity": float(specificity),
        "f1": float(f1_score(truth, predictions, zero_division=0)),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "support": int(len(truth)),
        "confusion_matrix": matrix.tolist(),
    }


def select_thresholds(
    truth: np.ndarray,
    probabilities: np.ndarray,
    minimum_screening_recall: float,
) -> tuple[dict[str, object], dict[str, object], list[dict[str, object]]]:
    rows = [
        threshold_metrics(truth, probabilities, float(threshold))
        for threshold in np.round(np.arange(0.05, 0.951, 0.01), 2)
    ]
    balanced = max(
        rows,
        key=lambda row: (
            row["balanced_accuracy"],
            row["recall"],
            row["precision"],
        ),
    )
    screening_candidates = [row for row in rows if row["recall"] >= minimum_screening_recall]
    if screening_candidates:
        screening = max(
            screening_candidates,
            key=lambda row: (row["specificity"], row["precision"], row["balanced_accuracy"]),
        )
    else:
        screening = max(rows, key=lambda row: (row["recall"], row["specificity"]))
    return balanced, screening, rows


def plot_history(history_frames: list[pd.DataFrame], path: Path) -> None:
    history = pd.concat(history_frames, ignore_index=True)
    metrics = [("loss", "val_loss"), ("auc", "val_auc"), ("pr_auc", "val_pr_auc")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for axis, (train_metric, val_metric) in zip(axes, metrics):
        axis.plot(history.index, history[train_metric], label=train_metric)
        axis.plot(history.index, history[val_metric], label=val_metric)
        axis.set_xlabel("Epoch")
        axis.set_title(train_metric.replace("_", " ").title())
        axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_confusion(matrix: list[list[int]], path: Path, title: str) -> None:
    values = np.asarray(matrix)
    fig, axis = plt.subplots(figsize=(5, 4))
    axis.imshow(values, cmap="Blues")
    axis.set_title(title)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    axis.set_xticks([0, 1], labels=["benign", "high risk"])
    axis.set_yticks([0, 1], labels=["benign", "high risk"])
    for row in range(2):
        for column in range(2):
            axis.text(column, row, int(values[row, column]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def evaluate_candidate(path: Path, dataset: tf.data.Dataset) -> tuple[tf.keras.Model, dict[str, float]]:
    model = tf.keras.models.load_model(path)
    metrics = {key: float(value) for key, value in model.evaluate(dataset, return_dict=True, verbose=0).items()}
    return model, metrics


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_random_seeds(args.seed)
    frame = maybe_downsample(load_manifest(args.manifest), args.sample_fraction, args.seed)
    splits = split_by_lesion_and_source(frame, args.val_size, args.test_size, args.seed)
    run_dir = create_run_dir(args.output_root)

    config = vars(args).copy()
    config = {key: str(value.resolve()) if isinstance(value, Path) else value for key, value in config.items()}
    config["tensorflow_version"] = tf.__version__
    config["physical_devices"] = [str(device) for device in tf.config.list_physical_devices()]
    save_json(run_dir / "run_config.json", config)
    summary = {
        "full_dataset": split_summary(frame),
        "train": split_summary(splits.train),
        "validation": split_summary(splits.val),
        "test": split_summary(splits.test),
    }
    save_json(run_dir / "split_summary.json", summary)
    for name, split in (("train", splits.train), ("val", splits.val), ("test", splits.test)):
        split.to_csv(run_dir / f"{name}_split.csv", index=False)

    print("Split summary:")
    print(json.dumps(summary, indent=2))
    weights = class_weights(splits.train["target"])
    print("Class weights:", weights)

    train_ds = build_dataset(
        splits.train,
        args.image_size,
        args.batch_size,
        True,
        args.data_workers,
        args.seed,
    )
    val_ds = build_dataset(
        splits.val,
        args.image_size,
        args.batch_size,
        False,
        args.data_workers,
        args.seed,
    )
    test_ds = build_dataset(
        splits.test,
        args.image_size,
        args.batch_size,
        False,
        args.data_workers,
        args.seed,
    )

    pretrained_weights = None if args.weights == "none" else args.weights
    model = build_model(args.image_size, args.dropout_rate, pretrained_weights)
    compile_model(model, args.learning_rate, args.weight_decay, args.label_smoothing)
    model.summary()
    head_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.head_epochs,
        class_weight=weights,
        callbacks=training_callbacks(run_dir, "head", patience=3),
        verbose=1,
    )
    history_frames = [pd.DataFrame(head_history.history).assign(stage="head")]

    candidate_paths = [run_dir / "best_head_model.keras"]
    if args.fine_tune_epochs:
        model = tf.keras.models.load_model(candidate_paths[0])
        trainable_layers = make_backbone_trainable(model, args.fine_tune_layers)
        print(f"Fine-tuning {trainable_layers} non-BatchNorm backbone layers.")
        compile_model(
            model,
            args.fine_tune_learning_rate,
            args.weight_decay,
            args.label_smoothing,
        )
        fine_history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.fine_tune_epochs,
            class_weight=weights,
            callbacks=training_callbacks(run_dir, "finetune", patience=4),
            verbose=1,
        )
        history_frames.append(pd.DataFrame(fine_history.history).assign(stage="finetune"))
        model.save(run_dir / "final_finetuned_model.keras")
        candidate_paths.append(run_dir / "best_finetune_model.keras")

    candidate_metrics: dict[str, dict[str, float]] = {}
    candidates: list[tuple[tf.keras.Model, Path, dict[str, float]]] = []
    for path in candidate_paths:
        candidate_model, metrics = evaluate_candidate(path, val_ds)
        candidate_metrics[path.name] = metrics
        candidates.append((candidate_model, path, metrics))
    model, selected_model_path, _ = max(candidates, key=lambda candidate: candidate[2]["auc"])
    model.save(run_dir / "best_model.keras")
    save_json(
        run_dir / "model_selection.json",
        {"selected_candidate": selected_model_path.name, "validation_metrics": candidate_metrics},
    )

    history = pd.concat(history_frames, ignore_index=True)
    history.to_csv(run_dir / "training_history.csv", index=False)
    plot_history(history_frames, run_dir / "training_curves.png")

    val_probabilities = model.predict(val_ds, verbose=1).ravel()
    test_probabilities = model.predict(test_ds, verbose=1).ravel()
    val_truth = splits.val["target"].to_numpy()
    test_truth = splits.test["target"].to_numpy()
    balanced_threshold, screening_threshold, threshold_rows = select_thresholds(
        val_truth,
        val_probabilities,
        args.minimum_screening_recall,
    )
    pd.DataFrame(threshold_rows).drop(columns=["confusion_matrix"]).to_csv(
        run_dir / "threshold_sweep.csv", index=False
    )
    save_json(
        run_dir / "selected_threshold.json",
        {
            "selected_threshold": balanced_threshold["threshold"],
            "selection_metric": "balanced_accuracy",
            "validation_metrics": balanced_threshold,
            "screening_threshold": screening_threshold["threshold"],
            "minimum_screening_recall": args.minimum_screening_recall,
            "screening_validation_metrics": screening_threshold,
        },
    )

    test_evaluation = {
        key: float(value)
        for key, value in model.evaluate(test_ds, return_dict=True, verbose=1).items()
    }
    test_evaluation["roc_auc_score"] = float(roc_auc_score(test_truth, test_probabilities))
    test_evaluation["average_precision_score"] = float(
        average_precision_score(test_truth, test_probabilities)
    )
    threshold_results = {
        "default_0_5": threshold_metrics(test_truth, test_probabilities, 0.5),
        "balanced_accuracy": threshold_metrics(
            test_truth, test_probabilities, float(balanced_threshold["threshold"])
        ),
        "screening": threshold_metrics(
            test_truth, test_probabilities, float(screening_threshold["threshold"])
        ),
    }
    save_json(
        run_dir / "test_metrics.json",
        {"probability_metrics": test_evaluation, "threshold_metrics": threshold_results},
    )

    selected_predictions = (
        test_probabilities >= float(balanced_threshold["threshold"])
    ).astype("int32")
    save_json(
        run_dir / "classification_report.json",
        classification_report(
            test_truth,
            selected_predictions,
            target_names=["benign", "high_risk"],
            output_dict=True,
            zero_division=0,
        ),
    )
    predictions = splits.test.copy()
    predictions["probability_high_risk"] = test_probabilities
    predictions["prediction"] = selected_predictions
    predictions.to_csv(run_dir / "test_predictions.csv", index=False)

    per_source = {}
    for source, group in predictions.groupby("source"):
        indices = group.index.to_numpy()
        positions = splits.test.index.get_indexer(indices)
        per_source[str(source)] = threshold_metrics(
            group["target"].to_numpy(),
            test_probabilities[positions],
            float(balanced_threshold["threshold"]),
        )
    save_json(run_dir / "test_metrics_by_source.json", per_source)
    plot_confusion(
        threshold_results["balanced_accuracy"]["confusion_matrix"],
        run_dir / "confusion_matrix.png",
        f"Test Confusion Matrix @ {balanced_threshold['threshold']:.2f}",
    )

    print("Selected model:", selected_model_path.name)
    print("Test metrics:")
    print(json.dumps({"probability_metrics": test_evaluation, "threshold_metrics": threshold_results}, indent=2))
    print("Artifacts saved to:", run_dir.resolve())


if __name__ == "__main__":
    main()
