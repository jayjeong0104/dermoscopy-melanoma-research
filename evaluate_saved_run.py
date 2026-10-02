from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import train
import tensorflow as tf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-evaluate a saved training run with validation-based threshold tuning."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Path to an artifacts/run_* directory created by train.py.",
    )
    parser.add_argument(
        "--model-name",
        default="best_model.keras",
        choices=["best_model.keras", "final_model.keras"],
        help="Saved model file inside the run directory to evaluate.",
    )
    parser.add_argument(
        "--threshold-metric",
        choices=train.THRESHOLD_METRICS,
        default="accuracy",
        help="Validation metric used to choose the decision threshold.",
    )
    parser.add_argument(
        "--min-threshold",
        type=float,
        default=0.10,
        help="Minimum threshold included in the validation sweep.",
    )
    parser.add_argument(
        "--max-threshold",
        type=float,
        default=0.90,
        help="Maximum threshold included in the validation sweep.",
    )
    parser.add_argument(
        "--threshold-step",
        type=float,
        default=0.01,
        help="Threshold sweep step size.",
    )
    return parser.parse_args()


def load_run_config(run_dir: Path) -> dict:
    config_path = run_dir / "run_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Run config not found: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def load_split(run_dir: Path, split_name: str) -> pd.DataFrame:
    split_path = run_dir / f"{split_name}_split.csv"
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")
    return pd.read_csv(split_path)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    run_config = load_run_config(run_dir)
    model_path = run_dir / args.model_name
    if not model_path.exists():
        raise FileNotFoundError(f"Saved model not found: {model_path}")

    image_size = int(run_config["image_size"])
    batch_size = int(run_config["batch_size"])
    val_frame = load_split(run_dir, "val")
    test_frame = load_split(run_dir, "test")

    val_ds = train.build_dataset(val_frame, image_size, batch_size, training=False)
    test_ds = train.build_dataset(test_frame, image_size, batch_size, training=False)

    model = tf.keras.models.load_model(model_path)
    keras_eval = model.evaluate(test_ds, return_dict=True, verbose=0)
    val_probabilities = train.predict_probabilities(model, val_ds, verbose=0)
    test_probabilities = train.predict_probabilities(model, test_ds, verbose=0)

    selected_threshold, sweep_rows = train.sweep_thresholds(
        truth=val_frame["target"].to_numpy(),
        probabilities=val_probabilities,
        metric_name=args.threshold_metric,
        min_threshold=args.min_threshold,
        max_threshold=args.max_threshold,
        step=args.threshold_step,
    )
    default_summary = train.compute_threshold_metrics(
        test_frame["target"].to_numpy(),
        test_probabilities,
        threshold=0.5,
    )
    tuned_summary = train.compute_threshold_metrics(
        test_frame["target"].to_numpy(),
        test_probabilities,
        threshold=float(selected_threshold["threshold"]),
    )
    shared_test_metrics = {
        "loss": float(keras_eval["loss"]),
        "keras_auc": float(keras_eval["auc"]),
        "roc_auc_score": float(train.roc_auc_score(test_frame["target"].to_numpy(), test_probabilities)),
    }

    stem = model_path.stem
    train.save_threshold_sweep(run_dir / f"{stem}_threshold_sweep.csv", sweep_rows)
    payload = {
        "model_name": args.model_name,
        "threshold_metric": args.threshold_metric,
        "selected_threshold": float(selected_threshold["threshold"]),
        "validation_metrics_at_selected_threshold": selected_threshold,
        "default_threshold_metrics": {
            **default_summary["metrics"],
            **shared_test_metrics,
        },
        "selected_threshold_metrics": {
            **tuned_summary["metrics"],
            **shared_test_metrics,
        },
    }
    train.save_json(run_dir / f"{stem}_evaluation.json", payload)
    train.save_json(
        run_dir / f"{stem}_classification_report.json",
        {
            "default_threshold": {
                "threshold": 0.5,
                "report": default_summary["report"],
            },
            "selected_threshold": {
                "threshold": float(selected_threshold["threshold"]),
                "report": tuned_summary["report"],
            },
        },
    )
    train.save_json(
        run_dir / f"{stem}_confusion_matrix.json",
        {
            "default_threshold": {
                "threshold": 0.5,
                "matrix": default_summary["confusion_matrix"],
            },
            "selected_threshold": {
                "threshold": float(selected_threshold["threshold"]),
                "matrix": tuned_summary["confusion_matrix"],
            },
        },
    )

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
