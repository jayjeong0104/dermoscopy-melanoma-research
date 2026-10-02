from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

import train_multitask as training
from evaluate_multitask_ph2 import build_ph2_dataset
from evaluate_ph2 import load_ph2_dataframe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare the saved blended and direct binary risk heads."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--ph2-root", type=Path, default=Path("PH2Dataset"))
    parser.add_argument("--model-name", default="best_model.keras")
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def restore_split(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["runtime_image_path"] = frame["image_path"].astype(str).map(training.runtime_path)
    frame["diagnosis_index"] = frame["diagnosis"].map(training.DIAGNOSIS_TO_INDEX).astype("int32")
    return frame


def risk_model(model: tf.keras.Model, layer_name: str) -> tf.keras.Model:
    return tf.keras.Model(
        inputs=model.inputs,
        outputs={
            "high_risk": model.get_layer(layer_name).output,
            "diagnosis": model.get_layer("diagnosis").output,
        },
        name=f"risk_ablation_{layer_name}",
    )


def prediction_range(probabilities: np.ndarray) -> dict[str, float]:
    return {
        "minimum": float(np.min(probabilities)),
        "mean": float(np.mean(probabilities)),
        "maximum": float(np.max(probabilities)),
    }


def evaluate_head(
    model: tf.keras.Model,
    val_dataset: tf.data.Dataset,
    test_dataset: tf.data.Dataset,
    ph2_dataset: tf.data.Dataset,
    val_truth: np.ndarray,
    test_truth: np.ndarray,
    ph2_truth: np.ndarray,
    minimum_recall: float,
) -> dict[str, object]:
    val_raw, _ = training.unpack_predictions(model, model.predict(val_dataset, verbose=0))
    test_raw, _ = training.unpack_predictions(model, model.predict(test_dataset, verbose=0))
    ph2_raw, _ = training.unpack_predictions(model, model.predict(ph2_dataset, verbose=0))

    temperature, validation_log_loss = training.fit_temperature(val_truth, val_raw)
    val_probabilities = training.apply_temperature(val_raw, temperature)
    test_probabilities = training.apply_temperature(test_raw, temperature)
    ph2_probabilities = training.apply_temperature(ph2_raw, temperature)
    balanced, operating, _ = training.select_thresholds(
        val_truth, val_probabilities, minimum_recall
    )
    operating_threshold = float(operating["threshold"])

    return {
        "calibration": {
            "temperature": temperature,
            "validation_log_loss": validation_log_loss,
        },
        "validation": {
            "raw_probability_metrics": training.probability_metrics(val_truth, val_raw),
            "calibrated_probability_metrics": training.probability_metrics(
                val_truth, val_probabilities
            ),
            "balanced_accuracy_threshold_metrics": balanced,
            "operating_threshold_metrics": operating,
        },
        "internal_test": {
            "raw_probability_metrics": training.probability_metrics(test_truth, test_raw),
            "calibrated_probability_metrics": training.probability_metrics(
                test_truth, test_probabilities
            ),
            "operating_threshold_metrics": training.threshold_metrics(
                test_truth, test_probabilities, operating_threshold
            ),
            "raw_probability_range": prediction_range(test_raw),
        },
        "ph2_external": {
            "raw_probability_metrics": training.probability_metrics(ph2_truth, ph2_raw),
            "calibrated_probability_metrics": training.probability_metrics(
                ph2_truth, ph2_probabilities
            ),
            "operating_threshold_metrics": training.threshold_metrics(
                ph2_truth, ph2_probabilities, operating_threshold
            ),
            "raw_probability_range": prediction_range(ph2_raw),
        },
    }


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    schema = json.loads((run_dir / "metadata_schema.json").read_text(encoding="utf-8"))
    encoder = training.MetadataEncoder(
        age_median=float(schema["age_median"]),
        feature_names=tuple(schema["feature_names"]),
    )

    val_frame = restore_split(run_dir / "val_split.csv")
    test_frame = restore_split(run_dir / "test_split.csv")
    val_dataset = training.build_dataset(
        val_frame,
        encoder,
        int(config["image_size"]),
        args.batch_size,
        False,
        int(config["data_workers"]),
        int(config["seed"]),
    )
    test_dataset = training.build_dataset(
        test_frame,
        encoder,
        int(config["image_size"]),
        args.batch_size,
        False,
        int(config["data_workers"]),
        int(config["seed"]),
    )

    ph2_frame = load_ph2_dataframe(args.ph2_root.resolve())
    ph2_dataset = build_ph2_dataset(
        ph2_frame,
        encoder,
        image_size=int(config["image_size"]),
        batch_size=args.batch_size,
    )

    saved_model = tf.keras.models.load_model(run_dir / args.model_name)
    results = {
        layer_name: evaluate_head(
            risk_model(saved_model, layer_name),
            val_dataset,
            test_dataset,
            ph2_dataset,
            val_frame["target"].to_numpy(),
            test_frame["target"].to_numpy(),
            ph2_frame["target"].to_numpy(),
            float(config["minimum_operating_recall"]),
        )
        for layer_name in ("high_risk", "direct_risk")
    }
    output_path = run_dir / "risk_head_ablation.json"
    training.save_json(output_path, results)
    print(json.dumps(results, indent=2))
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
