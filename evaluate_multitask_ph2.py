from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")

import numpy as np
import pandas as pd
import tensorflow as tf

import train_multitask as training
from evaluate_ph2 import load_ph2_dataframe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a multitask skin-lesion model on the external PH2 development set."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--ph2-root", type=Path, default=Path("PH2Dataset"))
    parser.add_argument("--model-name", default="best_model.keras")
    parser.add_argument("--batch-size", type=int, default=2)
    return parser.parse_args()


def build_ph2_dataset(
    frame: pd.DataFrame,
    encoder: training.MetadataEncoder,
    image_size: int,
    batch_size: int,
) -> tf.data.Dataset:
    metadata = frame.copy()
    metadata["age_approx"] = pd.NA
    metadata["sex"] = "unknown"
    metadata["anatom_site_general"] = "unknown"
    vectors = encoder.transform(metadata)
    dataset = tf.data.Dataset.from_tensor_slices(
        (frame["image_path"].astype(str).to_numpy(), vectors, frame["target"].to_numpy())
    )

    def decode(
        path: tf.Tensor, metadata_vector: tf.Tensor, target: tf.Tensor
    ) -> tuple[dict[str, tf.Tensor], dict[str, tf.Tensor]]:
        image = tf.io.decode_bmp(tf.io.read_file(path), channels=3)
        image = training.resize_and_normalize_image(image, image_size)
        image.set_shape([image_size, image_size, 3])
        metadata_vector.set_shape([len(encoder.feature_names)])
        labels = {
            "high_risk": tf.reshape(tf.cast(target, tf.float32), [1]),
            "diagnosis": tf.zeros([len(training.DIAGNOSES)], dtype=tf.float32),
        }
        return {"image": image, "metadata": metadata_vector}, labels

    return dataset.map(decode, num_parallel_calls=1).batch(batch_size).prefetch(1)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    model_path = run_dir / args.model_name
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    finalization = json.loads(
        (run_dir / "finalization_status.json").read_text(encoding="utf-8")
    )
    if finalization.get("status") != "complete":
        raise RuntimeError("Refusing external evaluation of a run that is not finalized.")
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    schema = json.loads((run_dir / "metadata_schema.json").read_text(encoding="utf-8"))
    calibration = json.loads((run_dir / "calibration.json").read_text(encoding="utf-8"))
    thresholds = json.loads((run_dir / "selected_threshold.json").read_text(encoding="utf-8"))
    blend_path = run_dir / "probability_blend.json"
    blend = (
        json.loads(blend_path.read_text(encoding="utf-8"))
        if blend_path.exists()
        else {"direct_weight": 1.0}
    )
    encoder = training.MetadataEncoder(
        age_median=float(schema["age_median"]),
        feature_names=tuple(schema["feature_names"]),
    )

    frame = load_ph2_dataframe(args.ph2_root.resolve())
    dataset = build_ph2_dataset(
        frame,
        encoder,
        image_size=int(config["image_size"]),
        batch_size=args.batch_size,
    )
    model = tf.keras.models.load_model(model_path, compile=False)
    tta_transforms = int(calibration.get("tta_transforms", config.get("tta_transforms", 1)))
    tta_scheme = calibration.get("tta_scheme", config.get("tta_scheme", "legacy_v1"))
    direct_probabilities, diagnosis_probabilities = training.predict_with_tta(
        model, dataset, tta_transforms, tta_scheme
    )
    binary_target = str(config.get("binary_target", "high_risk"))
    diagnosis_target_probabilities = training.diagnosis_target_probabilities(
        diagnosis_probabilities, binary_target
    )
    direct_weight = float(blend["direct_weight"])
    raw_probabilities = (
        direct_weight * direct_probabilities
        + (1.0 - direct_weight) * diagnosis_target_probabilities
    ).astype("float32")
    probabilities = training.apply_temperature(
        raw_probabilities, float(calibration["temperature"])
    )
    truth = frame["target"].to_numpy()
    threshold = float(thresholds["operating_threshold"])
    operating_metrics = training.threshold_metrics(truth, probabilities, threshold)
    operating_metrics["confidence_intervals_95"] = training.operating_confidence_intervals(
        operating_metrics
    )
    metrics = {
        "dataset": "PH2",
        "model": args.model_name,
        "binary_target": binary_target,
        "tta_transforms": tta_transforms,
        "tta_scheme": tta_scheme,
        "direct_weight": direct_weight,
        "sample_count": int(len(frame)),
        "class_counts": {
            "benign": int((truth == 0).sum()),
            "melanoma": int((truth == 1).sum()),
        },
        "raw_probability_metrics": training.probability_metrics(truth, raw_probabilities),
        "calibrated_probability_metrics": training.probability_metrics(truth, probabilities),
        "default_threshold_metrics": training.threshold_metrics(truth, probabilities, 0.5),
        "operating_threshold_metrics": operating_metrics,
    }
    training.save_json(run_dir / "ph2_external_metrics.json", metrics)

    predictions = frame.copy()
    predictions["probability_direct_raw"] = direct_probabilities
    predictions["probability_diagnosis_target_raw"] = diagnosis_target_probabilities
    predictions[f"probability_{binary_target}_raw"] = raw_probabilities
    predictions[f"probability_{binary_target}"] = probabilities
    predictions["prediction"] = (probabilities >= threshold).astype("int32")
    diagnosis_indices = diagnosis_probabilities.argmax(axis=1)
    predictions["diagnosis_prediction"] = [
        training.DIAGNOSES[index] for index in diagnosis_indices
    ]
    predictions["diagnosis_confidence"] = diagnosis_probabilities.max(axis=1)
    predictions.to_csv(run_dir / "ph2_external_predictions.csv", index=False)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
