from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report

import train_multitask as training


CANDIDATE_NAMES = ("best_head_model.keras", "best_finetune_model.keras")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Finalize a completed multitask run in a fresh, low-memory process. "
            "Prediction stages are cached so an interrupted evaluation can resume."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--data-workers", type=int, default=1)
    parser.add_argument("--prefetch-batches", type=int, default=1)
    parser.add_argument(
        "--baseline-run-dir", type=Path,
        help="Finalized incumbent to compare on validation only; defaults to the warm-start run.",
    )
    parser.add_argument(
        "--tta-transforms",
        type=int,
        choices=(1, 4, 8),
        help="Override the run configuration for final inference.",
    )
    parser.add_argument(
        "--force-predictions",
        action="store_true",
        help="Ignore valid prediction caches and run inference again.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_json_atomic(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def release_runtime_memory() -> None:
    tf.keras.backend.clear_session()
    gc.collect()
    training.trim_host_memory()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_prediction_arrays(
    direct: np.ndarray,
    diagnosis: np.ndarray,
    expected_rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    direct = np.asarray(direct, dtype="float32").reshape(-1)
    diagnosis = np.asarray(diagnosis, dtype="float32")
    if direct.shape != (expected_rows,):
        raise ValueError(
            f"Risk prediction shape {direct.shape} does not match ({expected_rows},)."
        )
    expected_diagnosis_shape = (expected_rows, len(training.DIAGNOSES))
    if diagnosis.shape != expected_diagnosis_shape:
        raise ValueError(
            f"Diagnosis prediction shape {diagnosis.shape} does not match "
            f"{expected_diagnosis_shape}."
        )
    if not np.isfinite(direct).all() or not np.isfinite(diagnosis).all():
        raise ValueError("Predictions contain NaN or infinite values.")
    if (direct < 0).any() or (direct > 1).any():
        raise ValueError("Risk predictions fall outside [0, 1].")
    if (diagnosis < 0).any() or (diagnosis > 1).any():
        raise ValueError("Diagnosis predictions fall outside [0, 1].")
    return direct, diagnosis


def write_prediction_cache(
    path: Path,
    direct: np.ndarray,
    diagnosis: np.ndarray,
    metadata: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, direct=direct, diagnosis=diagnosis)
    temporary.replace(path)
    save_json_atomic(path.with_suffix(".json"), metadata)


def read_prediction_cache(
    path: Path,
    expected_metadata: dict[str, object],
    expected_rows: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    metadata_path = path.with_suffix(".json")
    if not path.exists() or not metadata_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if any(metadata.get(key) != value for key, value in expected_metadata.items()):
        return None
    with np.load(path, allow_pickle=False) as cached:
        return validate_prediction_arrays(
            cached["direct"], cached["diagnosis"], expected_rows
        )


def build_inference_dataset(
    frame: pd.DataFrame,
    encoder: training.MetadataEncoder,
    image_size: int,
    batch_size: int,
    data_workers: int,
    prefetch_batches: int,
    seed: int,
) -> tf.data.Dataset:
    return training.build_dataset(
        frame,
        encoder,
        image_size,
        batch_size,
        False,
        data_workers,
        seed,
        prefetch_batches=prefetch_batches,
        mask_metadata=True,
    )


def prediction_input_fingerprint(frame: pd.DataFrame, encoder: training.MetadataEncoder) -> str:
    columns = ["source", "image_id", "sha256", "runtime_image_path"]
    if frame[columns].isna().any().any():
        raise ValueError("Prediction cache inputs contain missing identities.")
    digest = hashlib.sha256(frame[columns].astype("string").to_csv(index=False).encode("utf-8"))
    digest.update(json.dumps(asdict(encoder), sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


def predict_checkpoint(
    model_path: Path,
    frame: pd.DataFrame,
    encoder: training.MetadataEncoder,
    image_size: int,
    batch_size: int,
    data_workers: int,
    prefetch_batches: int,
    seed: int,
    transforms: int,
    cache_path: Path,
    force: bool,
    tta_scheme: str = "legacy_v1",
) -> tuple[np.ndarray, np.ndarray]:
    if transforms not in (1, 4, 8) or tta_scheme not in ("legacy_v1", "d4_v2"):
        raise ValueError("Invalid TTA count or scheme.")
    expected_metadata = {
        "cache_version": 3,
        "inference_batch_size": int(batch_size),
        "tensorflow_version": tf.__version__,
        "input_sha256": prediction_input_fingerprint(frame, encoder),
        "tta_scheme": tta_scheme,
        "model_sha256": sha256_file(model_path),
        "row_count": int(len(frame)),
        "image_size": int(image_size),
        "transforms": int(transforms),
        "masked_metadata": True,
    }
    if not force:
        cached = read_prediction_cache(cache_path, expected_metadata, len(frame))
        if cached is not None:
            print(f"Using prediction cache: {cache_path}")
            return cached

    views: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    missing_views: list[int] = []
    for transform_index in range(transforms):
        view_path = cache_path.with_name(
            f"{cache_path.stem}.view_{transform_index}{cache_path.suffix}"
        )
        view_metadata = {
            **expected_metadata,
            "transform_index": transform_index,
        }
        cached_view = None
        if not force:
            cached_view = read_prediction_cache(
                view_path, view_metadata, len(frame)
            )
        if cached_view is None:
            missing_views.append(transform_index)
        else:
            views[transform_index] = cached_view

    if missing_views:
        release_runtime_memory()
        dataset = build_inference_dataset(
            frame,
            encoder,
            image_size,
            batch_size,
            data_workers,
            prefetch_batches,
            seed,
        )
        model = tf.keras.models.load_model(model_path, compile=False)
        print(
            f"Predicting {len(frame)} images with {model_path.name}, "
            f"{len(missing_views)}/{transforms} uncached view(s), batch size {batch_size}."
        )
        for transform_index in missing_views:
            transformed = training.transform_tta_dataset(dataset, transform_index, tta_scheme)
            direct_view, diagnosis_view = training.unpack_predictions(
                model, model.predict(transformed, verbose=0)
            )
            direct_view, diagnosis_view = validate_prediction_arrays(
                direct_view, diagnosis_view, len(frame)
            )
            view_path = cache_path.with_name(
                f"{cache_path.stem}.view_{transform_index}{cache_path.suffix}"
            )
            view_metadata = {
                **expected_metadata,
                "transform_index": transform_index,
            }
            write_prediction_cache(
                view_path, direct_view, diagnosis_view, view_metadata
            )
            views[transform_index] = (direct_view, diagnosis_view)
            print(f"Cached TTA view {transform_index + 1}/{transforms}: {view_path.name}")
        del model, dataset
        release_runtime_memory()

    direct_sum: np.ndarray | None = None
    diagnosis_sum: np.ndarray | None = None
    for transform_index in range(transforms):
        direct_view, diagnosis_view = views[transform_index]
        if direct_sum is None or diagnosis_sum is None:
            direct_sum = direct_view.astype("float64")
            diagnosis_sum = diagnosis_view.astype("float64")
        else:
            direct_sum += direct_view
            diagnosis_sum += diagnosis_view
    if direct_sum is None or diagnosis_sum is None:
        raise RuntimeError("Prediction produced no TTA views.")
    direct = (direct_sum / transforms).astype("float32")
    diagnosis = (diagnosis_sum / transforms).astype("float32")
    write_prediction_cache(cache_path, direct, diagnosis, expected_metadata)
    return direct, diagnosis


def load_run_split(run_dir: Path, name: str, binary_target: str) -> pd.DataFrame:
    frame = training.load_manifest(run_dir / f"{name}_split.csv", binary_target)
    return training.configure_binary_target(frame, binary_target)


def candidate_metrics(
    frame: pd.DataFrame,
    direct: np.ndarray,
    diagnosis: np.ndarray,
    binary_target: str,
) -> dict[str, float]:
    lesions, lesion_diagnosis = training.aggregate_predictions_by_lesion(
        frame, direct, diagnosis
    )
    if lesion_diagnosis is None:
        raise RuntimeError("Candidate diagnosis probabilities were not aggregated.")
    image_truth = frame["target"].to_numpy(dtype="int32")
    lesion_truth = lesions["target"].to_numpy(dtype="int32")
    lesion_direct = lesions["probability"].to_numpy(dtype="float32")
    lesion_diagnosis_target = training.diagnosis_target_probabilities(
        lesion_diagnosis, binary_target
    )
    blend, _, _ = training.select_probability_blend(
        lesion_truth, lesion_direct, lesion_diagnosis_target
    )
    result = {
        f"image_{key}": value
        for key, value in training.probability_metrics(image_truth, direct).items()
    }
    result.update(
        {
            f"lesion_{key}": value
            for key, value in training.probability_metrics(
                lesion_truth, lesion_direct
            ).items()
        }
    )
    result.update({f"lesion_blended_{key}": value for key, value in blend.items()})
    return result


def add_confidence_intervals(metrics: dict[str, object]) -> dict[str, object]:
    result = dict(metrics)
    result["confidence_intervals_95"] = training.operating_confidence_intervals(result)
    return result


def cluster_bootstrap_confidence_intervals(
    truth: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    cluster_ids: np.ndarray,
    iterations: int = 1000,
    seed: int = 42,
) -> dict[str, object]:
    truth = np.asarray(truth, dtype="int32").reshape(-1)
    probabilities = np.asarray(probabilities, dtype="float32").reshape(-1)
    cluster_ids = np.asarray(cluster_ids).reshape(-1)
    if not (len(truth) == len(probabilities) == len(cluster_ids)):
        raise ValueError("Truth, probabilities, and cluster IDs must have equal lengths.")
    if iterations < 100:
        raise ValueError("Cluster bootstrap requires at least 100 iterations.")
    unique_clusters = pd.unique(cluster_ids)
    if len(unique_clusters) < 2:
        raise ValueError("Cluster bootstrap requires at least two independent groups.")
    cluster_indices = {
        cluster: np.flatnonzero(cluster_ids == cluster) for cluster in unique_clusters
    }
    metric_names = (
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "specificity",
        "roc_auc",
        "pr_auc",
    )
    samples: dict[str, list[float]] = {name: [] for name in metric_names}
    rng = np.random.default_rng(seed)
    valid_iterations = 0
    for _ in range(iterations):
        selected_clusters = rng.choice(
            unique_clusters, size=len(unique_clusters), replace=True
        )
        indices = np.concatenate(
            [cluster_indices[cluster] for cluster in selected_clusters]
        )
        sampled_truth = truth[indices]
        if np.unique(sampled_truth).size != 2:
            continue
        sampled_probabilities = probabilities[indices]
        threshold_result = training.threshold_metrics(
            sampled_truth, sampled_probabilities, threshold
        )
        probability_result = training.probability_metrics(
            sampled_truth, sampled_probabilities
        )
        for name in metric_names[:5]:
            samples[name].append(float(threshold_result[name]))
        samples["roc_auc"].append(float(probability_result["roc_auc"]))
        samples["pr_auc"].append(float(probability_result["pr_auc"]))
        valid_iterations += 1
    if valid_iterations < max(100, int(iterations * 0.8)):
        raise RuntimeError("Too few valid two-class cluster bootstrap samples.")
    return {
        "method": "percentile_cluster_bootstrap",
        "confidence_level": 0.95,
        "cluster_count": int(len(unique_clusters)),
        "requested_iterations": int(iterations),
        "valid_iterations": int(valid_iterations),
        "intervals": {
            name: {
                "lower": float(np.quantile(values, 0.025)),
                "upper": float(np.quantile(values, 0.975)),
            }
            for name, values in samples.items()
        },
    }


def evaluate_groups(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
    column: str,
) -> list[dict[str, object]]:
    working = frame.copy()
    working["probability"] = probabilities
    rows: list[dict[str, object]] = []
    for value, group in working.groupby(column, dropna=False, sort=True):
        truth = group["target"].to_numpy(dtype="int32")
        metrics = add_confidence_intervals(
            training.threshold_metrics(
                truth,
                group["probability"].to_numpy(dtype="float32"),
                threshold,
            )
        )
        rows.append(
            {
                "group": "unknown" if pd.isna(value) else str(value),
                "positive_support": int(truth.sum()),
                "negative_support": int(len(truth) - truth.sum()),
                "both_classes_present": bool(np.unique(truth).size == 2),
                **metrics,
            }
        )
    return rows


def build_subgroup_report(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, object]:
    working = frame.copy()
    working["sex_group"] = working["sex"].map(training.normalize_sex)
    working["site_group"] = working["anatom_site_general"].map(training.normalize_site)
    age = pd.to_numeric(working["age_approx"], errors="coerce")
    working["age_group"] = pd.cut(
        age,
        bins=[-np.inf, 29, 49, 69, np.inf],
        labels=["0-29", "30-49", "50-69", "70+"],
    ).astype("object")
    working.loc[age.isna(), "age_group"] = "unknown"
    return {
        "evaluation_unit": "image",
        "warning": (
            "These retrospective metadata slices are diagnostic only. They do not "
            "replace a prospectively powered fairness study, and skin-tone/device "
            "labels are unavailable."
        ),
        "source": evaluate_groups(working, probabilities, threshold, "source"),
        "sex": evaluate_groups(working, probabilities, threshold, "sex_group"),
        "site": evaluate_groups(working, probabilities, threshold, "site_group"),
        "age": evaluate_groups(working, probabilities, threshold, "age_group"),
        "missingness": {
            "age": int(age.isna().sum()),
            "sex": int(working["sex_group"].eq("unknown").sum()),
            "site": int(working["site_group"].eq("unknown").sum()),
            "skin_tone": int(len(working)),
            "capture_device": int(len(working)),
        },
    }


def save_final_metrics(
    run_dir: Path,
    config: dict[str, object],
    val_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    val_direct: np.ndarray,
    val_diagnosis: np.ndarray,
    test_direct: np.ndarray,
    test_diagnosis: np.ndarray,
) -> dict[str, object]:
    binary_target = str(config["binary_target"])
    transforms = int(config["finalization_tta_transforms"])
    minimum_recall = float(config["minimum_operating_recall"])

    val_lesions, val_lesion_diagnosis = training.aggregate_predictions_by_lesion(
        val_frame, val_direct, val_diagnosis
    )
    test_lesions, test_lesion_diagnosis = training.aggregate_predictions_by_lesion(
        test_frame, test_direct, test_diagnosis
    )
    if val_lesion_diagnosis is None or test_lesion_diagnosis is None:
        raise RuntimeError("Diagnosis probabilities were not aggregated.")

    val_truth = val_lesions["target"].to_numpy(dtype="int32")
    test_truth = test_lesions["target"].to_numpy(dtype="int32")
    val_direct_lesion = val_lesions["probability"].to_numpy(dtype="float32")
    test_direct_lesion = test_lesions["probability"].to_numpy(dtype="float32")
    val_diagnosis_target = training.diagnosis_target_probabilities(
        val_lesion_diagnosis, binary_target
    )
    test_diagnosis_target = training.diagnosis_target_probabilities(
        test_lesion_diagnosis, binary_target
    )
    selected_blend, val_raw, blend_rows = training.select_probability_blend(
        val_truth, val_direct_lesion, val_diagnosis_target
    )
    direct_weight = float(selected_blend["direct_weight"])
    test_raw = (
        direct_weight * test_direct_lesion
        + (1.0 - direct_weight) * test_diagnosis_target
    ).astype("float32")

    val_image_diagnosis_target = training.diagnosis_target_probabilities(
        val_diagnosis, binary_target
    )
    test_image_diagnosis_target = training.diagnosis_target_probabilities(
        test_diagnosis, binary_target
    )
    val_image_raw = (
        direct_weight * val_direct
        + (1.0 - direct_weight) * val_image_diagnosis_target
    ).astype("float32")
    test_image_raw = (
        direct_weight * test_direct
        + (1.0 - direct_weight) * test_image_diagnosis_target
    ).astype("float32")

    pd.DataFrame(blend_rows).to_csv(
        run_dir / "probability_blend_sweep.csv", index=False
    )
    training.save_json(
        run_dir / "probability_blend.json",
        {
            "selection_unit": "lesion",
            "selection_split": "validation",
            "direct_weight": direct_weight,
            "diagnosis_target_weight": 1.0 - direct_weight,
            "selected_validation_metrics": selected_blend,
            "direct_validation_metrics": training.probability_metrics(
                val_truth, val_direct_lesion
            ),
            "diagnosis_target_validation_metrics": training.probability_metrics(
                val_truth, val_diagnosis_target
            ),
        },
    )

    temperature, calibrated_validation_loss = training.fit_temperature(
        val_truth, val_raw
    )
    val_probability = training.apply_temperature(val_raw, temperature)
    test_probability = training.apply_temperature(test_raw, temperature)
    val_image_probability = training.apply_temperature(val_image_raw, temperature)
    test_image_probability = training.apply_temperature(test_image_raw, temperature)
    balanced_threshold, operating_threshold, threshold_rows = training.select_thresholds(
        val_truth, val_probability, minimum_recall
    )
    balanced_threshold = add_confidence_intervals(balanced_threshold)
    operating_threshold = add_confidence_intervals(operating_threshold)
    pd.DataFrame(threshold_rows).drop(columns=["confusion_matrix"]).to_csv(
        run_dir / "threshold_sweep.csv", index=False
    )
    training.save_json(
        run_dir / "calibration.json",
        {
            "method": "temperature_scaling",
            "evaluation_unit": "lesion",
            "tta_transforms": transforms,
            "tta_scheme": config.get("tta_scheme", "legacy_v1"),
            "temperature": temperature,
            "validation_log_loss": calibrated_validation_loss,
            "raw_validation_metrics": training.probability_metrics(val_truth, val_raw),
            "calibrated_validation_metrics": training.probability_metrics(
                val_truth, val_probability
            ),
        },
    )
    training.save_json(
        run_dir / "selected_threshold.json",
        {
            "balanced_accuracy_threshold": balanced_threshold["threshold"],
            "balanced_accuracy_validation_metrics": balanced_threshold,
            "operating_threshold": operating_threshold["threshold"],
            "minimum_operating_recall": minimum_recall,
            "operating_validation_metrics": operating_threshold,
        },
    )

    operating_value = float(operating_threshold["threshold"])
    selected_predictions = (test_probability >= operating_value).astype("int32")
    lesion_operating = add_confidence_intervals(
        training.threshold_metrics(test_truth, test_probability, operating_value)
    )
    lesion_groups = (
        test_frame[["lesion_id", "split_group_id"]]
        .drop_duplicates()
        .set_index("lesion_id")["split_group_id"]
    )
    if lesion_groups.index.duplicated().any():
        raise ValueError("A lesion maps to more than one split group in the test split.")
    test_cluster_ids = test_lesions["lesion_id"].map(lesion_groups)
    if test_cluster_ids.isna().any():
        raise ValueError("Test lesion predictions are missing split-group IDs.")
    cluster_intervals = cluster_bootstrap_confidence_intervals(
        test_truth,
        test_probability,
        operating_value,
        test_cluster_ids.astype(str).to_numpy(),
        seed=int(config.get("seed", 42)),
    )
    risk_results = {
        "evaluation_unit": "lesion",
        "tta_transforms": transforms,
        "tta_scheme": config.get("tta_scheme", "legacy_v1"),
        "direct_weight": direct_weight,
        "component_probability_metrics": {
            "direct_head": training.probability_metrics(
                test_truth, test_direct_lesion
            ),
            "diagnosis_target": training.probability_metrics(
                test_truth, test_diagnosis_target
            ),
        },
        "raw_probability_metrics": training.probability_metrics(test_truth, test_raw),
        "calibrated_probability_metrics": training.probability_metrics(
            test_truth, test_probability
        ),
        "threshold_metrics": {
            "default_0_5": training.threshold_metrics(
                test_truth, test_probability, 0.5
            ),
            "balanced_accuracy": training.threshold_metrics(
                test_truth,
                test_probability,
                float(balanced_threshold["threshold"]),
            ),
            "operating": lesion_operating,
        },
        "baselines": {
            "all_negative": training.threshold_metrics(
                test_truth, np.zeros(len(test_truth), dtype="float32"), 0.5
            ),
            "all_positive": training.threshold_metrics(
                test_truth, np.ones(len(test_truth), dtype="float32"), 0.5
            ),
        },
        "cluster_bootstrap_confidence_intervals_95": cluster_intervals,
        "classification_report": classification_report(
            test_truth,
            selected_predictions,
            labels=[0, 1],
            target_names=training.binary_class_names(binary_target),
            output_dict=True,
            zero_division=0,
        ),
    }
    diagnosis_results = training.diagnosis_metrics(
        test_lesions["diagnosis_index"].to_numpy(dtype="int32"),
        test_lesion_diagnosis,
    )

    image_truth = test_frame["target"].to_numpy(dtype="int32")
    image_predictions = (test_image_probability >= operating_value).astype("int32")
    image_operating = add_confidence_intervals(
        training.threshold_metrics(
            image_truth, test_image_probability, operating_value
        )
    )
    image_risk_results = {
        "evaluation_unit": "image",
        "tta_transforms": transforms,
        "tta_scheme": config.get("tta_scheme", "legacy_v1"),
        "direct_weight": direct_weight,
        "component_probability_metrics": {
            "direct_head": training.probability_metrics(image_truth, test_direct),
            "diagnosis_target": training.probability_metrics(
                image_truth, test_image_diagnosis_target
            ),
        },
        "raw_probability_metrics": training.probability_metrics(
            image_truth, test_image_raw
        ),
        "calibrated_probability_metrics": training.probability_metrics(
            image_truth, test_image_probability
        ),
        "threshold_metrics": {
            "default_0_5": training.threshold_metrics(
                image_truth, test_image_probability, 0.5
            ),
            "balanced_accuracy": training.threshold_metrics(
                image_truth,
                test_image_probability,
                float(balanced_threshold["threshold"]),
            ),
            "operating": image_operating,
        },
        "baselines": {
            "all_negative": training.threshold_metrics(
                image_truth, np.zeros(len(image_truth), dtype="float32"), 0.5
            ),
            "all_positive": training.threshold_metrics(
                image_truth, np.ones(len(image_truth), dtype="float32"), 0.5
            ),
        },
        "classification_report": classification_report(
            image_truth,
            image_predictions,
            labels=[0, 1],
            target_names=training.binary_class_names(binary_target),
            output_dict=True,
            zero_division=0,
        ),
    }
    image_diagnosis_results = training.diagnosis_metrics(
        test_frame["diagnosis_index"].to_numpy(dtype="int32"), test_diagnosis
    )
    training.save_json(run_dir / "test_risk_metrics.json", risk_results)
    training.save_json(run_dir / "test_diagnosis_metrics.json", diagnosis_results)
    training.save_json(run_dir / "test_image_risk_metrics.json", image_risk_results)
    training.save_json(
        run_dir / "test_image_diagnosis_metrics.json", image_diagnosis_results
    )

    validation_predictions = val_frame.drop(
        columns=["runtime_image_path"], errors="ignore"
    ).copy()
    validation_predictions["probability_direct_raw"] = val_direct
    validation_predictions["probability_diagnosis_target_raw"] = (
        val_image_diagnosis_target
    )
    validation_predictions[f"probability_{binary_target}_raw"] = val_image_raw
    validation_predictions[f"probability_{binary_target}"] = val_image_probability
    validation_predictions[f"{binary_target}_prediction"] = (
        val_image_probability >= operating_value
    ).astype("int32")
    validation_predictions["diagnosis_prediction"] = [
        training.DIAGNOSES[index] for index in val_diagnosis.argmax(axis=1)
    ]
    validation_predictions["diagnosis_confidence"] = val_diagnosis.max(axis=1)
    validation_predictions.to_csv(run_dir / "val_predictions.csv", index=False)

    image_predictions_frame = test_frame.drop(
        columns=["runtime_image_path"], errors="ignore"
    ).copy()
    image_predictions_frame["probability_direct_raw"] = test_direct
    image_predictions_frame["probability_diagnosis_target_raw"] = (
        test_image_diagnosis_target
    )
    image_predictions_frame[f"probability_{binary_target}_raw"] = test_image_raw
    image_predictions_frame[f"probability_{binary_target}"] = test_image_probability
    image_predictions_frame[f"{binary_target}_prediction"] = image_predictions
    image_predictions_frame["diagnosis_prediction"] = [
        training.DIAGNOSES[index] for index in test_diagnosis.argmax(axis=1)
    ]
    image_predictions_frame["diagnosis_confidence"] = test_diagnosis.max(axis=1)
    image_predictions_frame.to_csv(run_dir / "test_predictions.csv", index=False)

    for lesions, direct_values, diagnosis_values, raw_values, probability_values, name in (
        (
            val_lesions,
            val_direct_lesion,
            val_diagnosis_target,
            val_raw,
            val_probability,
            "val",
        ),
        (
            test_lesions,
            test_direct_lesion,
            test_diagnosis_target,
            test_raw,
            test_probability,
            "test",
        ),
    ):
        diagnosis_matrix = (
            val_lesion_diagnosis if name == "val" else test_lesion_diagnosis
        )
        lesions["probability_direct_raw"] = direct_values
        lesions["probability_diagnosis_target_raw"] = diagnosis_values
        lesions["probability_raw"] = raw_values
        lesions["probability"] = probability_values
        lesions["prediction"] = (probability_values >= operating_value).astype("int32")
        lesions["diagnosis_prediction"] = [
            training.DIAGNOSES[index] for index in diagnosis_matrix.argmax(axis=1)
        ]
        lesions["diagnosis_confidence"] = diagnosis_matrix.max(axis=1)
        lesions.to_csv(run_dir / f"{name}_lesion_predictions.csv", index=False)

    per_source = {
        str(source): add_confidence_intervals(
            training.threshold_metrics(
                group["target"].to_numpy(dtype="int32"),
                group["probability"].to_numpy(dtype="float32"),
                operating_value,
            )
        )
        for source, group in test_lesions.groupby("source")
    }
    diagnosis_error_summary = []
    for diagnosis_name, group in test_lesions.groupby("diagnosis"):
        diagnosis_error_summary.append(
            {
                "diagnosis": str(diagnosis_name),
                "lesion_support": int(len(group)),
                "image_support": int(group["image_count"].sum()),
                "correct_binary": int((group["target"] == group["prediction"]).sum()),
                "binary_error": int((group["target"] != group["prediction"]).sum()),
                "mean_probability": float(group["probability"].mean()),
            }
        )
    training.save_json(run_dir / "test_risk_metrics_by_source.json", per_source)
    training.save_json(
        run_dir / "test_errors_by_diagnosis.json", diagnosis_error_summary
    )
    training.save_json(
        run_dir / "test_subgroup_metrics.json",
        build_subgroup_report(test_frame, test_image_probability, operating_value),
    )

    training.plot_confusion(
        risk_results["threshold_metrics"]["operating"]["confusion_matrix"],
        training.binary_class_names(binary_target),
        run_dir / "risk_confusion_matrix.png",
        f"Risk Confusion Matrix @ {operating_value:.3f}",
    )
    training.plot_confusion(
        diagnosis_results["confusion_matrix"],
        list(training.DIAGNOSES),
        run_dir / "diagnosis_confusion_matrix.png",
        "Diagnosis Confusion Matrix",
    )
    return {
        "selected_threshold": operating_value,
        "lesion_test_operating_metrics": lesion_operating,
        "image_test_operating_metrics": image_operating,
        "lesion_test_probability_metrics": risk_results[
            "calibrated_probability_metrics"
        ],
    }


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.data_workers < 1 or args.prefetch_batches < 0:
        raise ValueError("Batch size/workers must be positive and prefetch cannot be negative.")
    run_dir = args.run_dir.resolve()
    required = (
        "run_config.json",
        "metadata_schema.json",
        "train_split.csv",
        "val_split.csv",
        "test_split.csv",
    )
    missing = [name for name in required if not (run_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Run is missing required artifacts: {missing}")

    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    schema = json.loads((run_dir / "metadata_schema.json").read_text(encoding="utf-8"))
    binary_target = str(config.get("binary_target", "high_risk"))
    config["binary_target"] = binary_target
    transforms = (
        args.tta_transforms
        if args.tta_transforms is not None
        else int(config.get("tta_transforms", 1))
    )
    config["finalization_tta_transforms"] = transforms
    training.configure_runtime(str(config.get("mixed_precision", "auto")))
    training.set_random_seeds(int(config.get("seed", 42)))

    status_path = run_dir / "finalization_status.json"
    status: dict[str, object] = {
        "status": "running",
        "started_at": utc_now(),
        "updated_at": utc_now(),
        "batch_size": args.batch_size,
        "tta_transforms": transforms,
        "completed_stages": [],
    }
    save_json_atomic(status_path, status)

    train_frame = load_run_split(run_dir, "train", binary_target)
    val_frame = load_run_split(run_dir, "val", binary_target)
    test_frame = load_run_split(run_dir, "test", binary_target)
    training.assert_split_integrity(
        training.SplitFrames(train=train_frame, val=val_frame, test=test_frame)
    )
    del train_frame
    gc.collect()
    status["completed_stages"] = ["split_integrity"]
    status["updated_at"] = utc_now()
    save_json_atomic(status_path, status)

    encoder = training.MetadataEncoder(
        age_median=float(schema["age_median"]),
        feature_names=tuple(schema["feature_names"]),
    )
    image_size = int(config["image_size"])
    seed = int(config.get("seed", 42))
    cache_dir = run_dir / "finalization_cache"
    candidates = [run_dir / name for name in CANDIDATE_NAMES if (run_dir / name).exists()]
    if not candidates:
        raise FileNotFoundError("No best head or fine-tune checkpoint is available.")

    metrics_by_candidate: dict[str, dict[str, float]] = {}
    for path in candidates:
        direct, diagnosis = predict_checkpoint(
            path,
            val_frame,
            encoder,
            image_size,
            args.batch_size,
            args.data_workers,
            args.prefetch_batches,
            seed,
            1,
            cache_dir / f"candidate_{path.stem}_validation_tta1.npz",
            args.force_predictions,
            tta_scheme=config.get("tta_scheme", "legacy_v1"),
        )
        metrics_by_candidate[path.name] = candidate_metrics(
            val_frame, direct, diagnosis, binary_target
        )
        del direct, diagnosis
        release_runtime_memory()

    selected_path = max(
        candidates,
        key=lambda path: (
            metrics_by_candidate[path.name].get("lesion_blended_pr_auc", -1.0),
            metrics_by_candidate[path.name].get("lesion_blended_roc_auc", -1.0),
        ),
    )
    best_model_path = run_dir / "best_model.keras"
    if selected_path.resolve() != best_model_path.resolve():
        shutil.copy2(selected_path, best_model_path)
    training.save_json(
        run_dir / "model_selection.json",
        {
            "selected_candidate": selected_path.name,
            "selection_unit": "lesion",
            "selection_split": "validation",
            "selection_primary_metric": "blended_pr_auc",
            "validation_metrics": metrics_by_candidate,
            "selected_model_sha256": sha256_file(best_model_path),
        },
    )
    status["completed_stages"] = ["split_integrity", "model_selection"]
    status["selected_candidate"] = selected_path.name
    status["updated_at"] = utc_now()
    save_json_atomic(status_path, status)

    history_frames = []
    for name, stage in (("training_head.csv", "head"), ("training_finetune.csv", "finetune")):
        path = run_dir / name
        if path.exists():
            history_frames.append(pd.read_csv(path).assign(stage=stage))
    if history_frames:
        pd.concat(history_frames, ignore_index=True).to_csv(
            run_dir / "training_history.csv", index=False
        )
        training.plot_history(history_frames, run_dir / "training_curves.png")

    selected_hash = sha256_file(best_model_path)[:16]
    val_direct, val_diagnosis = predict_checkpoint(
        best_model_path,
        val_frame,
        encoder,
        image_size,
        args.batch_size,
        args.data_workers,
        args.prefetch_batches,
        seed,
        transforms,
        cache_dir / f"selected_{selected_hash}_validation_tta{transforms}.npz",
        args.force_predictions,
        tta_scheme=config.get("tta_scheme", "legacy_v1"),
    )
    status["completed_stages"] = [
        "split_integrity",
        "model_selection",
        "validation_inference",
    ]
    status["updated_at"] = utc_now()
    save_json_atomic(status_path, status)

    test_direct, test_diagnosis = predict_checkpoint(
        best_model_path,
        test_frame,
        encoder,
        image_size,
        args.batch_size,
        args.data_workers,
        args.prefetch_batches,
        seed,
        transforms,
        cache_dir / f"selected_{selected_hash}_test_tta{transforms}.npz",
        args.force_predictions,
        tta_scheme=config.get("tta_scheme", "legacy_v1"),
    )
    status["completed_stages"] = [
        "split_integrity",
        "model_selection",
        "validation_inference",
        "test_inference",
    ]
    status["updated_at"] = utc_now()
    save_json_atomic(status_path, status)

    summary = save_final_metrics(
        run_dir,
        config,
        val_frame,
        test_frame,
        val_direct,
        val_diagnosis,
        test_direct,
        test_diagnosis,
    )
    integrity_files = (
        "run_config.json",
        "metadata_schema.json",
        "train_split.csv",
        "val_split.csv",
        "test_split.csv",
        "best_model.keras",
        "model_selection.json",
        "calibration.json",
        "selected_threshold.json",
        "test_risk_metrics.json",
    )
    status.update(
        {
            "status": "complete",
            "completed_at": utc_now(),
            "updated_at": utc_now(),
            "completed_stages": [
                "split_integrity",
                "model_selection",
                "validation_inference",
                "test_inference",
                "metrics_and_calibration",
            ],
            "summary": summary,
            "artifact_sha256": {
                name: sha256_file(run_dir / name) for name in integrity_files
            },
        }
    )
    save_json_atomic(status_path, status)
    from compare_model_runs import compare_runs, write_report

    baseline_dir = args.baseline_run_dir
    if baseline_dir is None and config.get("warm_start_model"):
        baseline_dir = Path(config["warm_start_model"]).parent
    try:
        if baseline_dir is None or baseline_dir.resolve() == run_dir:
            raise ValueError("A separate finalized baseline run is required for promotion.")
        promotion = compare_runs(
            baseline_dir, run_dir, float(config.get("minimum_operating_recall", 0.95))
        )
    except (OSError, ValueError, KeyError) as error:
        promotion = {
            "candidate_passes": False, "decision": "comparison_unavailable",
            "reason": str(error), "selection_split": "validation",
        }
    write_report(run_dir / "promotion_decision.json", promotion)
    print("Research model promotion:", promotion["decision"])
    print("Selected within-run model:", selected_path.name)
    print(json.dumps(summary, indent=2))
    print("Finalized artifacts:", run_dir)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        try:
            failed_args = parse_args()
            failed_run_dir = failed_args.run_dir.resolve()
            failed_status_path = failed_run_dir / "finalization_status.json"
            failed_status = (
                json.loads(failed_status_path.read_text(encoding="utf-8"))
                if failed_status_path.exists()
                else {}
            )
            failed_status.update(
                {
                    "status": "failed",
                    "updated_at": utc_now(),
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                }
            )
            save_json_atomic(failed_status_path, failed_status)
        except Exception:
            pass
        raise
