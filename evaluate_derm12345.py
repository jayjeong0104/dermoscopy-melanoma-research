from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")

import numpy as np
import pandas as pd
import tensorflow as tf

import train_multitask as training
from external_evaluation_contract import recorded_exposure_manifests, verify_inference_binding


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = (
    Path(r"D:\SkinCancerData\DERM12345\test")
    if os.name == "nt"
    else Path("/mnt/d/SkinCancerData/DERM12345/test")
)
IMAGE_SUFFIXES = {".jpg", ".jpeg"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved model on the locked DERM12345 patient-level test split."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument(
        "--metadata",
        type=Path,
        default=PROJECT_ROOT / "data" / "derm12345" / "derm12345_metadata_test.tab",
    )
    parser.add_argument(
        "--training-manifest",
        type=Path,
        help="Optional additional overlap manifest; the selected run and all ancestors are always audited.",
    )
    parser.add_argument("--model-name", default="best_model.keras")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--hash-workers", type=int, default=2)
    return parser.parse_args()


def index_images(root: Path) -> dict[str, Path]:
    if not root.exists():
        raise FileNotFoundError(f"DERM12345 test directory not found: {root}")
    image_index: dict[str, Path] = {}
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        key = path.stem.upper()
        if key in image_index:
            raise ValueError(f"Duplicate DERM12345 image ID: {key}")
        image_index[key] = path.resolve()
    return image_index


def configure_target(frame: pd.DataFrame, binary_target: str) -> pd.DataFrame:
    configured = frame.copy()
    malignancy = configured["malignancy"].astype(str).str.lower()
    main_class = configured["main_class_1"].astype(str).str.lower()
    if binary_target == "melanoma":
        configured["target"] = main_class.eq("melanoma").astype("int32")
        configured["target_name"] = np.where(
            configured["target"] == 1, "melanoma", "not_melanoma"
        )
    elif binary_target == "malignant":
        configured["target"] = malignancy.eq("malignant").astype("int32")
        configured["target_name"] = np.where(
            configured["target"] == 1, "malignant", "benign_or_indeterminate"
        )
    elif binary_target == "high_risk":
        configured["target"] = malignancy.isin({"malignant", "indeterminate"}).astype(
            "int32"
        )
        configured["target_name"] = np.where(
            configured["target"] == 1, "high_risk", "benign"
        )
    else:
        raise ValueError(f"Unsupported binary target: {binary_target}")
    return configured


def load_metadata(path: Path, dataset_root: Path, binary_target: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"DERM12345 metadata not found: {path}")
    frame = pd.read_csv(path, sep="\t")
    required = {
        "image_id",
        "patient_id",
        "split",
        "malignancy",
        "main_class_1",
        "label",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"DERM12345 metadata is missing columns: {sorted(missing)}")
    if not frame["split"].astype(str).str.lower().eq("test").all():
        raise ValueError("External evaluation metadata contains non-test rows.")
    if frame["image_id"].duplicated().any():
        raise ValueError("DERM12345 metadata contains duplicate image IDs.")

    image_index = index_images(dataset_root)
    frame["image_path"] = frame["image_id"].astype(str).str.upper().map(image_index)
    missing_images = frame.loc[frame["image_path"].isna(), "image_id"].head(10).tolist()
    if missing_images:
        raise FileNotFoundError(
            f"DERM12345 test images are incomplete; missing examples: {missing_images}"
        )
    archive_labels = frame["image_path"].map(lambda value: Path(value).parent.name.lower())
    label_mismatch = ~archive_labels.eq(frame["label"].astype(str).str.lower())
    if label_mismatch.any():
        examples = frame.loc[label_mismatch, ["image_id", "label", "image_path"]].head(5)
        raise ValueError(f"Archive folder labels disagree with metadata:\n{examples}")
    return configure_target(frame, binary_target)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_exact_overlap(
    frame: pd.DataFrame, training_manifest: Path | list[Path], workers: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if workers < 1:
        raise ValueError("--hash-workers must be at least 1.")
    manifests = [training_manifest] if isinstance(training_manifest, (str, Path)) else list(training_manifest)
    if not manifests:
        raise ValueError("At least one recorded exposure manifest is required.")
    training_frames = []
    for manifest in manifests:
        if not Path(manifest).is_file():
            raise FileNotFoundError(f"Training manifest not found: {manifest}")
        columns = pd.read_csv(manifest, nrows=0).columns
        use_columns = ["sha256"] + [name for name in ("patient_id", "source") if name in columns]
        training_frames.append(pd.read_csv(manifest, usecols=use_columns, dtype="string", low_memory=False))
    training_frame = pd.concat(training_frames, ignore_index=True)
    if training_frame["sha256"].isna().any():
        raise ValueError("Training manifest contains missing SHA-256 hashes.")
    training_hashes = set(training_frame["sha256"])
    if "patient_id" in training_frame:
        # Unlinked source-local patient IDs must not collide merely by spelling.
        patient_rows = training_frame
        if "source" in patient_rows:
            patient_rows = patient_rows.loc[patient_rows.source.isna() | patient_rows.source.eq("DERM12345")]
        training_patients = set(patient_rows["patient_id"].dropna().astype(str).str.split("::").str[-1])
        patient_overlap = training_patients & set(frame["patient_id"].astype(str))
        if patient_overlap:
            raise RuntimeError(
                f"Locked external test shares {len(patient_overlap)} patients with training."
            )
    with ThreadPoolExecutor(max_workers=workers) as executor:
        external_hashes = list(executor.map(sha256_file, frame["image_path"]))
    overlap = training_hashes & set(external_hashes)
    if overlap:
        raise RuntimeError(
            f"Locked external test contains {len(overlap)} exact training-image hashes."
        )
    audited = frame.copy()
    audited["sha256"] = external_hashes
    duplicate_rows = audited[audited["sha256"].duplicated(keep=False)].copy()
    for digest, group in duplicate_rows.groupby("sha256", sort=False):
        consistency_columns = ("patient_id", "target", "main_class_1", "malignancy", "label")
        conflicts = [
            column
            for column in consistency_columns
            if group[column].astype(str).nunique(dropna=False) != 1
        ]
        if conflicts:
            raise RuntimeError(
                "Locked external exact duplicate has conflicting identity/labels "
                f"for SHA-256 {digest}: {conflicts}"
            )
    if not duplicate_rows.empty:
        duplicate_rows["duplicate_group_size"] = duplicate_rows.groupby("sha256")[
            "sha256"
        ].transform("size")
        duplicate_rows["kept_for_evaluation"] = ~duplicate_rows["sha256"].duplicated(
            keep="first"
        )
    deduplicated = audited.drop_duplicates(subset=["sha256"], keep="first").copy()
    return deduplicated, duplicate_rows


def build_dataset(
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
        image = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
        image = training.resize_and_normalize_image(image, image_size)
        image.set_shape([image_size, image_size, 3])
        metadata_vector.set_shape([len(encoder.feature_names)])
        labels = {
            "high_risk": tf.reshape(tf.cast(target, tf.float32), [1]),
            "diagnosis": tf.zeros([len(training.DIAGNOSES)], dtype=tf.float32),
        }
        return {"image": image, "metadata": metadata_vector}, labels

    options = tf.data.Options()
    options.autotune.enabled = False
    options.deterministic = True
    options.threading.private_threadpool_size = 1
    options.threading.max_intra_op_parallelism = 1
    return dataset.with_options(options).map(decode, num_parallel_calls=1).batch(
        batch_size
    ).prefetch(1)


def metric_bundle(
    truth: np.ndarray,
    raw_probabilities: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, object]:
    operating = training.threshold_metrics(truth, probabilities, threshold)
    operating["confidence_intervals_95"] = training.operating_confidence_intervals(operating)
    all_negative = training.threshold_metrics(
        truth, np.zeros(len(truth), dtype="float32"), 0.5
    )
    return {
        "sample_count": int(len(truth)),
        "positive_count": int((truth == 1).sum()),
        "negative_count": int((truth == 0).sum()),
        "raw_probability_metrics": training.probability_metrics(truth, raw_probabilities),
        "calibrated_probability_metrics": training.probability_metrics(truth, probabilities),
        "default_threshold_metrics": training.threshold_metrics(truth, probabilities, 0.5),
        "operating_threshold_metrics": operating,
        "all_negative_baseline": all_negative,
    }


def acceptance_gate(metrics: dict[str, object]) -> dict[str, object]:
    operating = metrics["operating_threshold_metrics"]
    criteria = {
        "accuracy_at_least_0_95": float(operating["accuracy"]) >= 0.95,
        "balanced_accuracy_at_least_0_95": float(operating["balanced_accuracy"]) >= 0.95,
        "melanoma_recall_at_least_0_95": float(operating["recall"]) >= 0.95,
        "specificity_at_least_0_95": float(operating["specificity"]) >= 0.95,
    }
    return {"passed": all(criteria.values()), "criteria": criteria}


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    model_path = run_dir / args.model_name
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")

    finalization = json.loads(
        (run_dir / "finalization_status.json").read_text(encoding="utf-8")
    )
    if finalization.get("status") != "complete":
        raise RuntimeError("Refusing external evaluation of a run that is not finalized.")
    binding = verify_inference_binding(run_dir, model_path, finalization)
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
    binary_target = str(config.get("binary_target", "high_risk"))
    tta_transforms = int(calibration.get("tta_transforms", config.get("tta_transforms", 1)))
    tta_scheme = calibration.get("tta_scheme", config.get("tta_scheme", "legacy_v1"))
    encoder = training.MetadataEncoder(
        age_median=float(schema["age_median"]),
        feature_names=tuple(schema["feature_names"]),
    )

    frame = load_metadata(args.metadata.resolve(), args.dataset_root.resolve(), binary_target)
    print(f"Auditing {len(frame)} external images for training overlap.", flush=True)
    original_image_count = int(len(frame))
    exposure_manifests, ancestry = recorded_exposure_manifests(model_path, PROJECT_ROOT)
    if args.training_manifest is not None:
        exposure_manifests.append(args.training_manifest.resolve())
    frame, external_duplicates = audit_exact_overlap(
        frame, exposure_manifests, args.hash_workers
    )
    external_duplicates.to_csv(
        run_dir / "derm12345_external_exact_duplicates.csv", index=False
    )
    training.configure_runtime(str(config.get("mixed_precision", "auto")))
    dataset = build_dataset(frame, encoder, int(config["image_size"]), args.batch_size)
    model = tf.keras.models.load_model(model_path, compile=False)
    print(f"Evaluating {len(frame)} unique images with {tta_transforms} TTA views.", flush=True)
    direct_probabilities, diagnosis_probabilities = training.predict_with_tta(
        model, dataset, tta_transforms, tta_scheme
    )
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
    threshold = float(thresholds["operating_threshold"])
    truth = frame["target"].to_numpy(dtype="int32")

    image_metrics = metric_bundle(truth, raw_probabilities, probabilities, threshold)
    patient_predictions = pd.DataFrame(
        {
            "patient_id": frame["patient_id"].astype(str),
            "target": truth,
            "probability_raw": raw_probabilities,
            "probability": probabilities,
        }
    ).groupby("patient_id", sort=False).agg(
        target=("target", "max"),
        probability_raw=("probability_raw", "max"),
        probability=("probability", "max"),
        image_count=("target", "size"),
    ).reset_index()
    patient_metrics = metric_bundle(
        patient_predictions["target"].to_numpy(dtype="int32"),
        patient_predictions["probability_raw"].to_numpy(dtype="float32"),
        patient_predictions["probability"].to_numpy(dtype="float32"),
        threshold,
    )
    results = {
        "dataset": "DERM12345 official patient-level test split",
        "development_status": "external_development_cohort_previously_inspected_not_fresh_confirmation",
        "inference_binding": binding,
        "recorded_exposure_audit": ancestry,
        "additional_overlap_manifests": [str(args.training_manifest.resolve())] if args.training_manifest is not None else [],
        "binary_target": binary_target,
        "model": args.model_name,
        "tta_transforms": tta_transforms,
        "tta_scheme": tta_scheme,
        "direct_weight": direct_weight,
        "operating_threshold_from_internal_validation": threshold,
        "exact_training_hash_overlap": 0,
        "archive_image_count": original_image_count,
        "unique_image_evaluation_count": int(len(frame)),
        "exact_duplicate_rows_in_archive": int(len(external_duplicates)),
        "exact_duplicate_rows_excluded": int(
            len(external_duplicates)
            - external_duplicates.get("kept_for_evaluation", pd.Series(dtype=bool)).sum()
        ),
        "image_level": image_metrics,
        "patient_level_max_probability": patient_metrics,
        "external_95_percent_acceptance_gate": acceptance_gate(image_metrics),
    }
    training.save_json(run_dir / "derm12345_external_metrics.json", results)

    predictions = frame.copy()
    predictions["probability_direct_raw"] = direct_probabilities
    predictions["probability_diagnosis_target_raw"] = diagnosis_target_probabilities
    predictions["probability_raw"] = raw_probabilities
    predictions["probability"] = probabilities
    predictions["prediction"] = (probabilities >= threshold).astype("int32")
    diagnosis_indices = diagnosis_probabilities.argmax(axis=1)
    predictions["diagnosis_prediction"] = [
        training.DIAGNOSES[index] for index in diagnosis_indices
    ]
    predictions["diagnosis_confidence"] = diagnosis_probabilities.max(axis=1)
    predictions.to_csv(run_dir / "derm12345_external_predictions.csv", index=False)
    patient_predictions["prediction"] = (
        patient_predictions["probability"] >= threshold
    ).astype("int32")
    patient_predictions.to_csv(
        run_dir / "derm12345_external_patient_predictions.csv", index=False
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
