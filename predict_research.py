"""Local-only inference for the retained melanoma research checkpoint."""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import numpy as np
import pandas as pd
import tensorflow as tf

import train_multitask as training

ROOT = Path(__file__).resolve().parent


def verify_checkpoint(path, manifest):
    with Path(path).open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != manifest["checkpoint_sha256"]:
        raise ValueError("Checkpoint SHA-256 does not match the retained research release.")
    return digest


def aggregate_scores(direct, diagnoses, config):
    direct, diagnoses = np.asarray(direct), np.asarray(diagnoses)
    if (direct.ndim != 1 or len(direct) == 0 or diagnoses.shape != (len(direct), 8)
            or not np.isfinite(direct).all() or not np.isfinite(diagnoses).all()
            or ((direct < 0) | (direct > 1)).any() or ((diagnoses < 0) | (diagnoses > 1)).any()):
        raise ValueError("Invalid model scores.")
    clipped = np.clip(direct.astype("float64"), 1e-6, 1 - 1e-6)
    lesion_direct = np.asarray([1 / (1 + np.exp(-np.log(clipped / (1 - clipped)).mean()))], dtype="float32")
    lesion_diagnoses = diagnoses.astype("float32").mean(axis=0)
    raw = config["direct_weight"] * lesion_direct + (1 - config["direct_weight"]) * lesion_diagnoses[training.DIAGNOSIS_TO_INDEX["MEL"]]
    calibrated = training.apply_temperature(raw, config["temperature"])
    return float(calibrated[0])


def predict(paths, model_path, manifest_path=ROOT / "models/model_manifest.json"):
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8-sig"))
    digest = verify_checkpoint(model_path, manifest)
    config = manifest["inference"]
    if not paths or any(Path(path).suffix.lower() not in {".jpg", ".jpeg"} for path in paths):
        raise ValueError("Provide JPEG dermoscopy images of the same lesion.")
    encoder = training.MetadataEncoder(age_median=config["metadata_age_median"],
        feature_names=tuple(config["metadata_feature_names"]))
    unknown = pd.DataFrame({"age_approx": [np.nan] * len(paths), "sex": "unknown", "anatom_site_general": "unknown"})
    images = [training.resize_and_normalize_image(tf.io.decode_jpeg(tf.io.read_file(str(path)), channels=3),
        config["image_size"]) for path in paths]
    inputs = {"image": tf.stack(images), "metadata": encoder.transform(unknown)}
    # Keras ignores these placeholder targets during prediction; no labels are inferred.
    dataset = tf.data.Dataset.from_tensor_slices((inputs, tf.zeros(len(paths)))).batch(1)
    options = tf.data.Options()
    options.threading.private_threadpool_size = 1
    options.threading.max_intra_op_parallelism = 1
    dataset = dataset.with_options(options)
    model = tf.keras.models.load_model(model_path, compile=False, safe_mode=True)
    direct, diagnoses = training.predict_with_tta(model, dataset, config["tta_transforms"], config["tta_scheme"])
    score = aggregate_scores(direct, diagnoses, config)
    return {"model_id": manifest["model_id"], "checkpoint_sha256": digest,
        "image_count_same_lesion": len(paths), "research_score": score,
        "research_threshold": config["threshold"], "melanoma_research_flag": score >= config["threshold"],
        "clinical_use_validated": False,
        "warning": "Research output, not a diagnosis or an individual cancer-risk estimate. Non-melanoma includes other cancers."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", nargs="+", type=Path, required=True, help="JPEG images of ONE lesion.")
    parser.add_argument("--model", type=Path, default=ROOT / "models/melanoma-effnetv2b1-v0.1.0-research.keras")
    parser.add_argument("--acknowledge-research-use", action="store_true")
    args = parser.parse_args()
    if not args.acknowledge_research_use:
        parser.error("This is not a diagnostic tool. Explicitly pass --acknowledge-research-use for research inference.")
    print(json.dumps(predict(args.images, args.model), indent=2))
