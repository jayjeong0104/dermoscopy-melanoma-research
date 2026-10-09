"""Extract clean-reference train/development features without opening test images."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from artifact_io import save_csv
from compare_model_runs import file_hash
from foundation_common import ROOT, aggregate_features, fingerprint, read_chunk, runtime_path, save_chunk, save_json

RUN = ROOT / "artifacts/clean_melanoma_20261005/training/run_20261005_210209"
OUTPUT = ROOT / "artifacts/precision_rebuild_20261006/features_matched_runtime"
DIMENSIONS = 1289  # 1280 pooled features, one direct score, eight diagnosis scores.
CHUNK = 256
AGGREGATION_RULES = {
    "features": "arithmetic_mean",
    "direct_probability": "sigmoid_mean_clipped_logit",
    "direct_probability_clip": 1e-6,
    "diagnosis_probability": "arithmetic_mean",
    "output_dtype": "float32",
}


def now():
    return datetime.now(timezone.utc).isoformat()


def rows_for(run, split):
    if split not in {"train", "val"}:
        raise ValueError("Only training and reused development validation are permitted.")
    frame = pd.read_csv(run / f"{split}_split.csv", low_memory=False)
    required = ["source", "image_id", "lesion_id", "split_group_id", "sha256", "image_path", "target", "diagnosis"]
    if frame[required].isna().any().any() or frame.duplicated(["source", "image_id"]).any():
        raise ValueError("Invalid or duplicate feature identities.")
    if not frame.target.eq(frame.diagnosis.eq("MEL").astype(int)).all():
        raise ValueError("Expected the cleaned melanoma target.")
    return frame.sort_values(["source", "image_id"]).reset_index(drop=True)


def unknown_metadata(schema):
    values = {"age_scaled": float(schema["age_median"]) / 100., "age_missing": 1.,
              "sex_unknown": 1., "site_unknown": 1.}
    return np.asarray([values.get(name, 0.) for name in schema["feature_names"]], dtype="float32")


def chunk_key(plan, split, start, rows):
    return fingerprint({"plan": plan, "split": split, "start": start,
                        "rows": rows[["source", "image_id", "sha256"]].to_dict("records")})


def aggregate_checkpoint_values(frame, values):
    """Pool features and diagnosis probabilities; aggregate direct scores as in training."""
    if values.ndim != 2 or values.shape[1] != DIMENSIONS:
        raise ValueError("Expected pooled features, direct score and eight diagnosis scores.")
    lesions, pooled = aggregate_features(frame, values)
    clipped = np.clip(values[:, 1280].astype("float64"), 1e-6, 1. - 1e-6)
    working = frame[["source", "lesion_id"]].reset_index(drop=True).copy()
    working["logit"] = np.log(clipped / (1. - clipped))
    mean_logits = working.groupby(["source", "lesion_id"], sort=True).logit.mean()
    mean_logits = mean_logits.reindex(pd.MultiIndex.from_frame(lesions[["source", "lesion_id"]])).to_numpy()
    pooled[:, 1280] = (1. / (1. + np.exp(-mean_logits))).astype("float32")
    return lesions, pooled


def load_extracted(output, split):
    if split not in {"train", "val"}:
        raise ValueError("Test feature access is not supported.")
    marker = json.loads((output / f"{split}_complete.json").read_text())
    for name, expected in marker["outputs_sha256"].items():
        if file_hash(output / name) != expected:
            raise ValueError(f"Feature output changed: {name}")
    rows = pd.read_csv(output / f"{split}_lesions.csv", low_memory=False)
    values = read_chunk(output / f"{split}_lesion_features.npz", marker["plan_fingerprint"], len(rows), 1280)
    return rows, values


def verify_run(run):
    """Bind extraction to a completed, clean ImageNet run across OS paths."""
    config = json.loads((run / "run_config.json").read_text())
    if (config["binary_target"] != "melanoma" or config["warm_start_model"] is not None
            or config.get("weights") != "imagenet"):
        raise ValueError("Expected an ImageNet-only clean melanoma run.")
    complete = json.loads((run.parent.parent / "complete.json").read_text())
    recorded = {name.replace("\\", "/"): digest for name, digest in complete["outputs_sha256"].items()}
    if len(recorded) != len(complete["outputs_sha256"]):
        raise ValueError("Ambiguous completion paths.")
    for name in ("best_finetune_model.keras", "train_split.csv", "val_split.csv", "run_config.json", "metadata_schema.json"):
        relative = (run / name).relative_to(run.parent.parent).as_posix()
        if file_hash(run / name) != recorded.get(relative):
            raise ValueError(f"Clean run changed since completion: {name}")
    return config


def extract(output, batch_size=4, run=RUN):
    config = verify_run(run)
    import tensorflow as tf
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    if not tf.config.list_physical_devices("GPU"):
        raise RuntimeError("This bounded extraction requires the available GPU.")
    tf.keras.mixed_precision.set_global_policy(config["mixed_precision_policy"])
    schema = json.loads((run / "metadata_schema.json").read_text())
    model_path = run / "best_finetune_model.keras"
    output.mkdir(parents=True, exist_ok=True)
    plan = {"version": 4, "run": run.name, "model_sha256": file_hash(model_path),
        "splits_sha256": {split: file_hash(run / f"{split}_split.csv") for split in ("train", "val")},
        "metadata_schema_sha256": file_hash(run / "metadata_schema.json"),
        "batch_size": batch_size, "chunk_images": CHUNK,
        "code_sha256": {name: file_hash(ROOT / name) for name in
            ("extract_clean_reference.py", "foundation_common.py", "artifact_io.py", "compare_model_runs.py")},
        "tensorflow": tf.__version__, "keras": tf.keras.__version__, "image_size": config["image_size"],
        "mixed_precision_policy": tf.keras.mixed_precision.global_policy().name,
        "metadata": "unknown", "tta_transforms": 1, "feature_layer": "image_pool",
        "aggregation": AGGREGATION_RULES,
        "image_bytes_sha256_verified": True, "test_or_hiba_inference": False,
        "limitation": "Encoder selected on reused validation; training features are in-sample for encoder."}
    plan_path = output / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Extraction inputs or code changed; use a fresh directory.")
    save_json(plan_path, plan)
    snapshot = output / "code_snapshot"
    snapshot.mkdir(exist_ok=True)
    for name, expected in plan["code_sha256"].items():
        saved = snapshot / name
        if saved.exists() and file_hash(saved) != expected:
            raise ValueError(f"Extraction code snapshot changed: {name}")
        if not saved.exists():
            shutil.copyfile(ROOT / name, saved)
    model = tf.keras.models.load_model(model_path, compile=False)
    extractor = tf.keras.Model(model.inputs, {"features": model.get_layer("image_pool").output,
        "direct": model.get_layer("high_risk").output, "diagnosis": model.get_layer("diagnosis").output})
    metadata = unknown_metadata(schema)
    image_size = int(config["image_size"])

    @tf.function(reduce_retracing=True)
    def predict(inputs):
        outputs = extractor(inputs, training=False)
        return tf.concat([tf.cast(outputs["features"], tf.float32), outputs["direct"], outputs["diagnosis"]], axis=1)

    def dataset(frame):
        def generate():
            for row in frame.itertuples():
                data = runtime_path(row.image_path).read_bytes()
                if hashlib.sha256(data).hexdigest() != row.sha256:
                    raise ValueError(f"Image content changed: {row.source}/{row.image_id}")
                yield data
        ds = tf.data.Dataset.from_generator(generate, output_signature=tf.TensorSpec((), tf.string))
        def decode(data):
            pixels = tf.io.decode_jpeg(data, channels=3)
            pixels = tf.image.resize(pixels, [image_size, image_size], antialias=True)
            return {"image": tf.cast(pixels, tf.float32) / 255., "metadata": tf.constant(metadata)}
        opts = tf.data.Options()
        opts.deterministic = True
        opts.autotune.enabled = False
        opts.threading.private_threadpool_size = 2
        opts.threading.max_intra_op_parallelism = 1
        return ds.map(decode, num_parallel_calls=2).batch(batch_size).prefetch(1).with_options(opts)

    for split in ("val", "train"):
        if (output / f"{split}_complete.json").exists():
            load_extracted(output, split)
            continue
        frame = rows_for(run, split)
        if split == "val":
            sample = next(iter(dataset(frame.iloc[:batch_size])))
            expected = model.predict_on_batch(sample)
            observed = predict(sample).numpy()
            delta = float(max(np.max(np.abs(observed[:, 1280] - np.asarray(expected["high_risk"]).ravel())),
                              np.max(np.abs(observed[:, 1281:] - np.asarray(expected["diagnosis"])))))
            if delta > 1e-6:
                raise ValueError(f"Extractor changed original model predictions: {delta:g}")
            save_json(output / "prediction_equivalence.json", {"rows": len(observed), "maximum_absolute_difference": delta,
                "comparison": "Original checkpoint predict_on_batch versus extracted features+outputs, matched mixed-precision runtime."})
        parts = []
        for start in range(0, len(frame), CHUNK):
            rows = frame.iloc[start:start + CHUNK]
            path = output / f"{split}_{start:06d}.npz"
            key = chunk_key(plan, split, start, rows)
            if path.exists():
                values = read_chunk(path, key, len(rows), DIMENSIONS)
            else:
                values = np.concatenate([predict(inputs).numpy() for inputs in dataset(rows)])
                if values.shape != (len(rows), DIMENSIONS):
                    raise ValueError("Unexpected feature/output shape.")
                if ((values[:, 1280:] < 0) | (values[:, 1280:] > 1)).any():
                    raise ValueError("Invalid model probabilities.")
                save_chunk(path, values, key)
            parts.append(values)
            save_json(output / "status.json", {"status": "running", "split": split,
                "images_done": start + len(rows), "images_total": len(frame), "updated_at": now(), "pid": os.getpid()})
            print(f"{split}: {start + len(rows)}/{len(frame)}", flush=True)
        all_values = np.concatenate(parts)
        lesions, pooled = aggregate_checkpoint_values(frame, all_values)
        lesions["direct_probability"] = pooled[:, 1280]
        for index, name in enumerate(config["diagnoses"]):
            lesions[f"diagnosis_probability_{name}"] = pooled[:, 1281 + index]
        lesions["diagnosis_probability"] = lesions["diagnosis_probability_MEL"]
        save_csv(lesions, output / f"{split}_lesions.csv")
        key = fingerprint(plan)
        save_chunk(output / f"{split}_lesion_features.npz", pooled[:, :1280], key)
        names = [f"{split}_lesions.csv", f"{split}_lesion_features.npz", "plan.json"]
        save_json(output / f"{split}_complete.json", {"plan_fingerprint": key, "completed_at": now(),
            "outputs_sha256": {name: file_hash(output / name) for name in names}})
    save_json(output / "status.json", {"status": "complete", "updated_at": now(), "test_or_hiba_inference": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--run-dir", type=Path, default=RUN)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    import fcntl
    from run_foundation_experiment import assert_idle
    with (ROOT / "artifacts/recovery_pilot.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert_idle()
        extract(args.output, args.batch_size, args.run_dir)


if __name__ == "__main__":
    main()
