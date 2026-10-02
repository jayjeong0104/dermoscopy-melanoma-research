"""TensorFlow linear probes: training-only head selection, then validation-only scoring."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from artifact_io import save_csv, save_model
from compare_model_runs import file_hash, operating_metrics
from extract_foundation_features import load_features
from foundation_common import (ENCODERS, KEYS, LIMITATION, REFERENCE, aggregate_features,
                               inner_partition, save_json)
from operating_policy import crossfit_policy, decision_metrics


def build_head(tf, features, regularization: float, seed: int):
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    mean = features.mean(axis=0, dtype=np.float64).astype("float32")
    variance = np.maximum(features.var(axis=0, dtype=np.float64), 1e-6).astype("float32")
    inputs = tf.keras.Input((features.shape[1],), dtype="float32", name="frozen_image_embedding")
    x = tf.keras.layers.Normalization(mean=mean, variance=variance, name="training_only_normalization")(inputs)
    output = tf.keras.layers.Dense(1, activation="sigmoid", dtype="float32", name="melanoma",
        kernel_regularizer=tf.keras.regularizers.L2(regularization))(x)
    model = tf.keras.Model(inputs, output, name="melanoma_linear_probe")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=.003),
                  loss=tf.keras.losses.BinaryCrossentropy())
    return model


def dataset(tf, x, y, seed: int):
    options = tf.data.Options()
    options.threading.private_threadpool_size = 1
    options.threading.max_intra_op_parallelism = 1
    return (tf.data.Dataset.from_tensor_slices((x, y.astype("float32")[:, None]))
            .shuffle(len(x), seed=seed, reshuffle_each_iteration=True).batch(128)
            .with_options(options).prefetch(1))


def predict(model, x) -> np.ndarray:
    return np.concatenate([model(x[start:start + 512], training=False).numpy().ravel()
                           for start in range(0, len(x), 512)])


def train(experiment: Path, encoder: str, smoke: bool = False) -> None:
    import tensorflow as tf
    from run_foundation_experiment import verify_plan
    plan = verify_plan(experiment)
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.keras.mixed_precision.set_global_policy("float32")
    if smoke:
        rng = np.random.default_rng(7)
        x = rng.normal(size=(128, 12)).astype("float32")
        y = (x[:, 0] > 0).astype(int)
        model = build_head(tf, x[:100], .001, 42)
        model.fit(dataset(tf, x[:100], y[:100], 42), epochs=2, verbose=2)
        path = experiment / "smoke_head.keras"
        save_model(model, path)
        restored = tf.keras.models.load_model(path)
        np.testing.assert_allclose(predict(model, x), predict(restored, x), atol=1e-6)
        assert np.all(np.isfinite(predict(model, x)))
        save_json(experiment / "smoke_head.json", {"synthetic_only": True,
                  "tensorflow": tf.__version__, "save_reload_verified": True})
        return
    destination = experiment / "heads" / encoder
    destination.mkdir(parents=True, exist_ok=True)
    rows, features = load_features(experiment, encoder, "train")
    meta, x = aggregate_features(rows, features)
    del features
    y = meta.target.to_numpy(dtype=int)
    held = inner_partition(meta)
    save_csv(meta[KEYS].assign(inner_holdout=held.astype(int)), destination / "inner_partition.csv")
    print(f"PID {os.getpid()}: {encoder}, {len(x)} training lesions, inner holdout {int(held.sum())}", flush=True)
    choices = []
    for regularization in plan["head"]["l2_grid"]:
        name = f"l2_{regularization:g}"
        report_path = destination / f"{name}.json"
        checkpoint = destination / f"{name}.keras"
        if report_path.exists():
            saved = json.loads(report_path.read_text())
            if saved["checkpoint_sha256"] != file_hash(checkpoint):
                raise ValueError("Completed inner-fit checkpoint changed.")
            choices.append(saved)
            continue
        model = build_head(tf, x[~held], regularization, 42)
        batches = dataset(tf, x[~held], y[~held], 42)
        history, best, stale = [], None, 0
        for epoch in range(1, plan["head"]["maximum_epochs"] + 1):
            loss = float(model.fit(batches, epochs=1, verbose=0).history["loss"][-1])
            metrics = operating_metrics(y[held], predict(model, x[held]), .95)
            record = {"epoch": epoch, "loss": loss, **metrics}
            history.append(record)
            score = (metrics["specificity"], metrics["pr_auc"])
            if best is None or score > (best["specificity"], best["pr_auc"]):
                best, stale = record, 0
                save_model(model, checkpoint)
            else:
                stale += 1
            if epoch % 10 == 0 or epoch == 1:
                print(f"{encoder} {name} epoch {epoch}: inner AP={metrics['pr_auc']:.5f}, "
                      f"specificity@95={metrics['specificity']:.5f}", flush=True)
            save_json(destination / f"{name}_history.json", {"history": history})
            save_json(destination / "progress.json", {"pid": os.getpid(), "phase": "inner_selection",
                      "regularization": regularization, "epoch": epoch, "best_epoch": best["epoch"]})
            if epoch >= plan["head"]["minimum_epochs"] and stale >= plan["head"]["patience"]:
                break
        selected = {"regularization": regularization, "epoch": best["epoch"],
                    "metrics": best, "checkpoint_sha256": file_hash(checkpoint),
                    "selection_data": "original training split inner group holdout only"}
        save_json(report_path, selected)
        choices.append(selected)
    selected = max(choices, key=lambda item: (item["metrics"]["specificity"], item["metrics"]["pr_auc"]))
    save_json(destination / "selection.json", {"selected": selected, "candidates": choices,
              "validation_used_for_hyperparameters": False, "limitation": LIMITATION})

    # Fit both predeclared seeds before opening any outer-validation features.
    models = []
    for seed in plan["head"]["seeds"]:
        checkpoint = destination / f"seed{seed}.keras"
        marker = destination / f"seed{seed}_fit.json"
        if marker.exists():
            saved = json.loads(marker.read_text())
            if saved["checkpoint_sha256"] != file_hash(checkpoint) or saved["selection"] != selected:
                raise ValueError("Completed final head changed.")
        else:
            model = build_head(tf, x, selected["regularization"], seed)
            batches = dataset(tf, x, y, seed)
            # Small head fits restart deterministically if interrupted; extraction never restarts.
            history = model.fit(batches, epochs=selected["epoch"], verbose=2).history
            save_model(model, checkpoint)
            save_json(marker, {"seed": seed, "selection": selected, "history": history,
                               "checkpoint_sha256": file_hash(checkpoint), "tensorflow": tf.__version__})
        models.append((seed, checkpoint))
    rows, features = load_features(experiment, encoder, "val")
    val, features = aggregate_features(rows, features)
    reference = pd.read_csv(REFERENCE)
    if not val[KEYS].equals(reference[KEYS]):
        raise ValueError("Outer validation identities or labels differ from the retained reference.")
    results = {}
    for seed, checkpoint in models:
        model = tf.keras.models.load_model(checkpoint)
        values = predict(model, features)
        metrics, decisions = crossfit_policy(reference, values, folds=reference.policy_fold.to_numpy())
        output = destination / f"seed{seed}_validation_predictions.csv"
        save_csv(decisions, output)
        results[str(seed)] = {"metrics": metrics, "predictions_sha256": file_hash(output),
                              "checkpoint_sha256": file_hash(checkpoint),
                              "by_source": {str(source): decision_metrics(group.target.to_numpy(), group.prediction.to_numpy())
                                            for source, group in decisions.groupby("source")}}
        print(f"{encoder} seed {seed} OUTER DEVELOPMENT: {json.dumps(metrics)}", flush=True)
    save_json(destination / "validation_report.json", {"results": results, "limitation": LIMITATION,
              "test_inference": False, "automatic_promotion": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--encoder", choices=ENCODERS, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    train(args.experiment_dir.resolve(), args.encoder, args.smoke)
