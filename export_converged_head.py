"""Export converged logistic parameters as a verified TensorFlow feature classifier."""
import argparse
from pathlib import Path

import numpy as np

from artifact_io import save_model
from compare_model_runs import file_hash
from converged_probe import load_parameters, probabilities
from foundation_common import save_json


def export(parameters_path, output_path, verification_features):
    import tensorflow as tf
    tf.keras.backend.clear_session()
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    parameters = load_parameters(parameters_path)
    inputs = tf.keras.Input((len(parameters["weights"]),), dtype="float64", name="frozen_image_embedding")
    normalized = tf.keras.layers.Normalization(mean=parameters["mean"].astype("float64"),
        variance=parameters["scale"] ** 2, dtype="float64", name="training_only_normalization")(inputs)
    dense = tf.keras.layers.Dense(1, activation="sigmoid", dtype="float64", name="melanoma")
    model = tf.keras.Model(inputs, dense(normalized), name="converged_melanoma_feature_head")
    dense.set_weights([parameters["weights"][:, None], np.atleast_1d(parameters["bias"])])
    save_model(model, output_path)
    restored = tf.keras.models.load_model(output_path, compile=False)
    expected = probabilities(parameters, verification_features)
    actual = restored(np.asarray(verification_features, dtype="float64"), training=False).numpy().ravel()
    np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-8)
    report = {"tensorflow": tf.__version__, "source_parameters_sha256": file_hash(parameters_path),
        "checkpoint_sha256": file_hash(output_path), "maximum_probability_difference": float(np.max(np.abs(actual - expected))),
        "verified_rows": len(actual), "input": "Frozen lesion embeddings, not raw images",
        "decision_threshold_included": False, "automatic_promotion": False}
    save_json(output_path.with_suffix(".export.json"), report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--arm", choices=("incumbent", "panderm_base", "fusion"), required=True)
    args = parser.parse_args()
    from run_convergence_study import load_split, SOURCE
    _, features = load_split(SOURCE, "train")
    report = export(args.study_dir / f"final_{args.arm}_start0.npz",
        args.study_dir / f"{args.arm}_research_head.keras", features[args.arm][:512])
    print(report)
