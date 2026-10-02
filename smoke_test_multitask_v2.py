from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import tensorflow as tf

import train_multitask as training


def main() -> None:
    training.set_random_seeds(123)
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    metadata_size = 2 + len(training.SEX_CATEGORIES) + len(training.SITE_CATEGORIES)
    model = training.build_model(
        image_size=96,
        metadata_size=metadata_size,
        dropout_rate=0.20,
        metadata_modality_dropout=0.50,
        risk_features="image_only",
        backbone_name="efficientnetv2b0",
        weights=None,
    )
    training.compile_model(
        model,
        learning_rate=1e-3,
        weight_decay=1e-5,
        label_smoothing=0.01,
        diagnosis_loss_weight=0.50,
        risk_loss_name="focal",
        focal_gamma=1.5,
        gradient_accumulation_steps=2,
    )

    layer_names = {layer.name for layer in model.layers}
    forbidden = {"direct_risk", "diagnosis_risk", "risk_pair"}
    assert not (layer_names & forbidden), "The fixed blended-risk architecture is still present."

    images = np.random.default_rng(123).random((2, 96, 96, 3), dtype=np.float32)
    augmentation = model.get_layer("augmentation")
    assert all(layer.compute_dtype == "float32" for layer in augmentation.layers)
    assert augmentation(images, training=True).dtype == tf.float32

    metadata_a = np.zeros((2, metadata_size), dtype=np.float32)
    metadata_b = np.ones((2, metadata_size), dtype=np.float32)
    output_a = model({"image": images, "metadata": metadata_a}, training=False)
    output_b = model({"image": images, "metadata": metadata_b}, training=False)
    np.testing.assert_allclose(
        output_a["high_risk"].numpy(),
        output_b["high_risk"].numpy(),
        atol=1e-7,
        err_msg="Primary risk output must not depend on metadata in image-only mode.",
    )

    high_risk = model.get_layer("high_risk")
    kernel, bias = high_risk.get_weights()
    high_risk.set_weights([np.zeros_like(kernel), np.full_like(bias, 12.0)])
    uncapped = model({"image": images, "metadata": metadata_a}, training=False)["high_risk"]
    assert float(tf.reduce_min(uncapped)) > 0.99, "Risk output is still capped below one."

    labels = {
        "high_risk": np.array([[0.0], [1.0]], dtype=np.float32),
        "diagnosis": tf.one_hot([2, 4], depth=len(training.DIAGNOSES)).numpy(),
    }
    sample_weights = {
        "high_risk": np.ones(2, dtype=np.float32),
        "diagnosis": np.array([0.0, 1.0], dtype=np.float32),
    }
    model.train_on_batch(
        {"image": images, "metadata": metadata_a},
        labels,
        sample_weight=sample_weights,
    )

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "smoke.keras"
        model.save(path)
        restored = tf.keras.models.load_model(path, compile=False)
        restored_output = restored(
            {"image": images, "metadata": metadata_a}, training=False
        )["high_risk"]
        assert restored_output.shape == (2, 1)

    print("Multitask v2 architecture, loss, training step, and serialization passed.")


if __name__ == "__main__":
    main()
