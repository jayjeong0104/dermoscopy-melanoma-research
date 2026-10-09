from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import tensorflow as tf

import train_multitask as training


class ValidationPassPolicyTests(unittest.TestCase):
    def test_default_preserves_validation_and_guard_requires_lesion_selection_without_finalization(self):
        with patch("sys.argv", ["train_multitask.py"]):
            args = training.parse_args()
        self.assertFalse(args.skip_keras_validation)
        dataset = object()
        self.assertIs(training.validation_data_for_fit(args, dataset), dataset)
        args.skip_keras_validation = True
        for metric, finalization in (("image_pr_auc", "none"), ("lesion_pr_auc", "fresh_process"),
                                     ("lesion_pr_auc", "in_process")):
            args.checkpoint_metric, args.finalization_mode = metric, finalization
            with self.assertRaisesRegex(ValueError, "skip-keras-validation"):
                training.validation_data_for_fit(args, dataset)
        for metric in ("lesion_pr_auc", "lesion_operating"):
            args.checkpoint_metric, args.finalization_mode = metric, "none"
            self.assertIsNone(training.validation_data_for_fit(args, dataset))

    def test_legacy_resume_defaults_off_and_cannot_silently_enable(self):
        with TemporaryDirectory(dir=Path(__file__).parent) as directory:
            run = Path(directory)
            config = {key: None for key in training.RESUME_CONFIG_KEYS}
            config.pop("skip_keras_validation")
            encoder = training.MetadataEncoder.fit(pd.DataFrame({"age_approx": [40]}))
            (run / "run_config.json").write_text(json.dumps(config))
            (run / "metadata_schema.json").write_text(json.dumps(asdict(encoder)))
            (run / "split_summary.json").write_text("{}")
            training.validate_resume_state(run, config, encoder, {})
            with self.assertRaisesRegex(ValueError, "skip_keras_validation"):
                training.validate_resume_state(run, dict(config, skip_keras_validation=True), encoder, {})

    def test_skip_omits_only_duplicate_test_steps_with_same_weights_and_selection_metrics(self):
        previous = tf.keras.mixed_precision.global_policy()
        tf.keras.mixed_precision.set_global_policy("float32")
        x = np.array([[-1., 0.], [-.5, .1], [.5, .1], [1., 0.]], dtype="float32")
        y = np.array([0, 0, 1, 1], dtype="float32")
        labels = {"high_risk": y[:, None], "diagnosis": tf.one_hot([5, 5, 4, 4], 8)}
        ds = tf.data.Dataset.from_tensor_slices((x, labels)).batch(2)
        options = tf.data.Options()
        options.threading.private_threadpool_size = 1
        ds = ds.with_options(options)
        frame = pd.DataFrame({"source": ["synthetic"] * 4, "image_id": list("abcd"),
            "lesion_id": list("abcd"), "split_group_id": list("abcd"), "target": y.astype(int),
            "target_name": ["not_melanoma"] * 2 + ["melanoma"] * 2,
            "diagnosis": ["NV", "NV", "MEL", "MEL"], "diagnosis_index": [5, 5, 4, 4]})
        outcomes = []
        try:
            for skip in (False, True):
                tf.keras.utils.set_random_seed(42)
                inputs = tf.keras.Input((2,))
                model = tf.keras.Model(inputs, {
                    "high_risk": tf.keras.layers.Dense(1, activation="sigmoid", name="high_risk")(inputs),
                    "diagnosis": tf.keras.layers.Dense(8, activation="softmax", name="diagnosis")(inputs),
                })
                model.compile(tf.keras.optimizers.SGD(.01), loss={"high_risk": "binary_crossentropy",
                    "diagnosis": "categorical_crossentropy"})
                counter = tf.Variable(0, dtype=tf.int32)
                original = model.test_step

                def counted_test_step(data, counter=counter, original=original):
                    counter.assign_add(1)
                    return original(data)

                model.test_step = counted_test_step
                args = argparse.Namespace(skip_keras_validation=skip, checkpoint_metric="lesion_pr_auc", finalization_mode="none")
                with TemporaryDirectory(dir=Path(__file__).parent) as directory:
                    callbacks = training.training_callbacks(Path(directory), "head", patience=3,
                        lesion_validation=training.LesionValidation(frame, ds, "melanoma", .95))
                    history = model.fit(ds, epochs=2, verbose=0,
                        validation_data=training.validation_data_for_fit(args, ds), callbacks=callbacks)
                    outcomes.append((int(counter.numpy()), model.get_weights(), history.history))
            self.assertEqual([result[0] for result in outcomes], [4, 0])
            for first, second in zip(outcomes[0][1], outcomes[1][1]):
                np.testing.assert_array_equal(first, second)
            for name in ("val_lesion_pr_auc", "val_lesion_specificity", "val_lesion_recall"):
                np.testing.assert_array_equal(outcomes[0][2][name], outcomes[1][2][name])
            self.assertIn("val_loss", outcomes[0][2])
            self.assertNotIn("val_loss", outcomes[1][2])
        finally:
            tf.keras.mixed_precision.set_global_policy(previous)


if __name__ == "__main__":
    unittest.main()
