from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import operating_policy as policy
import train_multitask as training
from test_targeted_recovery import example_frame


class OperatingPolicyTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame({"source": ["source"] * 100, "lesion_id": [f"l{i:03d}" for i in range(100)],
                             "split_group_id": [f"g{i // 2:03d}" for i in range(100)], "target": [0, 1] * 50})

    def test_groups_never_cross_folds_and_order_does_not_change_membership(self):
        frame = self.frame()
        original = frame.assign(fold=policy.policy_folds(frame)).set_index("lesion_id").fold.sort_index()
        shuffled = frame.sample(frac=1, random_state=7)
        reordered = shuffled.assign(fold=policy.policy_folds(shuffled)).set_index("lesion_id").fold.sort_index()
        pd.testing.assert_series_equal(original, reordered)
        self.assertTrue(frame.assign(fold=policy.policy_folds(frame)).groupby("split_group_id").fold.nunique().eq(1).all())

    def test_scoring_labels_do_not_choose_their_own_threshold(self):
        frame = self.frame()
        folds = policy.policy_folds(frame)
        values = np.random.default_rng(9).uniform(0.05, 0.95, len(frame))
        _, first = policy.crossfit_policy(frame, values, folds=folds)
        changed = frame.copy()
        changed.loc[folds == 0, "target"] = 1 - changed.loc[folds == 0, "target"]
        _, second = policy.crossfit_policy(changed, values, folds=folds)
        np.testing.assert_array_equal(first.loc[folds == 0, "threshold"], second.loc[folds == 0, "threshold"])

    def test_overlap_missing_and_invalid_predictions_are_rejected(self):
        frame = self.frame()
        folds = policy.policy_folds(frame)
        broken = folds.copy()
        broken[0] = (broken[1] + 1) % 5
        with self.assertRaisesRegex(ValueError, "crosses"):
            policy.crossfit_policy(frame, np.ones(100) * 0.5, folds=broken)
        with self.assertRaises(ValueError):
            policy.crossfit_policy(frame, np.full(100, np.nan), folds=folds)
        with self.assertRaises(ValueError):
            policy.crossfit_policy(frame.assign(split_group_id=None), np.full(100, 0.5), folds=folds)

    def test_crossfit_recall_is_measured_not_assumed(self):
        frame = self.frame()
        values = np.random.default_rng(42).random(100)
        metrics, predictions = policy.crossfit_policy(frame, values)
        self.assertEqual(metrics["false_negatives"], int(((predictions.target == 1) & (predictions.prediction == 0)).sum()))
        self.assertEqual(metrics["recall"], 1 - metrics["false_negatives"] / 50)

    def test_infeasible_recall_cannot_beat_feasible_specificity(self):
        self.assertLess(policy.selection_score({"recall": .94, "specificity": 1}),
                        policy.selection_score({"recall": .95, "specificity": 0}))

    def test_gate_requires_real_specificity_gain_without_extra_misses(self):
        baseline = {"recall": .96, "specificity": .7, "pr_auc": .56, "roc_auc": .93}
        self.assertFalse(policy.promotion_gate(baseline, baseline)["passes"])
        better = {**baseline, "specificity": .72}
        self.assertTrue(policy.promotion_gate(baseline, better)["passes"])
        for update in ({"recall": .95}, {"pr_auc": .55}, {"roc_auc": .92}):
            self.assertFalse(policy.promotion_gate(baseline, {**better, **update})["passes"])

    def test_paired_uncertainty_rejects_different_folds(self):
        frame = self.frame()
        _, output = policy.crossfit_policy(frame, np.linspace(0, 1, 100))
        result = policy.paired_decision_intervals(output, output, iterations=100)
        self.assertEqual(result["delta_intervals_95"]["specificity"], {"lower": 0.0, "upper": 0.0})
        with self.assertRaises(ValueError):
            policy.paired_decision_intervals(output, output.assign(policy_fold=7), iterations=100)


class TrainingStabilityTests(unittest.TestCase):
    def test_real_operating_fit_archives_epochs_and_preserves_frozen_norm(self):
        tf = training.tf
        frame = OperatingPolicyTests().frame()
        image_input, metadata_input = tf.keras.Input((3, 3, 3), name="image"), tf.keras.Input((2,), name="metadata")
        features = tf.keras.layers.Concatenate()([tf.keras.layers.Flatten()(image_input), metadata_input])
        norm = tf.keras.layers.BatchNormalization(name="head_bn")
        features = norm(features)
        model = tf.keras.Model({"image": image_input, "metadata": metadata_input}, {
            "high_risk": tf.keras.layers.Dense(1, activation="sigmoid", name="high_risk")(features),
            "diagnosis": tf.keras.layers.Dense(8, activation="softmax", name="diagnosis")(features)})
        training.freeze_batch_norm_layers(model)
        old_weights = [value.copy() for value in norm.get_weights()]
        model.compile(optimizer="adam", loss={"high_risk": "binary_crossentropy", "diagnosis": "categorical_crossentropy"})
        frame["diagnosis"] = np.where(frame.target.eq(1), "MEL", "NV")
        frame["diagnosis_index"] = frame.diagnosis.map(training.DIAGNOSIS_TO_INDEX)
        frame = training.configure_binary_target(frame, "melanoma")
        inputs = {"image": tf.ones((100, 3, 3, 3)), "metadata": tf.ones((100, 2))}
        labels = {"high_risk": frame.target.to_numpy(dtype="float32").reshape(-1, 1),
                  "diagnosis": tf.one_hot(frame.diagnosis_index.to_numpy(), 8)}
        ds = tf.data.Dataset.from_tensor_slices((inputs, labels)).batch(4)
        with TemporaryDirectory() as directory:
            run = Path(directory)
            callbacks = training.training_callbacks(run, "head", 3, archive_epochs=True,
                lesion_validation=training.LesionValidation(frame, ds, "melanoma", .95, operating=True))
            model.fit(ds, epochs=2, steps_per_epoch=1, callbacks=callbacks, verbose=0)
            self.assertTrue((run / "epoch_head_01.keras").is_file())
            self.assertTrue((run / "epoch_head_02.keras").is_file())
            history = pd.read_csv(run / "training_head.csv")
            self.assertTrue(np.isfinite(history.val_operating_score).all())
        for old, new in zip(old_weights, norm.get_weights()):
            np.testing.assert_array_equal(old, new)
        tf.keras.backend.clear_session()

    def test_nested_batchnorm_statistics_and_weights_stay_frozen(self):
        tf = training.tf
        nested = tf.keras.Sequential([tf.keras.Input((3,)), tf.keras.layers.BatchNormalization(name="nested_bn")])
        model = tf.keras.Sequential([tf.keras.Input((3,)), nested,
                                     tf.keras.layers.BatchNormalization(name="head_bn"), tf.keras.layers.Dense(1)])
        names = training.freeze_batch_norm_layers(model)
        self.assertEqual(len(names), 2)
        layers = [nested.layers[0], model.get_layer("head_bn")]
        before = [[value.copy() for value in layer.get_weights()] for layer in layers]
        model.compile(optimizer="adam", loss="mse")
        model.train_on_batch(np.ones((4, 3)) * 7, np.ones((4, 1)))
        for layer, weights in zip(layers, before):
            self.assertFalse(layer.trainable)
            for old, new in zip(weights, layer.get_weights()):
                np.testing.assert_array_equal(old, new)

    def test_epoch_archives_survive_a_worse_later_metric(self):
        tf = training.tf
        model = tf.keras.Sequential([tf.keras.Input((2,)), tf.keras.layers.Dense(1)])
        with TemporaryDirectory() as directory:
            run = Path(directory)
            callbacks = training.training_callbacks(run, "finetune", 3, archive_epochs=True)
            archive = callbacks[0]
            archive.set_model(model)
            archive.on_epoch_end(0, {"val_high_risk_pr_auc": .8})
            archive.on_epoch_end(1, {"val_high_risk_pr_auc": .7})
            self.assertTrue((run / "epoch_finetune_01.keras").exists())
            self.assertTrue((run / "epoch_finetune_02.keras").exists())

    def test_operating_callback_uses_fixed_blend_and_group_crossfit(self):
        frame = example_frame()
        from unittest.mock import Mock
        model = Mock()
        risk = np.where(frame.target.eq(1), .9, .1)
        diagnosis = np.zeros((len(frame), 8))
        diagnosis[:, 4], diagnosis[:, 5] = risk, 1 - risk
        model.predict.return_value = {"high_risk": risk, "diagnosis": diagnosis}
        callback = training.LesionValidation(frame, None, "melanoma", .95, operating=True)
        callback.set_model(model)
        logs = {}
        callback.on_epoch_end(0, logs)
        self.assertEqual(callback.monitor, "val_operating_score")
        self.assertEqual(logs["val_crossfit_recall"], 1)
        self.assertEqual(logs["val_operating_score"], 1)

    def test_legacy_defaults_and_invalid_finalizer_are_explicit(self):
        with patch("sys.argv", ["train_multitask.py"]):
            args = training.parse_args()
        self.assertFalse(args.freeze_batch_norm)
        self.assertFalse(args.archive_epoch_checkpoints)
        with TemporaryDirectory() as directory:
            args.manifest = Path(directory) / "manifest.csv"
            args.manifest.touch()
            args.checkpoint_metric = "lesion_operating"
            with self.assertRaisesRegex(ValueError, "validation-only"):
                training.validate_args(args)


if __name__ == "__main__":
    unittest.main()
