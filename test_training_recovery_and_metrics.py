from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import tensorflow as tf

import train_multitask as training
from training_callback_state import TrainingStateBackup


class DiagnosisSupervisionTests(unittest.TestCase):
    def test_unknown_dummy_index_is_not_scored_as_actinic_keratosis(self):
        predictions = np.eye(8)[[4, 4]]
        result = training.diagnosis_metrics(np.array([4, 0]), predictions, np.array([1, 0]))
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["support"], 1)
        self.assertEqual(result["excluded_unlabeled"], 1)
        self.assertEqual(np.sum(result["confusion_matrix"]), 1)

    def test_no_supervision_produces_no_fabricated_metrics(self):
        result = training.diagnosis_metrics(np.array([0]), np.eye(8)[[4]], np.array([0]))
        self.assertIsNone(result["accuracy"])
        self.assertEqual(result["status"], "no_supervised_diagnosis_labels")
        self.assertEqual(np.sum(result["confusion_matrix"]), 0)

    def test_aggregation_retains_mask_and_legacy_unknown_fallback(self):
        frame = pd.DataFrame({"lesion_id": ["a", "a", "b"], "source": ["x"] * 3,
                              "target": [1, 1, 0], "target_name": ["MEL", "MEL", "NOT_MEL"],
                              "diagnosis": ["MEL", "MEL", "UNKNOWN"], "diagnosis_index": [4, 4, 0]})
        lesions, _ = training.aggregate_predictions_by_lesion(frame, np.array([.8, .9, .1]))
        self.assertEqual(lesions.diagnosis_label_available.tolist(), [1, 0])
        frame["diagnosis_label_available"] = [1, 0, 0]
        with self.assertRaisesRegex(ValueError, "conflicting"):
            training.aggregate_predictions_by_lesion(frame, np.array([.8, .9, .1]))

    def test_invalid_mask_rejected_and_legacy_known_labels_unchanged(self):
        truth = np.array([0, 4])
        predictions = np.eye(8)[truth]
        self.assertEqual(training.diagnosis_metrics(truth, predictions)["accuracy"], 1.0)
        for mask in (np.array([1]), np.array([1, .5]), np.array([1, np.nan])):
            with self.subTest(mask=mask), self.assertRaises(ValueError):
                training.diagnosis_metrics(truth, predictions, mask)


class TrainingRecoveryTests(unittest.TestCase):
    def test_backup_epoch_takes_precedence_over_ahead_log_and_preserves_audit(self):
        with TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = Path(directory)
            (path / "backup_head").mkdir()
            (path / "backup_head" / "training_metadata.json").write_text('{"epoch": 2}')
            pd.DataFrame({"epoch": [0, 1, 2], "val_high_risk_pr_auc": [.8, .7, .6]}).to_csv(
                path / "training_head.csv", index=False)
            self.assertEqual(training.stage_resume_epoch(path, "head"), 2)
            training.training_callbacks(path, "head", patience=2, append_history=True)
            self.assertEqual(pd.read_csv(path / "training_head.csv").epoch.tolist(), [0, 1])
            self.assertEqual(pd.read_csv(path / "training_head_unbacked_epochs.csv").epoch.tolist(), [2])

    def model(self):
        model = tf.keras.Sequential([tf.keras.Input((1,)), tf.keras.layers.Dense(1, use_bias=False)])
        model.compile(tf.keras.optimizers.SGD(.01), loss="mse")
        model.optimizer.build(model.trainable_variables)
        return model

    def begin(self, callbacks, model):
        model.stop_training = False
        for callback in callbacks:
            callback.set_model(model)
            callback.set_params({"epochs": 6, "steps": 1, "verbose": 0})
            callback.on_train_begin({})

    def epoch(self, callbacks, epoch, metric):
        logs = {"val_high_risk_pr_auc": metric}
        for callback in callbacks:
            callback.on_epoch_end(epoch, logs)

    def test_resume_matches_uninterrupted_lr_patience_and_best_weights(self):
        with TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = Path(directory)
            model = self.model()
            callbacks = training.training_callbacks(path, "head", patience=2)
            self.assertIsInstance(callbacks[-1], TrainingStateBackup)
            self.begin(callbacks, model)
            model.set_weights([np.array([[2.]], dtype="float32")])
            self.epoch(callbacks, 0, .8)
            model.set_weights([np.array([[3.]], dtype="float32")])
            self.epoch(callbacks, 1, .7)
            self.assertAlmostEqual(float(model.optimizer.learning_rate.numpy()), .004)
            for callback in callbacks:
                if isinstance(callback, tf.keras.callbacks.CSVLogger):
                    callback.on_train_end({})
            restored = self.model()
            resumed_callbacks = training.training_callbacks(path, "head", patience=2, append_history=True,
                                                            initial_best=.8)
            self.begin(resumed_callbacks, restored)
            self.assertAlmostEqual(float(restored.optimizer.learning_rate.numpy()), .004)
            np.testing.assert_array_equal(restored.get_weights()[0], [[3.]])
            state = resumed_callbacks[-1].controllers
            self.assertEqual(state["stopping"].wait, 1)
            self.assertEqual(state["stopping"].best, .8)
            self.assertEqual(state["plateau"].best, .8)
            self.epoch(resumed_callbacks, 2, .6)
            self.assertTrue(restored.stop_training)
            self.assertAlmostEqual(float(restored.optimizer.learning_rate.numpy()), .0016)
            for callback in resumed_callbacks:
                callback.on_train_end({})
            np.testing.assert_array_equal(restored.get_weights()[0], [[2.]])
            self.assertFalse((path / "backup_head").exists())
            # Close the interrupted logger without invoking its backup cleanup.
            for callback in callbacks:
                if isinstance(callback, tf.keras.callbacks.CSVLogger):
                    callback.on_train_end({})

    def test_legacy_recovery_without_counters_is_explicit(self):
        with TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = Path(directory)
            model = self.model()
            callbacks = training.training_callbacks(path, "head", patience=2)
            self.begin(callbacks, model)
            self.epoch(callbacks, 0, .8)
            for callback in callbacks:
                if isinstance(callback, tf.keras.callbacks.CSVLogger):
                    callback.on_train_end({})
            metadata_path = path / "backup_head" / "training_metadata.json"
            metadata = json.loads(metadata_path.read_text())
            metadata.pop("validation_callback_state")
            metadata_path.write_text(json.dumps(metadata))
            resumed_callbacks = training.training_callbacks(path, "head", patience=2, append_history=True)
            with self.assertWarnsRegex(RuntimeWarning, "continuation is not exact"):
                self.begin(resumed_callbacks, self.model())
            for callback in callbacks + resumed_callbacks:
                if isinstance(callback, tf.keras.callbacks.CSVLogger):
                    callback.on_train_end({})


if __name__ == "__main__":
    unittest.main()
