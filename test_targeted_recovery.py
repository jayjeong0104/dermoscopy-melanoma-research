from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock

import numpy as np
import pandas as pd

import train_multitask as training
from finalize_multitask_run import prediction_input_fingerprint, read_prediction_cache, write_prediction_cache
from targeted_sampling import assert_training_only, attach_weights, mine_weights


def example_frame() -> pd.DataFrame:
    frame = pd.DataFrame({
        "source": ["A"] * 10 + ["B"] * 10 + ["A"] * 5,
        "image_id": [f"i{i}" for i in range(25)],
        "lesion_id": [f"l{i}" for i in range(25)],
        "split_group_id": [f"p{i}" for i in range(25)],
        "sha256": [f"h{i}" for i in range(25)],
        "diagnosis": ["NV"] * 20 + ["MEL"] * 5,
        "target": [0] * 20 + [1] * 5,
        "age_approx": [50] * 25, "sex": ["unknown"] * 25,
        "anatom_site_general": ["unknown"] * 25,
    })
    frame["diagnosis_index"] = frame.diagnosis.map(training.DIAGNOSIS_TO_INDEX)
    return training.configure_binary_target(frame, "melanoma")


class TargetedSamplingTests(unittest.TestCase):
    def test_capped_source_stratified_mining_and_no_dropped_images(self):
        frame = example_frame()
        scores = frame.assign(probability=np.linspace(0, 1, len(frame)))
        result = mine_weights(frame, scores)
        self.assertEqual(len(result), len(frame))
        self.assertEqual(set(result.loc[result.sampling_weight.eq(2), "image_id"]), {"i8", "i9", "i18", "i19"})
        self.assertTrue(result.loc[result.target.eq(1), "sampling_weight"].eq(1).all())

    def test_other_views_of_same_lesion_receive_same_weight(self):
        frame = example_frame()
        extra = frame.iloc[[9]].assign(image_id="extra", sha256="extra_hash")
        expanded = pd.concat([frame, extra], ignore_index=True)
        result = mine_weights(expanded, frame.assign(probability=np.arange(25) / 25))
        self.assertEqual(result.loc[result.lesion_id.eq("l9"), "sampling_weight"].tolist(), [2, 2])

    def test_holdout_overlap_rejected_for_each_grouping_key(self):
        train = example_frame()
        for column in ("sha256", "lesion_id", "split_group_id"):
            held = train.iloc[[0]].copy()
            for key in ("sha256", "lesion_id", "split_group_id"):
                held[key] = "new_" + key
            held[column] = train.iloc[0][column]
            with self.subTest(column=column), self.assertRaisesRegex(ValueError, "overlap"):
                assert_training_only(train, [held])

    def test_weight_file_rejects_extra_holdout_rows_labels_and_positive_weight(self):
        frame = example_frame()
        weights = mine_weights(frame, frame.assign(probability=np.arange(25) / 25))
        with TemporaryDirectory() as directory:
            path = Path(directory) / "weights.csv"
            weights.to_csv(path, index=False)
            attached = attach_weights(frame.iloc[::-1], path)
            self.assertEqual(attached.image_id.tolist(), frame.iloc[::-1].image_id.tolist())
            for changed in (weights.iloc[:-1], weights.assign(target=1),
                            weights.assign(sampling_weight=2), weights.assign(sampling_weight=3)):
                changed.to_csv(path, index=False)
                with self.assertRaises(ValueError):
                    attach_weights(frame, path)

    def test_mining_rejects_nontraining_predictions_and_nonfinite_scores(self):
        frame = example_frame()
        for scores in (frame.iloc[:-1].assign(probability=0.5), frame.assign(probability=np.nan),
                       frame.assign(probability=0.5, target=1)):
            with self.assertRaises(ValueError):
                mine_weights(frame, scores)

    def test_sampler_preserves_class_and_source_mass(self):
        tf = training.tf
        frame = example_frame()
        frame["sampling_weight"] = 1.0
        frame.loc[frame.image_id.isin(["i8", "i9", "i18", "i19"]), "sampling_weight"] = 2.0
        # Metadata encodes strata and hard status so sampling is tested through the real decoder.
        frame["age_approx"] = np.where(frame.target.eq(1), 90,
                                       np.where(frame.source.eq("A"), 20, 50) + frame.sampling_weight * 5)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "image.jpg"
            path.write_bytes(tf.io.encode_jpeg(tf.zeros((4, 4, 3), dtype=tf.uint8)).numpy())
            frame["runtime_image_path"] = str(path)
            ds = training.build_dataset(frame, training.MetadataEncoder.fit(frame), 4, 100,
                                        True, 1, 42, 0.5, 0.45)
            batches = list(ds.take(40).as_numpy_iterator())
            ages = np.concatenate([batch[0]["metadata"][:, 0] for batch in batches])
            positive = np.concatenate([batch[1]["high_risk"].ravel() for batch in batches]).astype(bool)
            self.assertAlmostEqual(float(positive.mean()), 0.45, delta=0.04)
            negative_ages = ages[~positive]
            self.assertAlmostEqual(float((negative_ages < 0.4).mean()), 0.5, delta=0.05)
            hard = np.isclose(negative_ages, 0.3) | np.isclose(negative_ages, 0.6)
            self.assertAlmostEqual(float(hard.mean()), 1 / 3, delta=0.05)


class PredictionProtocolTests(unittest.TestCase):
    def test_lesion_checkpoint_integrates_with_real_keras_fit(self):
        tf = training.tf
        frame = example_frame()
        inputs = {"image": tf.zeros((25, 3, 3, 3)), "metadata": tf.zeros((25, 2))}
        labels = {"high_risk": frame.target.to_numpy(dtype="float32").reshape(-1, 1),
                  "diagnosis": tf.one_hot(frame.diagnosis_index.to_numpy(), 8)}
        ds = tf.data.Dataset.from_tensor_slices((inputs, labels)).batch(25)
        image_input, metadata_input = tf.keras.Input((3, 3, 3), name="image"), tf.keras.Input((2,), name="metadata")
        features = tf.keras.layers.Concatenate()([tf.keras.layers.Flatten()(image_input), metadata_input])
        outputs = {"high_risk": tf.keras.layers.Dense(1, activation="sigmoid", name="high_risk")(features),
                   "diagnosis": tf.keras.layers.Dense(8, activation="softmax", name="diagnosis")(features)}
        model = tf.keras.Model({"image": image_input, "metadata": metadata_input}, outputs)
        model.compile(optimizer="adam", loss={"high_risk": "binary_crossentropy", "diagnosis": "categorical_crossentropy"})
        with TemporaryDirectory() as directory:
            run = Path(directory)
            callbacks = training.training_callbacks(run, "head", 3, lesion_validation=
                training.LesionValidation(frame, ds, "melanoma", 0.95))
            model.fit(ds, validation_data=ds, epochs=1, callbacks=callbacks, verbose=0)
            self.assertTrue((run / "best_head_model.keras").is_file())
            history = pd.read_csv(run / "training_head.csv")
            self.assertTrue(np.isfinite(history.val_lesion_pr_auc).all())
        tf.keras.backend.clear_session()

    def test_corrected_eight_views_are_unique_and_legacy_is_preserved(self):
        tf = training.tf
        original = tf.reshape(tf.range(27, dtype=tf.float32), (1, 3, 3, 3))
        inputs = {"image": original, "metadata": tf.ones((1, 2))}
        labels, weights = {"high_risk": tf.ones((1, 1))}, {"high_risk": tf.ones((1,))}
        ds = tf.data.Dataset.from_tensors((inputs, labels, weights))
        views = []
        for index in range(8):
            transformed = next(iter(training.transform_tta_dataset(ds, index, "d4_v2")))
            array = transformed[0]["image"].numpy()
            views.append(array.tobytes())
            np.testing.assert_array_equal(np.sort(array.ravel()), original.numpy().ravel())
            np.testing.assert_array_equal(transformed[2]["high_risk"], weights["high_risk"])
        self.assertEqual(len(set(views)), 8)
        legacy = [next(iter(training.transform_tta_dataset(ds, i)))[0]["image"].numpy() for i in (5, 7)]
        np.testing.assert_array_equal(*legacy)

    def test_cache_fingerprint_includes_row_order_hash_and_metadata_schema(self):
        frame = example_frame().assign(runtime_image_path="image.jpg")
        encoder = training.MetadataEncoder.fit(frame)
        original = prediction_input_fingerprint(frame, encoder)
        for other in (frame.iloc[::-1], frame.assign(sha256="changed")):
            self.assertNotEqual(original, prediction_input_fingerprint(other, encoder))
        self.assertNotEqual(original, prediction_input_fingerprint(
            frame, training.MetadataEncoder(20, encoder.feature_names)))
        with TemporaryDirectory() as directory:
            path = Path(directory) / "cache.npz"
            meta = {"tta_scheme": "legacy_v1", "input_sha256": original}
            write_prediction_cache(path, np.ones(2) / 2, np.ones((2, 8)) / 8, meta)
            self.assertIsNone(read_prediction_cache(path, {**meta, "tta_scheme": "d4_v2"}, 2))

    def test_lesion_callback_runs_before_checkpoint_and_uses_aggregated_blend(self):
        frame = example_frame()
        direct = np.full(len(frame), 0.5)
        diagnosis = np.zeros((len(frame), 8), dtype="float32")
        diagnosis[:, 4] = np.where(frame.target.eq(1), 0.9, 0.1)
        diagnosis[:, 5] = 1 - diagnosis[:, 4]
        model = Mock()
        model.predict.return_value = {"high_risk": direct, "diagnosis": diagnosis}
        callback = training.LesionValidation(frame, None, "melanoma", 0.95)
        callback.set_model(model)
        logs = {"val_high_risk_pr_auc": 0.2}
        callback.on_epoch_end(0, logs)
        self.assertEqual(logs["val_lesion_pr_auc"], 1.0)
        self.assertEqual(logs["val_lesion_specificity"], 1.0)
        with TemporaryDirectory() as directory:
            callbacks = training.training_callbacks(Path(directory), "head", 3, lesion_validation=callback)
            self.assertIs(callbacks[0], callback)
            checkpoint = next(c for c in callbacks if isinstance(c, training.tf.keras.callbacks.ModelCheckpoint))
            self.assertEqual(checkpoint.monitor, "val_lesion_pr_auc")


if __name__ == "__main__":
    unittest.main()
