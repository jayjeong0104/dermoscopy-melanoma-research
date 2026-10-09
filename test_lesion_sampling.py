from pathlib import Path
from dataclasses import asdict
import json
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from lesion_sampling import exposure_audit, lesion_rows, row_dataset, stratum_mass
import train_multitask as training


def example():
    return pd.DataFrame([{"source": "S", "diagnosis": "MEL", "target": 1,
        "image_id": f"image_{lesion}_{photo:02d}", "lesion_id": f"lesion_{lesion}",
        "split_group_id": f"patient_{lesion}"}
        for lesion, count in enumerate((1, 3, 27)) for photo in range(count)])


class LesionSamplingTests(unittest.TestCase):
    def test_rows_keep_every_original_image_and_ignore_dataframe_index(self):
        frame = example().sample(frac=1, random_state=42)
        before = frame.copy()
        rows, starts, counts = lesion_rows(frame)
        np.testing.assert_array_equal(np.sort(rows), np.arange(len(frame)))
        np.testing.assert_array_equal(counts, [1, 3, 27])
        for start, count in zip(starts, counts):
            self.assertEqual(frame.iloc[rows[start:start + count]].lesion_id.nunique(), 1)
        pd.testing.assert_frame_equal(frame, before)

    def test_sampling_visits_each_lesion_equally_and_all_photos_remain_reachable(self):
        frame = example()
        indices = np.asarray(list(row_dataset(frame, 42).take(3000).as_numpy_iterator()))
        selected = frame.iloc[indices]
        self.assertEqual(selected.lesion_id.value_counts().tolist(), [1000, 1000, 1000])
        self.assertEqual(set(selected.image_id), set(frame.image_id))
        for begin in range(0, len(selected), 3):
            self.assertEqual(selected.iloc[begin:begin + 3].lesion_id.nunique(), 3)
        np.testing.assert_array_equal(indices[:60], list(row_dataset(frame, 42).take(60).as_numpy_iterator()))
        self.assertFalse(np.array_equal(indices[:60], list(row_dataset(frame, 137).take(60).as_numpy_iterator())))

    def test_duplicate_missing_conflicting_strata_and_groups_are_rejected(self):
        frame = example()
        variants = [pd.concat([frame, frame.iloc[:1]]), frame.assign(image_id=None),
                    frame.assign(target=[0] + [1] * (len(frame) - 1))]
        broken = frame.copy()
        broken.loc[2, "split_group_id"] = "wrong"
        variants.append(broken)
        for bad in variants:
            with self.assertRaises(ValueError):
                lesion_rows(bad)

    def test_exact_exposure_preserves_class_and_stratum_mass(self):
        positive = example()
        negative = example().assign(diagnosis="NV", target=0,
            image_id=lambda f: "n_" + f.image_id, lesion_id=lambda f: "n_" + f.lesion_id)
        frame = pd.concat([positive, negative], ignore_index=True)
        report = exposure_audit(frame)
        self.assertEqual(report["deleted_images"], 0)
        for stratum in report["strata"]:
            mass = .45 if stratum["target"] else .55
            self.assertAlmostEqual(stratum["source_diagnosis_probability"], mass)
            self.assertAlmostEqual(stratum["new_each_lesion_probability"] * 3, mass)
            self.assertAlmostEqual(stratum["old_max_lesion_probability"] / stratum["old_min_lesion_probability"], 27)

    def test_build_dataset_decodes_and_preserves_labels_metadata_and_eval_order(self):
        tf = training.tf
        frame = example()
        frame["diagnosis_index"] = training.DIAGNOSIS_TO_INDEX["MEL"]
        frame["age_approx"], frame["sex"], frame["anatom_site_general"] = 50, "unknown", "unknown"
        encoder = training.MetadataEncoder.fit(frame)
        with TemporaryDirectory() as directory:
            image_path = Path(directory) / "image.jpg"
            tf.io.write_file(str(image_path), tf.io.encode_jpeg(tf.ones((20, 20, 3), tf.uint8) * 128))
            frame["runtime_image_path"] = str(image_path)
            # Age acts as a row identity, verifying the complete row travels together.
            frame["age_approx"] = np.arange(len(frame)) + 20
            ds = training.build_dataset(frame, encoder, 20, 3, True, 1, 42, sampling_unit="lesion")
            for features, labels, weights in ds.take(2):
                self.assertEqual(features["image"].shape, (3, 20, 20, 3))
                np.testing.assert_array_equal(labels["high_risk"], np.ones((3, 1)))
                np.testing.assert_array_equal(np.argmax(labels["diagnosis"], axis=1), [4] * 3)
                np.testing.assert_array_equal(weights["high_risk"], np.ones(3))
            old = training.build_dataset(frame, encoder, 20, 4, False, 1, 42)
            new = training.build_dataset(frame, encoder, 20, 4, False, 1, 42, sampling_unit="lesion")
            for old_batch, new_batch in zip(old, new):
                for old_value, new_value in zip(tf.nest.flatten(old_batch), tf.nest.flatten(new_batch)):
                    np.testing.assert_array_equal(old_value, new_value)
            image_input = tf.keras.Input((20, 20, 3), name="image")
            metadata_input = tf.keras.Input((len(encoder.feature_names),), name="metadata")
            features = tf.keras.layers.Concatenate()([
                tf.keras.layers.GlobalAveragePooling2D()(image_input), metadata_input])
            model = tf.keras.Model({"image": image_input, "metadata": metadata_input}, {
                "high_risk": tf.keras.layers.Dense(1, activation="sigmoid", name="high_risk")(features),
                "diagnosis": tf.keras.layers.Dense(8, activation="softmax", name="diagnosis")(features)})
            model.compile(optimizer="adam", loss={"high_risk": "binary_crossentropy", "diagnosis": "categorical_crossentropy"})
            history = model.fit(ds, epochs=1, steps_per_epoch=2, verbose=0)
            self.assertTrue(np.isfinite(history.history["loss"]).all())
        training.tf.keras.backend.clear_session()

    def test_legacy_default_and_incompatible_weights(self):
        with patch("sys.argv", ["train_multitask.py"]):
            args = training.parse_args()
        self.assertEqual(args.sampling_unit, "image")
        self.assertEqual(args.sampling_mass_unit, "image")
        self.assertIn("sampling_unit", training.RESUME_CONFIG_KEYS)
        self.assertIn("sampling_mass_unit", training.RESUME_CONFIG_KEYS)
        with TemporaryDirectory() as directory:
            args.manifest = Path(directory) / "manifest.csv"
            args.manifest.touch()
            args.sampling_unit = "lesion"
            args.hard_negative_weights = Path(directory) / "weights.csv"
            args.hard_negative_weights.touch()
            with self.assertRaisesRegex(ValueError, "hard-negative"):
                training.validate_args(args)

    def test_natural_lesion_mass_is_invariant_to_repeated_photographs(self):
        positive = example()
        negatives = example().assign(source="N", diagnosis="NV", target=0,
            image_id=lambda f: "n_" + f.image_id, lesion_id=lambda f: "n_" + f.lesion_id)
        negative_rare = negatives.iloc[:1].assign(source="R", diagnosis="DF",
            image_id="r", lesion_id="rare")
        frame = pd.concat([positive, negatives, negative_rare], ignore_index=True)
        report = exposure_audit(frame, alpha=1, positive_fraction=.25, mass_unit="lesion")
        positive_mass = sum(row["source_diagnosis_probability"] for row in report["strata"] if row["target"])
        self.assertAlmostEqual(positive_mass, .25)
        negative_rates = [row["new_each_lesion_probability"] for row in report["strata"] if not row["target"]]
        np.testing.assert_allclose(negative_rates, [.75 / 4] * 2)
        duplicated_photos = pd.concat([frame, frame.loc[frame.source.eq("N")].assign(
            image_id=lambda f: "copy_" + f.image_id)], ignore_index=True)
        other = exposure_audit(duplicated_photos, alpha=1, positive_fraction=.25, mass_unit="lesion")
        np.testing.assert_allclose([s["source_diagnosis_probability"] for s in report["strata"]],
                                   [s["source_diagnosis_probability"] for s in other["strata"]])
        self.assertEqual(stratum_mass(negatives, 1), len(negatives))
        self.assertEqual(stratum_mass(negatives, 1, "lesion"), 3)

    def test_invalid_sampling_mass_inputs_are_rejected(self):
        for unit, alpha in [("patient", 1), ("image", float("nan")), ("lesion", 2)]:
            with self.assertRaises(ValueError):
                stratum_mass(example(), alpha, unit)
        with self.assertRaisesRegex(ValueError, "identities"):
            stratum_mass(example().assign(lesion_id=None), 1, "lesion")

    def test_legacy_resume_keeps_image_mass_and_rejects_policy_change(self):
        with patch("sys.argv", ["train_multitask.py"]):
            args = training.parse_args()
        config = json.loads(json.dumps(vars(args), default=str))
        saved = dict(config)
        saved.pop("sampling_mass_unit")
        frame = example().assign(age_approx=50, sex="unknown", anatom_site_general="unknown")
        encoder = training.MetadataEncoder.fit(frame)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "run_config.json").write_text(json.dumps(saved), encoding="utf-8")
            (root / "metadata_schema.json").write_text(json.dumps(asdict(encoder)), encoding="utf-8")
            (root / "split_summary.json").write_text("{}", encoding="utf-8")
            training.validate_resume_state(root, config, encoder, {})
            config["sampling_mass_unit"] = "lesion"
            with self.assertRaisesRegex(ValueError, "sampling_mass_unit"):
                training.validate_resume_state(root, config, encoder, {})

    def test_realized_mass_counts_lesions_instead_of_repeated_photos(self):
        tf = training.tf
        a = example().query("lesion_id == 'lesion_2'").assign(source="A")
        b = example().drop_duplicates("lesion_id").assign(source="B",
            image_id=lambda f: "b_" + f.image_id, lesion_id=lambda f: "b_" + f.lesion_id)
        frame = pd.concat([a, b], ignore_index=True)
        frame["diagnosis_index"] = training.DIAGNOSIS_TO_INDEX["MEL"]
        frame["age_approx"], frame["sex"], frame["anatom_site_general"] = 50, "unknown", "unknown"
        encoder = training.MetadataEncoder.fit(frame)
        with TemporaryDirectory() as directory:
            paths = {s: str(Path(directory) / f"{s}.jpg") for s in ("A", "B")}
            for source, intensity in (("A", 20), ("B", 220)):
                tf.io.write_file(paths[source], tf.io.encode_jpeg(tf.ones((20, 20, 3), tf.uint8) * intensity))
            frame["runtime_image_path"] = frame.source.map(paths)
            for mass_unit, expected in (("image", .9), ("lesion", .25)):
                ds = training.build_dataset(frame, encoder, 20, 32, True, 1, 42,
                    sampling_alpha=1, sampling_unit="lesion", sampling_mass_unit=mass_unit)
                observed = []
                for inputs, _, _ in ds.take(40):
                    observed.extend((tf.reduce_mean(inputs["image"], axis=(1, 2, 3)).numpy() < .5).tolist())
                self.assertLess(abs(np.mean(observed) - expected), .05)

    def test_multiple_strata_keep_images_labels_and_class_probability_aligned(self):
        tf = training.tf
        positive = example()
        negative = example().assign(diagnosis="NV", target=0,
            image_id=lambda f: "n_" + f.image_id, lesion_id=lambda f: "n_" + f.lesion_id)
        frame = pd.concat([positive, negative], ignore_index=True)
        frame["diagnosis_index"] = frame.diagnosis.map(training.DIAGNOSIS_TO_INDEX)
        frame["age_approx"], frame["sex"], frame["anatom_site_general"] = 50, "unknown", "unknown"
        encoder = training.MetadataEncoder.fit(frame)
        with TemporaryDirectory() as directory:
            paths = [str(Path(directory) / f"class_{label}.jpg") for label in (0, 1)]
            for label, intensity in ((0, 220), (1, 20)):
                tf.io.write_file(paths[label], tf.io.encode_jpeg(tf.ones((20, 20, 3), tf.uint8) * intensity))
            frame["runtime_image_path"] = [paths[label] for label in frame.target]
            ds = training.build_dataset(frame, encoder, 20, 16, True, 1, 42,
                positive_sampling_fraction=.45, sampling_unit="lesion")
            targets = []
            for inputs, labels, _ in ds.take(40):
                truth = labels["high_risk"].numpy().reshape(-1)
                targets.extend(truth)
                np.testing.assert_array_equal(np.argmax(labels["diagnosis"], axis=1), np.where(truth == 1, 4, 5))
                np.testing.assert_allclose(tf.reduce_mean(inputs["image"], axis=(1, 2, 3)),
                    np.where(truth == 1, 20 / 255, 220 / 255), atol=1e-5)
            self.assertLess(abs(np.mean(targets) - .45), .08)


if __name__ == "__main__":
    unittest.main()
