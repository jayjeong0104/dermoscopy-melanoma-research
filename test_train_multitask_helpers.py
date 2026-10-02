from __future__ import annotations

import unittest
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

import train_multitask as training
from evaluate_derm12345 import audit_exact_overlap
from finalize_multitask_run import (
    cluster_bootstrap_confidence_intervals,
    read_prediction_cache,
    validate_prediction_arrays,
    write_prediction_cache,
)
from build_siim_training_folds import assign_training_folds
from prepare_derm12345_training import map_diagnosis
from prepare_siim_isic2020 import map_diagnosis as map_siim_diagnosis


class BinaryTargetTests(unittest.TestCase):
    def test_fixed_training_budget_is_optional_and_positive(self) -> None:
        with patch("sys.argv", ["train_multitask.py"]):
            args = training.parse_args()
        self.assertIsNone(args.steps_per_epoch)
        self.assertIn("steps_per_epoch", training.RESUME_CONFIG_KEYS)
        with TemporaryDirectory() as temporary_directory:
            args.manifest = Path(temporary_directory) / "manifest.csv"
            args.manifest.touch()
            for invalid in (0, -1):
                args.steps_per_epoch = invalid
                with self.assertRaisesRegex(ValueError, "steps-per-epoch"):
                    training.validate_args(args)
            args.steps_per_epoch = 1000
            training.validate_args(args)

    def setUp(self) -> None:
        self.frame = pd.DataFrame(
            {
                "diagnosis": ["AK", "BCC", "MEL", "NV", "SCC"],
                "target": [1, 1, 1, 0, 1],
            }
        )

    def test_malignant_excludes_premalignant_ak(self) -> None:
        configured = training.configure_binary_target(self.frame, "malignant")
        self.assertEqual(configured["target"].tolist(), [0, 1, 1, 0, 1])

    def test_melanoma_is_one_vs_rest(self) -> None:
        configured = training.configure_binary_target(self.frame, "melanoma")
        self.assertEqual(configured["target"].tolist(), [0, 0, 1, 0, 0])

    def test_derm_taxonomy_mapping(self) -> None:
        cases = [
            ({"super_class": "melanocytic", "main_class_1": "melanoma", "malignancy": "malignant", "label": "lmm"}, "MEL"),
            ({"super_class": "nonmelanocytic", "main_class_1": "keratinocytic", "malignancy": "malignant", "label": "bcc"}, "BCC"),
            ({"super_class": "nonmelanocytic", "main_class_1": "keratinocytic", "malignancy": "malignant", "label": "scc"}, "SCC"),
            ({"super_class": "nonmelanocytic", "main_class_1": "keratinocytic", "malignancy": "indeterminate", "label": "ak"}, "AK"),
            ({"super_class": "melanocytic", "main_class_1": "dysplastic", "malignancy": "benign", "label": "jd"}, "NV"),
        ]
        for values, expected in cases:
            with self.subTest(values=values):
                self.assertEqual(map_diagnosis(pd.Series(values)), expected)

    def test_siim_unknown_benign_does_not_supervise_diagnosis_head(self) -> None:
        self.assertEqual(map_siim_diagnosis("melanoma"), ("MEL", 1))
        self.assertEqual(map_siim_diagnosis("nevus"), ("NV", 1))
        self.assertEqual(map_siim_diagnosis("unknown"), ("UNKNOWN", 0))

    def test_melanoma_target_accepts_confirmed_unknown_benign(self) -> None:
        frame = pd.DataFrame(
            {
                "diagnosis": ["MEL", "NV", "UNKNOWN"],
                "target": [1, 0, 0],
                "diagnosis_label_available": [1, 1, 0],
            }
        )
        configured = training.configure_binary_target(frame, "melanoma")
        self.assertEqual(configured["target"].tolist(), [1, 0, 0])
        with self.assertRaisesRegex(ValueError, "melanoma-vs-rest"):
            training.configure_binary_target(frame, "high_risk")

    def test_siim_patients_are_distributed_only_across_training_folds(self) -> None:
        groups = pd.DataFrame(
            {
                "split_group_id": [f"patient_{index}" for index in range(12)],
                "patient_melanoma": [0] * 9 + [1] * 3,
            }
        )
        assigned = assign_training_folds(groups, seed=42)
        self.assertEqual(set(assigned["fold"]), {2, 3, 4})
        self.assertEqual(set(assigned["split_group_id"]), set(groups["split_group_id"]))


class EvaluationHelperTests(unittest.TestCase):
    def test_validation_blend_can_select_diagnosis_head(self) -> None:
        truth = np.asarray([0, 0, 1, 1], dtype="int32")
        direct = np.asarray([0.9, 0.8, 0.2, 0.1], dtype="float32")
        diagnosis = np.asarray([0.1, 0.2, 0.8, 0.9], dtype="float32")
        selected, probabilities, _ = training.select_probability_blend(
            truth, direct, diagnosis
        )
        self.assertEqual(selected["direct_weight"], 0.0)
        np.testing.assert_allclose(probabilities, diagnosis)

    def test_diagnosis_probabilities_map_to_melanoma(self) -> None:
        probabilities = np.zeros((2, len(training.DIAGNOSES)), dtype="float32")
        probabilities[:, training.DIAGNOSIS_TO_INDEX["MEL"]] = [0.25, 0.75]
        np.testing.assert_allclose(
            training.diagnosis_target_probabilities(probabilities, "melanoma"),
            [0.25, 0.75],
        )

    def test_lesion_probabilities_average_in_logit_space(self) -> None:
        frame = pd.DataFrame(
            {
                "lesion_id": ["A", "A", "B"],
                "source": ["source", "source", "source"],
                "target": [1, 1, 0],
                "target_name": ["positive", "positive", "negative"],
                "diagnosis": ["MEL", "MEL", "NV"],
                "diagnosis_index": [4, 4, 5],
            }
        )
        risk = np.asarray([0.2, 0.8, 0.1], dtype="float32")
        diagnosis = np.zeros((3, len(training.DIAGNOSES)), dtype="float32")
        diagnosis[:, 4] = [0.7, 0.9, 0.0]
        diagnosis[:, 5] = [0.3, 0.1, 1.0]

        lesions, diagnosis_probabilities = training.aggregate_predictions_by_lesion(
            frame, risk, diagnosis
        )

        self.assertEqual(lesions["image_count"].tolist(), [2, 1])
        self.assertAlmostEqual(float(lesions.loc[0, "probability"]), 0.5, places=6)
        self.assertAlmostEqual(float(lesions.loc[1, "probability"]), 0.1, places=6)
        self.assertIsNotNone(diagnosis_probabilities)
        np.testing.assert_allclose(diagnosis_probabilities[0, [4, 5]], [0.8, 0.2])

    def test_split_integrity_rejects_exact_hash_leakage(self) -> None:
        def split(lesion_id: str, sha256: str) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "lesion_id": [lesion_id],
                    "split_group_id": [lesion_id],
                    "sha256": [sha256],
                }
            )

        splits = training.SplitFrames(
            train=split("train", "same"),
            val=split("val", "same"),
            test=split("test", "different"),
        )
        with self.assertRaisesRegex(RuntimeError, "Exact image leakage"):
            training.assert_split_integrity(splits)

    def test_predefined_folds_keep_multilesion_patient_together(self) -> None:
        frame = pd.DataFrame(
            {
                "split_group_id": ["patient_a", "patient_a", "patient_b", "patient_c"],
                "lesion_id": ["a1", "a2", "b1", "c1"],
                "sha256": ["a1", "a2", "b1", "c1"],
            }
        )
        assignments = pd.DataFrame(
            {
                "split_group_id": ["patient_a", "patient_b", "patient_c"],
                "fold": [0, 1, 2],
            }
        )
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "folds.csv"
            assignments.to_csv(path, index=False)
            splits = training.split_from_fold_assignments(frame, path, 0, 1)
        self.assertEqual(set(splits.test["lesion_id"]), {"a1", "a2"})
        self.assertEqual(set(splits.val["lesion_id"]), {"b1"})
        self.assertEqual(set(splits.train["lesion_id"]), {"c1"})

    def test_wilson_interval_contains_observed_accuracy(self) -> None:
        interval = training.wilson_interval(95, 100)
        self.assertLess(interval["lower"], 0.95)
        self.assertGreater(interval["upper"], 0.95)

    def test_prediction_cache_is_invalidated_when_model_metadata_changes(self) -> None:
        direct = np.asarray([0.1, 0.9], dtype="float32")
        diagnosis = np.zeros((2, len(training.DIAGNOSES)), dtype="float32")
        diagnosis[:, training.DIAGNOSIS_TO_INDEX["MEL"]] = direct
        metadata = {
            "model_sha256": "first",
            "row_count": 2,
            "image_size": 96,
            "transforms": 1,
            "masked_metadata": True,
        }
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "predictions.npz"
            write_prediction_cache(path, direct, diagnosis, metadata)
            cached = read_prediction_cache(path, metadata, expected_rows=2)
            self.assertIsNotNone(cached)
            changed = {**metadata, "model_sha256": "second"}
            self.assertIsNone(read_prediction_cache(path, changed, expected_rows=2))

    def test_prediction_validation_rejects_non_finite_values(self) -> None:
        direct = np.asarray([0.1, np.nan], dtype="float32")
        diagnosis = np.zeros((2, len(training.DIAGNOSES)), dtype="float32")
        with self.assertRaisesRegex(ValueError, "NaN or infinite"):
            validate_prediction_arrays(direct, diagnosis, expected_rows=2)

    def test_cluster_bootstrap_reports_group_aware_intervals(self) -> None:
        truth = np.asarray([0, 0, 1, 1, 0, 1], dtype="int32")
        probability = np.asarray([0.1, 0.2, 0.8, 0.9, 0.3, 0.7], dtype="float32")
        clusters = np.asarray(["a", "a", "b", "b", "c", "d"])
        result = cluster_bootstrap_confidence_intervals(
            truth,
            probability,
            threshold=0.5,
            cluster_ids=clusters,
            iterations=200,
            seed=7,
        )
        self.assertEqual(result["cluster_count"], 4)
        self.assertGreaterEqual(result["valid_iterations"], 160)
        self.assertIn("roc_auc", result["intervals"])

    def test_external_exact_duplicates_are_counted_once_when_labels_match(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first = root / "first.jpg"
            second = root / "second.jpg"
            first.write_bytes(b"exact-copy")
            second.write_bytes(b"exact-copy")
            manifest = root / "training.csv"
            pd.DataFrame({"sha256": ["not-the-external-hash"]}).to_csv(
                manifest, index=False
            )
            frame = pd.DataFrame(
                {
                    "image_path": [first, second],
                    "patient_id": ["patient", "patient"],
                    "target": [1, 1],
                    "main_class_1": ["melanoma", "melanoma"],
                    "malignancy": ["malignant", "malignant"],
                    "label": ["mel", "mel"],
                }
            )
            unique, duplicates = audit_exact_overlap(frame, manifest, workers=1)
        self.assertEqual(len(unique), 1)
        self.assertEqual(len(duplicates), 2)
        self.assertEqual(int(duplicates["kept_for_evaluation"].sum()), 1)

    def test_external_patient_overlap_preserves_leading_zeros(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            manifest = Path(temporary_directory) / "training.csv"
            pd.DataFrame({"sha256": ["a" * 64], "patient_id": ["00123"]}).to_csv(manifest, index=False)
            frame = pd.DataFrame({"patient_id": ["00123"], "image_path": ["unused.jpg"]})
            with self.assertRaisesRegex(RuntimeError, "shares 1 patients"):
                audit_exact_overlap(frame, manifest, workers=1)

    def test_model_builds_with_current_augmentations(self) -> None:
        model = training.build_model(
            image_size=96,
            metadata_size=12,
            dropout_rate=0.35,
            metadata_modality_dropout=0.5,
            risk_features="image_only",
            backbone_name="efficientnetv2b0",
            weights=None,
        )
        self.assertEqual(set(model.output_names), {"diagnosis", "high_risk"})

    def test_resize_before_normalization_matches_previous_order(self) -> None:
        pixels = np.arange(17 * 13 * 3, dtype="uint8").reshape(17, 13, 3)
        image = training.tf.convert_to_tensor(pixels)
        previous = training.tf.image.resize(
            training.tf.image.convert_image_dtype(image, training.tf.float32),
            [11, 11],
            antialias=True,
        )
        optimized = training.resize_and_normalize_image(image, 11)
        np.testing.assert_allclose(optimized.numpy(), previous.numpy(), atol=1e-6)

    def test_tta_preserves_sample_weights(self) -> None:
        image = training.tf.reshape(
            training.tf.range(4, dtype=training.tf.float32), [1, 2, 2, 1]
        )
        inputs = {
            "image": image,
            "metadata": training.tf.zeros([1, 2], dtype=training.tf.float32),
        }
        labels = {"high_risk": training.tf.constant([[1.0]])}
        sample_weights = {"high_risk": training.tf.constant([1.0])}
        dataset = training.tf.data.Dataset.from_tensors(
            (inputs, labels, sample_weights)
        )

        transformed = training.transform_tta_dataset(dataset, transform_index=1)
        transformed_inputs, transformed_labels, transformed_weights = next(
            iter(transformed)
        )

        np.testing.assert_allclose(
            transformed_inputs["image"].numpy(),
            training.tf.image.flip_left_right(image).numpy(),
        )
        np.testing.assert_allclose(
            transformed_labels["high_risk"].numpy(), labels["high_risk"].numpy()
        )
        np.testing.assert_allclose(
            transformed_weights["high_risk"].numpy(),
            sample_weights["high_risk"].numpy(),
        )

    def test_stage_resume_epoch_uses_history_and_backup(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            run_dir = Path(temporary_directory)
            pd.DataFrame({"epoch": [0, 1], "val_high_risk_pr_auc": [0.4, 0.5]}).to_csv(
                run_dir / "training_head.csv", index=False
            )
            backup_dir = run_dir / "backup_head"
            backup_dir.mkdir()
            (backup_dir / "training_metadata.json").write_text(
                '{"epoch": 2, "batch": 0}', encoding="utf-8"
            )
            self.assertEqual(training.stage_resume_epoch(run_dir, "head"), 2)


if __name__ == "__main__":
    unittest.main()
