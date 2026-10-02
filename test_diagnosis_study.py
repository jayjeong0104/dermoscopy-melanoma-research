import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import diagnosis_probe as probe
import run_diagnosis_study as study
from converged_probe import save_parameters
from operating_policy import crossfit_policy
from queue_diagnosis_study import dependency_state
from run_diagnosis_study import fit_cached, inner_folds, load_split, masks, selection_key


class DiagnosisStudyTests(unittest.TestCase):
    def data(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=(120, 5)).astype("float32")
        classes = ("BCC", "MEL", "NV")
        labels = np.asarray(classes)[np.argmax(x[:, :3], axis=1)]
        return x, labels, classes

    def test_multiclass_gradient_matches_finite_differences(self):
        x, labels, classes = self.data()
        y = np.array([classes.index(value) for value in labels])
        theta = np.linspace(-.2, .3, (x.shape[1] + 1) * len(classes))
        _, actual = probe.objective_gradient(theta, x, y, .01, len(classes))
        expected = []
        for index in range(len(theta)):
            step = np.zeros_like(theta)
            step[index] = 1e-6
            expected.append((probe.objective_gradient(theta + step, x, y, .01, len(classes))[0] -
                probe.objective_gradient(theta - step, x, y, .01, len(classes))[0]) / 2e-6)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)

    def test_extreme_logits_finite_and_intercept_unpenalized(self):
        x = np.array([[1000., -1000.], [-1000., 1000.]])
        theta = np.array([1., -1., -1., 1., 0., 0.])
        loss, gradient = probe.objective_gradient(theta, x, np.array([1, 0]), .01, 2)
        self.assertTrue(np.isfinite(loss) and np.isfinite(gradient).all())
        theta[:4] = 0
        theta[-2:] = [2, -2]
        self.assertEqual(probe.objective_gradient(theta, x, np.array([1, 0]), .01, 2)[0],
                         probe.objective_gradient(theta, x, np.array([1, 0]), 10., 2)[0])

    def test_converged_starts_and_training_only_normalization(self):
        x, labels, classes = self.data()
        first, report = probe.fit(x, labels, .01, classes=classes)
        second, _ = probe.fit(x, labels, .01, seed=137, classes=classes)
        self.assertLess(report["gradient_max_abs"], 1e-6)
        np.testing.assert_allclose(probe.probabilities(first, x), probe.probabilities(second, x), atol=1e-5)
        np.testing.assert_allclose(first["mean"], x.mean(axis=0, dtype=np.float64), atol=1e-6)
        before = first["mean"].copy()
        np.testing.assert_allclose(probe.probabilities(first, x + 10).sum(axis=1), 1)
        np.testing.assert_array_equal(first["mean"], before)

    def test_bad_inputs_and_nonconvergence_fail_closed(self):
        x, labels, classes = self.data()
        for values, l2 in ((labels[:-1], .01), (np.full(len(labels), "NV"), .01), (labels, 0)):
            with self.assertRaises(ValueError):
                probe.fit(x, values, l2, classes=classes)
        with self.assertRaises(RuntimeError):
            probe.fit(x, labels, .01, maxiter=1, classes=classes)

    def test_npz_and_real_tensorflow_export_roundtrip(self):
        x, labels, classes = self.data()
        model, _ = probe.fit(x, labels, .01, classes=classes)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.npz"
            save_parameters(path, model)
            np.testing.assert_array_equal(probe.probabilities(model, x), probe.probabilities(probe.load_parameters(path), x))
            report = probe.export(path, Path(directory) / "model.keras", x)
            self.assertLess(report["maximum_probability_difference"], 1e-8)
            self.assertEqual(report["classes"], list(classes))
            self.assertFalse(report["automatic_promotion"])

    def test_training_calibration_and_scoring_groups_are_disjoint(self):
        frame = pd.DataFrame({"source": ["A"] * 200, "lesion_id": [str(i) for i in range(200)],
            "target": [(i // 2) % 2 for i in range(200)], "split_group_id": [str(i // 2) for i in range(200)]})
        folds = inner_folds(frame)
        np.testing.assert_array_equal(folds, inner_folds(frame))
        scored = np.zeros(len(frame), dtype=int)
        for fold in range(5):
            training, calibration, scoring = masks(folds, fold)
            groups = [set(frame.loc[mask, "split_group_id"]) for mask in (training, calibration, scoring)]
            self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
            self.assertTrue(np.all(training.astype(int) + calibration + scoring == 1))
            scored += scoring
        np.testing.assert_array_equal(scored, np.ones(len(frame)))

    def test_selection_rejects_good_average_hiding_weak_source(self):
        weak = {"pr_auc": .9, "by_source": {"A": {"recall": 1., "specificity": .9}, "B": {"recall": .85, "specificity": .9}}}
        stable = {"pr_auc": .8, "by_source": {"A": {"recall": .96, "specificity": .7}, "B": {"recall": .96, "specificity": .7}}}
        self.assertGreater(selection_key(stable), selection_key(weak))

    def test_fit_resume_checks_training_labels_not_only_saved_weights(self):
        x, labels, _ = self.data()
        y = (labels == "MEL").astype(int)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            fit_cached(output, "head", "binary", x, y, .01, 0)
            with patch("run_diagnosis_study.binary.fit") as fit:
                fit_cached(output, "head", "binary", x, y, .01, 0)
                fit.assert_not_called()
            with self.assertRaisesRegex(ValueError, "changed"):
                fit_cached(output, "head", "binary", x, 1 - y, .01, 0)

    def test_test_split_cannot_be_loaded(self):
        with self.assertRaisesRegex(ValueError, "disabled"):
            load_split("test")

    def test_dependency_failed_or_tampered_report_cannot_start_training(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "status.json").write_text(json.dumps({"status": "running", "pid": 2, "driver_pid": 1}))
            self.assertEqual(dependency_state(path), ("waiting", 1))
            (path / "status.json").write_text(json.dumps({"status": "failed"}))
            with self.assertRaises(RuntimeError):
                dependency_state(path)
            (path / "status.json").write_text(json.dumps({"status": "complete"}))
            (path / "validation_report.json").write_bytes(b"{}")
            (path / "complete.json").write_text(json.dumps({"report_sha256": hashlib.sha256(b"{}").hexdigest()}))
            self.assertEqual(dependency_state(path), ("ready", None))
            (path / "validation_report.json").write_bytes(b"changed")
            with self.assertRaises(ValueError):
                dependency_state(path)

    def test_end_to_end_synthetic_study_and_completed_resume(self):
        rng = np.random.default_rng(92)

        def cohort(count, prefix):
            labels = np.resize(np.asarray(probe.CLASSES), count)
            frame = pd.DataFrame({"source": ["A"] * count, "diagnosis": labels,
                "target": (labels == "MEL").astype(int),
                "lesion_id": [f"{prefix}{i}" for i in range(count)],
                "split_group_id": [f"{prefix}{i}" for i in range(count)],
                "image_id": [f"{prefix}{i}" for i in range(count)], "image_path": "never_opened.jpg"})
            x = rng.normal(size=(count, 6)).astype("float32")
            x[:, 0] += frame.target.to_numpy() * 2
            return frame, x

        train, val = cohort(320, "train"), cohort(160, "val")
        _, reference = crossfit_policy(val[0], .2 + .6 * val[0].target.to_numpy())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dependency, output = root / "context", root / "study"
            (dependency / "features/panderm_full_frame").mkdir(parents=True)
            output.mkdir()
            (dependency / "validation_report.json").write_bytes(b"{}")
            (dependency / "complete.json").write_text(json.dumps({"report_sha256": hashlib.sha256(b"{}").hexdigest()}))
            (dependency / "features/panderm_full_frame/complete.json").write_text("{}")
            reference.to_csv(root / "reference.csv", index=False)
            plan = {"calibration_target": .96, "final_starts": [0, 137], "start_probability_tolerance": 1e-4}
            with patch.object(study, "CONTEXT", dependency), patch.object(study, "REFERENCE", root / "reference.csv"), \
                    patch.object(study, "GRID", (.01,)), patch.object(study, "load_split", side_effect=lambda split: train if split == "train" else val):
                study.run(output, plan)
                result = json.loads((output / "validation_report.json").read_text())
                self.assertEqual(len(result["results"]), 4)
                self.assertFalse(result["automatic_promotion"])
                self.assertFalse(result["test_inference"])
                with patch.object(study, "fit_cached") as fit:
                    study.run(output, plan)
                    fit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
