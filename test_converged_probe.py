import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from converged_probe import fit, load_parameters, objective_gradient, probabilities, save_parameters
from run_convergence_study import local_path, train_cached, load_split


class ConvergedProbeTests(unittest.TestCase):
    def data(self):
        rng = np.random.default_rng(5)
        x = rng.normal(size=(200, 8)).astype("float32")
        y = (x[:, 0] + .3 * rng.normal(size=len(x)) > .3).astype(int)
        return x, y

    def test_analytic_gradient_matches_finite_differences(self):
        x, y = self.data()
        theta = np.linspace(-.2, .3, x.shape[1] + 1)
        _, actual = objective_gradient(theta, x, y, .02)
        expected = []
        for i in range(len(theta)):
            step = np.zeros_like(theta)
            step[i] = 1e-6
            expected.append((objective_gradient(theta + step, x, y, .02)[0] -
                             objective_gradient(theta - step, x, y, .02)[0]) / 2e-6)
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-8)

    def test_intercept_is_not_regularized(self):
        x, y = self.data()
        theta = np.zeros(x.shape[1] + 1)
        theta[-1] = 2
        first = objective_gradient(theta, x, y, .001)
        second = objective_gradient(theta, x, y, 10.)
        self.assertEqual(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])

    def test_extreme_logits_are_finite(self):
        x = np.array([[1000., -1000.], [-1000., 1000.]])
        value, gradient = objective_gradient(np.array([1., -1., 0.]), x, np.array([0, 1]), .01)
        self.assertTrue(np.isfinite(value))
        self.assertTrue(np.isfinite(gradient).all())

    def test_converged_starts_agree(self):
        x, y = self.data()
        first, one = fit(x, y, .01)
        second, two = fit(x, y, .01, seed=137)
        self.assertLess(one["gradient_max_abs"], 1e-6)
        self.assertLess(two["gradient_max_abs"], 1e-6)
        np.testing.assert_allclose(probabilities(first, x), probabilities(second, x), atol=1e-5)
        self.assertLess(abs(one["objective"] - two["objective"]), 1e-9)

    def test_insufficient_budget_fails_closed(self):
        x, y = self.data()
        with self.assertRaisesRegex(RuntimeError, "did not converge"):
            fit(x, y, .01, maxiter=1)

    def test_standardization_uses_training_only_and_handles_constant_columns(self):
        x, y = self.data()
        x[:, -1] = 7
        model, _ = fit(x[:160], y[:160], .01)
        np.testing.assert_allclose(model["mean"], x[:160].mean(axis=0, dtype=np.float64), atol=1e-6)
        self.assertGreater(model["scale"][-1], 0)
        heldout = x[160:] + 100
        before = model["mean"].copy()
        self.assertTrue(np.isfinite(probabilities(model, heldout)).all())
        np.testing.assert_array_equal(before, model["mean"])

    def test_invalid_training_inputs_rejected(self):
        x, y = self.data()
        for bad_x, bad_y, l2 in ((x, y, 0), (x, y, -1), (x, y[:-1], .01),
                                 (x, np.zeros_like(y), .01), (x * np.nan, y, .01)):
            with self.assertRaises(ValueError):
                fit(bad_x, bad_y, l2)

    def test_parameter_roundtrip(self):
        x, y = self.data()
        model, _ = fit(x, y, .01)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.npz"
            save_parameters(path, model)
            loaded = load_parameters(path)
            np.testing.assert_array_equal(probabilities(model, x), probabilities(loaded, x))

    def test_completed_fit_is_reused_and_hash_tampering_fails(self):
        x, y = self.data()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_cached(root, "head", x, y, .01, 0)
            with patch("run_convergence_study.fit") as optimizer:
                train_cached(root, "head", x, y, .01, 0)
                optimizer.assert_not_called()
            (root / "head.npz").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "changed"):
                train_cached(root, "head", x, y, .01, 0)

    def test_feature_loading_rejects_test_split(self):
        with self.assertRaisesRegex(ValueError, "Test feature extraction"):
            load_split(Path("not_read"), "test")

    def test_wsl_provenance_path_is_portable(self):
        expected = "c:/data/model.keras" if os.name == "nt" else "/mnt/c/data/model.keras"
        self.assertEqual(str(local_path("/mnt/c/data/model.keras")).replace("\\", "/").casefold(), expected)


if __name__ == "__main__":
    unittest.main()
