"""Regression tests for exact threshold decisions on finite validation scores."""
import unittest

import numpy as np

import train_multitask as training


class EmpiricalThresholdTests(unittest.TestCase):
    def test_resolves_decisions_between_legacy_grid_points(self):
        truth = np.array([0, 0, 1, 1])
        scores = np.array([.1001, .1002, .1003, .1004])
        balanced, operating, _ = training.select_thresholds(truth, scores, 1.)
        self.assertEqual(operating["threshold"], .1003)
        self.assertEqual(operating["accuracy"], 1.)
        self.assertEqual(balanced["balanced_accuracy"], 1.)
        _, legacy, _ = training.select_thresholds(truth, scores, 1., method="legacy_grid")
        self.assertEqual(legacy["accuracy"], .5)

    def test_guarantees_feasible_recall_for_near_zero_and_zero_scores(self):
        for scores in (np.array([.0001, .0002, .0003, .0004]), np.zeros(4)):
            with self.subTest(scores=scores):
                _, operating, _ = training.select_thresholds(np.array([0, 0, 1, 1]), scores, 1.)
                self.assertEqual(operating["recall"], 1.)

    def test_confusion_counts_match_predictions_including_ties_and_boundaries(self):
        rng = np.random.default_rng(19)
        truth = rng.integers(0, 2, 100)
        scores = rng.choice(np.array([0., .1001, .1002, .5005, .9999, 1.]), 100)
        _, _, rows = training.select_thresholds(truth, scores, .95)
        self.assertEqual(len(rows), len(np.unique(scores)))
        for row in rows:
            expected = training.threshold_metrics(truth, scores, row["threshold"])
            self.assertEqual(row["confusion_matrix"], expected["confusion_matrix"])
            for name in ("accuracy", "balanced_accuracy", "precision", "recall", "specificity", "f1",
                         "matthews_correlation_coefficient"):
                self.assertAlmostEqual(row[name], expected[name])

    def test_both_objectives_match_exhaustive_empirical_search(self):
        truth = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        scores = np.array([.02, .03, .03, .05, .05, .051, .052, 1.])
        balanced, operating, _ = training.select_thresholds(truth, scores, .75)
        rows = [training.threshold_metrics(truth, scores, value) for value in np.unique(scores)]
        expected_balanced = max(rows, key=lambda row: (row["balanced_accuracy"], row["f1"], row["recall"]))
        expected_operating = max([row for row in rows if row["recall"] >= .75],
            key=lambda row: (row["accuracy"], row["precision"], row["specificity"]))
        self.assertEqual(balanced["threshold"], expected_balanced["threshold"])
        self.assertEqual(operating["threshold"], expected_operating["threshold"])

    def test_rejects_invalid_scores_labels_and_recall(self):
        for truth, scores, recall in (([0, 1], [0., np.nan], .95), ([0, 1], [0., 1.1], .95),
                ([0, 0], [0., .2], .95), ([0, 1], [0.], .95), ([0, 1], [0., .2], 0.)):
            with self.subTest(truth=truth, scores=scores, recall=recall), self.assertRaises(ValueError):
                training.select_thresholds(np.asarray(truth), np.asarray(scores), recall)


if __name__ == "__main__":
    unittest.main()
