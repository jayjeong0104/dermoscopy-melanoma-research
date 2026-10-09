import json
import unittest

import numpy as np
import pandas as pd

from development_evaluation import (confusion_metrics, evaluate_development,
    fit_recall_threshold, grouped_folds, validate_folds)


class DevelopmentEvaluationTests(unittest.TestCase):
    def data(self):
        rng = np.random.default_rng(301)
        n = 200
        target = np.tile([0, 1], n // 2)
        return pd.DataFrame({"source": np.resize(["A", "A", "B", "B"], n),
            "lesion_id": [f"l{i:04d}" for i in range(n)],
            "split_group_id": [f"p{i // 2:04d}" for i in range(n)], "target": target,
            "diagnosis": np.where(target, "MEL", "NV"),
            "direct_probability": np.clip(.15 + .55 * target + rng.normal(0, .15, n), .001, .999),
            "diagnosis_probability": np.clip(.2 + .4 * target + rng.normal(0, .15, n), .001, .999)})

    def test_folds_stable_to_input_order_and_patient_grouped(self):
        data = self.data()
        folds = grouped_folds(data)
        shuffled = data.sample(frac=1., random_state=15).reset_index(drop=True)
        mapped = dict(zip(shuffled.lesion_id, grouped_folds(shuffled)))
        np.testing.assert_array_equal(folds, data.lesion_id.map(mapped))
        self.assertEqual(data.assign(fold=folds).groupby("split_group_id").fold.nunique().max(), 1)

    def test_scoring_labels_cannot_change_their_fitted_threshold_or_blend(self):
        data = self.data()
        folds = grouped_folds(data)
        before, _ = evaluate_development(data, folds=folds)
        altered = data.copy()
        altered.loc[folds == 0, "target"] = 1 - altered.loc[folds == 0, "target"]
        after, _ = evaluate_development(altered, folds=folds)
        for candidate in before["results"]:
            a, b = before["results"][candidate]["folds"][0], after["results"][candidate]["folds"][0]
            self.assertEqual(a["threshold"], b["threshold"])
            self.assertEqual(a["first_component_weight"], b["first_component_weight"])

    def test_exact_threshold_handles_scores_below_old_grid_and_ties(self):
        y = np.array([1, 1, 1, 0, 0])
        p = np.array([.0001, .0001, .0002, .0001, .00001])
        threshold = fit_recall_threshold(y, p, .95)
        self.assertEqual(threshold, .0001)
        metrics = confusion_metrics(y, p >= threshold)
        self.assertEqual(metrics["true_positives"], 3)
        self.assertEqual(metrics["false_positives"], 1)

    def test_single_class_diagnosis_slices_are_defined_without_fake_auc(self):
        report, decisions = evaluate_development(self.data())
        self.assertEqual(len(decisions), 3 * 200)
        for result in report["results"].values():
            self.assertIsNone(result["by_diagnosis"]["NV"]["recall"])
            self.assertIsNone(result["by_diagnosis"]["MEL"]["specificity"])
            self.assertIsNone(result["by_diagnosis"]["MEL"]["roc_auc"])
            self.assertIsNone(result["by_diagnosis"]["NV"]["pr_auc"])
            self.assertTrue(result["descriptive_tradeoffs"]["uses_same_cohort_labels_for_threshold_and_metrics"])
            self.assertFalse(result["descriptive_tradeoffs"]["for_model_selection_or_deployment"])
        json.dumps(report, allow_nan=False)
        self.assertFalse(report["automatic_promotion"])
        self.assertFalse(report["test_or_hiba_access"])
        self.assertTrue(report["checkpoint_selection_already_used_development"])

    def test_invalid_inputs_and_crossing_groups_fail_closed(self):
        data = self.data()
        for altered in (pd.concat([data, data.iloc[:1]], ignore_index=True),
                data.assign(direct_probability=np.nan), data.assign(diagnosis_probability=1.1)):
            with self.assertRaises(ValueError):
                evaluate_development(altered)
        folds = grouped_folds(data)
        folds[0] = (folds[1] + 1) % 5
        with self.assertRaisesRegex(ValueError, "crosses"):
            validate_folds(data, folds)

    def test_fixed_candidate_scoring_and_no_blend(self):
        data = self.data()
        report, predictions = evaluate_development(data, score_columns=("direct_probability",), blend_columns=None)
        self.assertEqual(list(report["results"]), ["direct_probability"])
        self.assertEqual(len(predictions), len(data))
        np.testing.assert_array_equal(predictions.probability, data.direct_probability)
        expected = confusion_metrics(data.target.to_numpy(), predictions.prediction.to_numpy())
        self.assertEqual(report["results"]["direct_probability"]["metrics"]["confusion_matrix"], expected["confusion_matrix"])


if __name__ == "__main__":
    unittest.main()
