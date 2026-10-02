import unittest

import numpy as np
import pandas as pd

from audit_error_reduction import compare, validate_predictions


class ErrorReductionTests(unittest.TestCase):
    def fixture(self):
        # All eight combinations of truth, previous call, and new call.
        truth = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        old = np.array([0, 0, 1, 1, 0, 0, 1, 1])
        new = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        frame = pd.DataFrame({"source": ["A"] * 8, "lesion_id": [str(i) for i in range(8)],
            "split_group_id": [str(i) for i in range(8)], "target": truth,
            "policy_fold": [0, 1, 0, 1, 0, 1, 0, 1], "threshold": .5})
        baseline = frame.assign(prediction=old, probability=.2 + .6 * old)
        candidate = frame.assign(prediction=new, probability=.2 + .6 * new)
        metadata = frame.assign(diagnosis=np.where(truth, "MEL", "NV"),
            image_id=frame.lesion_id, image_path="not_opened.jpg")
        return baseline, candidate, metadata

    def test_all_transitions_counted_and_net_changes_consistent(self):
        report, _ = compare(*self.fixture())
        for key, value in report["transitions"].items():
            self.assertEqual(value, 2 if key == "correct_in_both" else 1)
        for key in ("net_fewer_errors", "net_fewer_false_alarms", "net_fewer_missed_melanomas"):
            self.assertEqual(report[key], 0)

    def test_shuffled_rows_align_by_identity(self):
        old, new, metadata = self.fixture()
        expected, _ = compare(old, new, metadata)
        actual, _ = compare(old, new.sample(frac=1, random_state=7), metadata)
        self.assertEqual(expected, actual)

    def test_accuracy_gain_cannot_hide_new_missed_melanoma(self):
        old, new, metadata = self.fixture()
        old["prediction"], old["probability"] = 1, .8
        new["prediction"] = new.target
        new.loc[4, "prediction"] = 0
        new["probability"] = .2 + .6 * new.prediction
        report, _ = compare(old, new, metadata)
        checks = report["diagnostic_checks_not_a_replacement_promotion_gate"]
        self.assertTrue(checks["fewer_total_errors"])
        self.assertTrue(checks["higher_precision"])
        self.assertFalse(checks["no_net_increase_in_missed_melanomas"])

    def test_single_class_subgroups_use_null_not_invented_rates(self):
        report, _ = compare(*self.fixture())
        groups = report["subgroups"]["diagnosis"]
        self.assertIsNone(groups["NV"]["candidate"]["sensitivity"])
        self.assertIsNone(groups["MEL"]["candidate"]["specificity"])

    def test_equal_overall_recall_cannot_hide_source_regression(self):
        old, new, metadata = self.fixture()
        for frame in (old, new, metadata):
            frame["source"] = np.tile(["B", "A"], 4)
        report, _ = compare(old, new, metadata)
        checks = report["diagnostic_checks_not_a_replacement_promotion_gate"]
        self.assertTrue(checks["no_net_increase_in_missed_melanomas"])
        self.assertFalse(checks["no_source_has_more_missed_melanomas"])
        warning = next(row for row in report["source_regression_warnings"] if row["source"] == "B")
        self.assertEqual(warning["additional_missed_melanomas"], 1)

    def test_different_patient_label_fold_or_missing_row_rejected(self):
        for column, value in (("split_group_id", "changed"), ("target", 1), ("policy_fold", 1)):
            old, new, metadata = self.fixture()
            new.loc[0, column] = value
            with self.assertRaises(ValueError):
                compare(old, new, metadata)
        old, new, metadata = self.fixture()
        with self.assertRaises(ValueError):
            compare(old, new.iloc[1:], metadata)

    def test_invalid_scores_decisions_or_duplicates_rejected(self):
        for column, value in (("prediction", 2), ("probability", float("nan")),
                              ("probability", 1.5), ("threshold", float("inf")), ("prediction", 1)):
            old, _, _ = self.fixture()
            old.loc[0, column] = value
            with self.assertRaises(ValueError):
                validate_predictions(old)
        old, _, _ = self.fixture()
        with self.assertRaises(ValueError):
            validate_predictions(pd.concat([old, old.iloc[:1]], ignore_index=True))

    def test_metadata_mismatch_rejected(self):
        old, new, metadata = self.fixture()
        metadata.loc[0, "diagnosis"] = "MEL"
        with self.assertRaisesRegex(ValueError, "Diagnosis"):
            compare(old, new, metadata)
        with self.assertRaises(ValueError):
            compare(old, new, metadata.iloc[1:])


if __name__ == "__main__":
    unittest.main()
