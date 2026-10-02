import unittest

import numpy as np
import pandas as pd

from analyze_recovery_errors import error_breakdown, paired_bootstrap


class ErrorAnalysisTests(unittest.TestCase):
    def test_single_class_group_has_null_undefined_rate(self):
        frame = pd.DataFrame({"diagnosis": ["NV", "NV", "MEL", "MEL"],
                              "target": [0, 0, 1, 1], "probability": [.1, .9, .2, .8]})
        rows = {row["group"]: row for row in error_breakdown(frame, .5, "diagnosis")}
        self.assertIsNone(rows["NV"]["sensitivity"])
        self.assertIsNone(rows["MEL"]["specificity"])
        self.assertEqual(rows["NV"]["false_positives"], 1)
        self.assertEqual(rows["MEL"]["false_negatives"], 1)

    def test_paired_identical_models_have_zero_delta(self):
        truth = np.array([0, 1] * 4)
        scores = np.array([.2, .8] * 4)
        groups = np.repeat(["p1", "p2", "p3", "p4"], 2)
        result = paired_bootstrap(truth, scores, scores, groups, .5, .5, iterations=100)
        self.assertEqual(result["cluster_count"], 4)
        for interval in result["delta_intervals_95"].values():
            self.assertEqual(interval, {"lower": 0.0, "upper": 0.0})

    def test_rejects_missing_groups_and_misalignment(self):
        truth = np.array([0, 1, 0, 1])
        scores = np.array([.1, .9, .2, .8])
        with self.assertRaisesRegex(ValueError, "aligned"):
            paired_bootstrap(truth, scores, scores, np.array(["p1"]), .5, .5)
        with self.assertRaisesRegex(ValueError, "Missing"):
            paired_bootstrap(truth, scores, scores, np.array(["p1", None, "p3", "p4"]), .5, .5)


if __name__ == "__main__":
    unittest.main()
