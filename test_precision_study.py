import unittest

import numpy as np
import pandas as pd

from study_precision import paired_precision


class PairedPrecisionTests(unittest.TestCase):
    def frame(self):
        # Two lesions per patient so uncertainty resamples patients together.
        return pd.DataFrame({"source": ["A"] * 80,
            "lesion_id": [f"lesion{i}" for i in range(80)],
            "split_group_id": [f"patient{i // 2}" for i in range(80)],
            "target": np.tile([1, 0], 40), "policy_fold": np.repeat(np.arange(5), 16),
            "prediction": np.ones(80, dtype=int)})

    def test_identical_decisions_have_zero_delta(self):
        frame = self.frame()
        result = paired_precision(frame, frame.copy(), iterations=200)
        self.assertEqual(result["groups"], 40)
        self.assertEqual(result["delta_intervals_95"]["precision"], {"lower": 0., "upper": 0.})
        self.assertEqual(result["delta_intervals_95"]["sensitivity"], {"lower": 0., "upper": 0.})

    def test_removing_false_alarms_preserves_recall_and_increases_precision(self):
        frame = self.frame()
        improved = frame.assign(prediction=frame.target)
        result = paired_precision(frame, improved, iterations=200)
        self.assertEqual(result["delta_intervals_95"]["precision"], {"lower": .5, "upper": .5})
        self.assertEqual(result["delta_intervals_95"]["sensitivity"], {"lower": 0., "upper": 0.})

    def test_patient_or_label_mismatch_rejected(self):
        for key, value in (("split_group_id", "unrelated"), ("target", 0), ("policy_fold", 4)):
            frame = self.frame()
            changed = frame.copy()
            changed.loc[0, key] = value
            with self.assertRaisesRegex(ValueError, "identical"):
                paired_precision(frame, changed, iterations=200)


if __name__ == "__main__":
    unittest.main()
