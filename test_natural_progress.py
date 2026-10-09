"""Guard paired comparisons against changed case identity and supervision."""
import unittest

import pandas as pd

from review_natural_progress import align_reference


class AlignmentTests(unittest.TestCase):
    def setUp(self):
        self.rows = pd.DataFrame({"source": ["S", "S"], "lesion_id": ["a", "b"],
            "split_group_id": ["a", "b"], "target": [1, 0], "diagnosis": ["MEL", "NV"],
            "image_count": [2, 1], "direct_probability": [.8, .2]})

    def test_shuffled_predictions_are_matched_by_identity(self):
        result = align_reference(self.rows.iloc[::-1], self.rows)
        pd.testing.assert_frame_equal(result, self.rows)

    def test_changed_labels_or_photo_counts_are_rejected(self):
        for name, value in (("target", 0), ("split_group_id", "wrong"), ("image_count", 3)):
            changed = self.rows.copy()
            changed.loc[0, name] = value
            with self.subTest(name=name), self.assertRaises(ValueError):
                align_reference(self.rows, changed)


if __name__ == "__main__":
    unittest.main()
