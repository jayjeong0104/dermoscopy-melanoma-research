import unittest

import pandas as pd

from prepare_siim_isic2020 import melanoma_targets


class MelanomaTargetTests(unittest.TestCase):
    def test_unknown_other_malignancy_does_not_become_melanoma(self):
        frame = pd.DataFrame({
            "diagnosis": ["UNKNOWN", "MEL", "BCC"],
            "diagnosis_label_available": [0, 1, 1],
            "target": [1, 1, 1], "melanoma_label": [0, 1, 0],
        })
        self.assertEqual(melanoma_targets(frame).tolist(), [0, 1, 0])

    def test_mixed_derm_and_legacy_siim_rows(self):
        frame = pd.DataFrame({
            "diagnosis": ["UNKNOWN", "UNKNOWN", "MEL"],
            "diagnosis_label_available": [0, 0, 1],
            "target": [1, 0, 1], "melanoma_label": [0, None, None],
        }, index=[7, 11, 42])
        result = melanoma_targets(frame)
        self.assertEqual(result.tolist(), [0, 0, 1])
        self.assertEqual(result.index.tolist(), [7, 11, 42])

    def test_legacy_without_explicit_labels_is_supported(self):
        frame = pd.DataFrame({"diagnosis": ["UNKNOWN", "MEL", "BCC"],
                              "diagnosis_label_available": [0, 1, 1], "target": [0, 1, 1]})
        self.assertEqual(melanoma_targets(frame).tolist(), [0, 1, 0])

    def test_explicit_labels_must_be_binary_without_truncation(self):
        for invalid in [0.5, 2, -1, "unknown"]:
            with self.subTest(invalid=invalid):
                frame = pd.DataFrame({"diagnosis": ["UNKNOWN"], "diagnosis_label_available": [0],
                                      "target": [0], "melanoma_label": [invalid]})
                with self.assertRaises(ValueError):
                    melanoma_targets(frame)

    def test_supervised_explicit_label_must_match_diagnosis(self):
        frame = pd.DataFrame({"diagnosis": ["MEL"], "target": [1], "melanoma_label": [0]})
        with self.assertRaisesRegex(ValueError, "disagrees"):
            melanoma_targets(frame)

    def test_invalid_fallback_or_supervision_does_not_silently_coerce(self):
        for supervision, target in [(0, 0.5), (0.5, 0)]:
            with self.subTest(supervision=supervision, target=target):
                frame = pd.DataFrame({"diagnosis": ["UNKNOWN"],
                                      "diagnosis_label_available": [supervision], "target": [target]})
                with self.assertRaises(ValueError):
                    melanoma_targets(frame)


if __name__ == "__main__":
    unittest.main()
