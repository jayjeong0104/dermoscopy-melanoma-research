import unittest

import numpy as np
import pandas as pd

from evaluate_hiba import MELANOMA, id_overlap, measure, select_dermoscopy


class HibaEvaluationTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame({
            "isic_id": ["ISIC_0000001", "ISIC_0000002", "ISIC_0000003"],
            "lesion_id": ["IL_1", "IL_2", "IL_3"], "patient_id": ["IP_1", "IP_2", "IP_3"],
            "image_type": ["dermoscopic", "dermoscopic", "clinical: overview"],
            "copyright_license": ["CC-BY"] * 3,
            "diagnosis_2": [MELANOMA, "Malignant epidermal proliferations", MELANOMA]})

    def test_target_is_melanoma_not_all_cancer(self):
        selected = select_dermoscopy(self.frame())
        self.assertEqual(selected.target.tolist(), [1, 0])

    def test_conflicting_lesion_fails(self):
        frame = self.frame()
        frame.loc[1, "lesion_id"] = "IL_1"
        with self.assertRaises(ValueError):
            select_dermoscopy(frame)

    def test_identity_overlap_including_namespaces(self):
        known = pd.DataFrame({"image_id": ["other"], "patient_id": ["source::IP_2"],
                              "lesion_id": ["other"], "source_lesion_id": ["other"]})
        self.assertEqual(id_overlap(self.frame(), known).tolist(), [False, True, False])

    def test_fixed_threshold_confusion_counts(self):
        result = measure([1, 1, 0, 0], np.array([.8, .05, .2, .01]), .1)
        self.assertEqual([result[k] for k in ["tp", "fn", "fp", "tn"]], [1, 1, 1, 1])
        self.assertEqual(result["accuracy"], .5)
        self.assertEqual(result["roc_auc"], .75)


if __name__ == "__main__":
    unittest.main()
