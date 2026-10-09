import unittest

import pandas as pd

from prepare_derm12345_training import correct_derm_taxonomy, repair_existing_manifest


def record(label, main, malignancy="malignant"):
    return {
        "source": "DERM12345", "image_id": label, "lesion_id": "DERM::" + label,
        "split_group_id": "patient::" + label, "sha256": "hash::" + label,
        "image_path": "unchanged/" + label, "derm_label": label,
        "original_malignancy": malignancy, "main_class_1": main,
        "super_class": "melanocytic" if main in {"melanoma", "banal"} else "nonmelanocytic",
        "diagnosis": "VASC", "risk_group": "benign", "target": 0,
    }


class DermTaxonomyTests(unittest.TestCase):
    def test_unsupported_malignancies_keep_binary_but_mask_auxiliary_supervision(self):
        original = pd.DataFrame([
            record("ks", "vascular"), record("dfsp", "fibro_histiocytic"),
            record("ch", "keratinocytic"), record("mpd", "keratinocytic"),
        ])
        corrected = correct_derm_taxonomy(original)
        self.assertTrue(corrected.diagnosis.eq("UNKNOWN").all())
        self.assertTrue(corrected.diagnosis_label_available.eq(0).all())
        self.assertTrue(corrected.melanoma_label.eq(0).all())
        self.assertTrue(corrected.high_risk_label.eq(1).all())
        self.assertTrue(corrected.risk_group.eq("malignant").all())
        self.assertTrue(original.diagnosis.eq("VASC").all())

    def test_supported_diagnoses_retain_supervision(self):
        data = pd.DataFrame([
            record("mel", "melanoma"), record("bcc", "keratinocytic"),
            record("scc", "keratinocytic"), record("bd", "keratinocytic"),
            record("ha", "vascular", "benign"), record("df", "fibro_histiocytic", "benign"),
        ])
        fixed = correct_derm_taxonomy(data)
        self.assertEqual(fixed.diagnosis.tolist(), ["MEL", "BCC", "SCC", "SCC", "VASC", "DF"])
        self.assertTrue(fixed.diagnosis_label_available.eq(1).all())
        self.assertEqual(fixed.melanoma_label.tolist(), [1, 0, 0, 0, 0, 0])

    def test_repair_preserves_identities_and_distinguishes_melanoma_from_other_cancer(self):
        original = pd.DataFrame([record("ks", "vascular"), record("mel", "melanoma")])
        fixed = repair_existing_manifest(original)
        self.assertEqual(fixed.target.tolist(), [0, 1])
        self.assertEqual(fixed.high_risk_label.tolist(), [1, 1])
        self.assertEqual(fixed.risk_group.tolist(), ["malignant", "malignant"])
        for key in ("source", "image_id", "image_path", "sha256", "lesion_id", "split_group_id", "original_malignancy", "derm_label"):
            pd.testing.assert_series_equal(fixed[key], original[key])

    def test_existing_siim_binary_only_rows_keep_nonmelanoma_label(self):
        data = pd.DataFrame([{
            "source": "SIIM_ISIC_2020", "diagnosis": "UNKNOWN",
            "diagnosis_label_available": 0, "target": 0,
        }])
        fixed = repair_existing_manifest(data)
        self.assertEqual(fixed.target.tolist(), [0])
        self.assertTrue(fixed.high_risk_label.isna().all())

    def test_fresh_high_risk_derm_manifest_roundtrip_preserves_other_cancers(self):
        raw = pd.DataFrame([record("ks", "vascular"), record("ch", "keratinocytic"),
                            record("mel", "melanoma")])
        prepared = correct_derm_taxonomy(raw)
        self.assertEqual(prepared.target.tolist(), [1, 1, 1])
        fixed = repair_existing_manifest(prepared)
        self.assertEqual(fixed.target.tolist(), [0, 0, 1])
        self.assertEqual(fixed.diagnosis_label_available.tolist(), [0, 0, 1])
        self.assertEqual(fixed.high_risk_label.tolist(), [1, 1, 1])
        pd.testing.assert_frame_equal(repair_existing_manifest(fixed), fixed)

    def test_binary_only_rows_honor_valid_explicit_label_over_broad_target(self):
        raw = pd.DataFrame({"source": ["OTHER"], "diagnosis": ["UNKNOWN"],
                            "diagnosis_label_available": [0], "target": [1], "melanoma_label": [0]})
        self.assertEqual(repair_existing_manifest(raw).target.tolist(), [0])
        for invalid in (.5, 2, "bad"):
            with self.assertRaises(ValueError):
                repair_existing_manifest(raw.assign(melanoma_label=invalid))
        with self.assertRaisesRegex(ValueError, "Cannot infer"):
            repair_existing_manifest(raw.drop(columns="melanoma_label"))

    def test_missing_raw_taxonomy_fails_instead_of_guessing(self):
        data = pd.DataFrame([record("ks", "vascular")]).drop(columns="original_malignancy")
        with self.assertRaisesRegex(ValueError, "missing columns"):
            repair_existing_manifest(data)

    def test_inconsistent_melanoma_malignancy_fails(self):
        data = pd.DataFrame([record("mel", "melanoma", "benign")])
        with self.assertRaisesRegex(ValueError, "not marked malignant"):
            correct_derm_taxonomy(data)


if __name__ == "__main__":
    unittest.main()
