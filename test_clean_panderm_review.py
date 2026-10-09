import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import run_clean_panderm_review as study


class CleanPanDermReviewTests(unittest.TestCase):
    def test_excludes_test_and_incumbent_features(self):
        with patch.object(study, "load_features") as loader:
            with self.assertRaisesRegex(ValueError, "Only train"):
                study.only_panderm("test")
            loader.assert_not_called()
        with patch.object(study, "load_features", return_value=(None, np.zeros((2, 2048)))) as loader:
            with self.assertRaisesRegex(ValueError, "fused features are forbidden"):
                study.only_panderm("train")
            loader.assert_called_once_with(study.SOURCE, "panderm_full_frame", "train")

    def test_grouped_roles_and_source_aware_selection(self):
        meta = pd.DataFrame({"source": ["A"] * 200, "lesion_id": [str(i) for i in range(200)],
            "split_group_id": [str(i // 2) for i in range(200)], "target": [(i // 2) % 2 for i in range(200)]})
        folds = study.grouped_folds(meta)
        for fold in range(5):
            score, calibration = folds == fold, folds == (fold + 1) % 5
            fit = ~(score | calibration)
            groups = [set(meta.loc[mask, "split_group_id"]) for mask in (fit, calibration, score)]
            self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
        self.assertGreater(study.selection_key({"by_source": {"A": {"recall": .95, "specificity": .5}}, "pr_auc": .5}),
            study.selection_key({"by_source": {"A": {"recall": .94, "specificity": .99}}, "pr_auc": .99}))

    def test_synthetic_selection_precedes_validation_and_completed_resume(self):
        rng = np.random.default_rng(99)

        def cohort(count, prefix):
            y = np.tile([0, 1], count // 2)
            frame = pd.DataFrame({"source": ["A"] * count, "lesion_id": [f"{prefix}{i}" for i in range(count)],
                "split_group_id": [f"{prefix}{i}" for i in range(count)], "target": y})
            x = rng.normal(size=(count, 6)).astype("float32")
            x[:, 0] += 2 * y
            return frame, frame, x

        train, val = cohort(200, "train"), cohort(100, "val")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)

            def features(split):
                if split == "val":
                    self.assertTrue((output / "selection.json").exists())
                    self.assertTrue((output / "final_start137.json").exists())
                    self.assertTrue((output / "panderm_only_research_head.keras").exists())
                    return val
                self.assertEqual(split, "train")
                return train

            plan = {"l2_grid": [.01], "calibration_target": .96,
                "final_starts": [0, 137], "start_probability_tolerance": 1e-4}
            with patch.object(study, "only_panderm", side_effect=features), \
                    patch.object(study, "assert_training_only"):
                study.run(output, plan)
            result = json.loads((output / "validation_report.json").read_text())
            self.assertFalse(result["retained_model_comparison_valid"])
            self.assertFalse(result["pretrained_overlap_verified"])
            self.assertFalse(result["automatic_promotion"])
            with patch.object(study, "only_panderm") as loader:
                study.run(output, plan)
                loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
