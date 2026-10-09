"""Synthetic CPU check of training selection before validation-feature access."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from compare_model_runs import file_hash
import run_natural_feature_heads as study


class NaturalFeatureHeadTests(unittest.TestCase):
    def test_all_arm_selection_and_fits_precede_validation_and_resume_is_read_only(self):
        rng = np.random.default_rng(327)

        def cohort(count, prefix):
            truth = np.tile([0, 1], count // 2)
            meta = pd.DataFrame({"source": ["A"] * (count // 2) + ["B"] * (count // 2),
                "lesion_id": [f"{prefix}{index}" for index in range(count)],
                "split_group_id": [f"{prefix}{index}" for index in range(count)], "target": truth,
                "direct_probability": .2 + .6 * truth, "diagnosis_probability": .3 + .4 * truth})
            x = rng.normal(size=(count, 6)); x[:, 0] += 2 * truth
            return meta, {"natural_b1": x[:, :3], "natural_b1_plus_panderm": x,
                "corrected_panderm_only": x[:, 3:]}

        train, val = cohort(200, "train"), cohort(100, "val")
        with tempfile.TemporaryDirectory() as temporary, threadpool_limits(limits=2):
            root = Path(temporary); features = root / "features"; features.mkdir()
            output = root / "study"; output.mkdir()
            for name in ("plan.json", "train_complete.json", "val_complete.json"):
                (features / name).write_text("{}")
            protocol = {"feature_plan_sha256": file_hash(features / "plan.json"),
                "panderm_feature_inputs_sha256": {},
                "arms": ["natural_b1", "natural_b1_plus_panderm"], "evaluation_arms": list(train[1]),
                "fixed_panderm_l2": .01, "l2_grid": [.1], "folds": 5, "calibration_target": .95,
                "final_starts": [0, 137], "development_policy_seed": 20261006,
                "reference_blend": {"direct_weight": 1.}, "limitations": []}
            access = []

            def load(split):
                access.append(split)
                if split == "val":
                    marker = json.loads((output / "training_complete.json").read_text())
                    self.assertFalse(marker["outer_validation_used_for_selection"])
                    self.assertEqual(set(marker["heads_sha256"]), set(protocol["evaluation_arms"]))
                    selected = json.loads((output / "selection.json").read_text())
                    self.assertEqual(set(selected), set(protocol["evaluation_arms"]))
                    for arm in protocol["evaluation_arms"]:
                        self.assertTrue((output / f"{arm}_final_start137.json").exists())
                    return val
                self.assertEqual(split, "train")
                return train

            with patch.object(study, "FEATURES", features), patch.object(study, "feature_arms", side_effect=load), \
                    patch.object(study, "checkpoint_scores", return_value=val[0]), \
                    patch.object(study, "panderm_input_hashes", return_value={}), patch("builtins.print"):
                study.execute(output, protocol)
            self.assertEqual(access, ["train", "val"])
            self.assertFalse(json.loads((output / "complete.json").read_text())["automatic_promotion"])
            with patch.object(study, "feature_arms") as loader:
                study.execute(output, protocol)
                loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
