"""CPU-only checks for lesion alignment and completed-checkpoint provenance."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from compare_model_runs import file_hash
import run_precision_rebuild as study


class PrecisionRebuildTests(unittest.TestCase):
    def metadata(self):
        return pd.DataFrame({"source": ["A", "A", "B"], "lesion_id": ["a", "b", "c"],
            "split_group_id": ["p1", "p2", "p3"], "target": [0, 1, 0], "image_count": [2, 1, 3]})

    def test_features_follow_lesion_identity_when_rows_are_reordered(self):
        meta = self.metadata()
        features = np.repeat(np.arange(3)[:, None], 768, axis=1).astype(float)
        wanted = meta.iloc[[2, 0, 1]].reset_index(drop=True)
        with patch.object(study, "only_panderm", return_value=(None, meta, features)):
            observed = study.aligned_panderm("val", wanted)
        np.testing.assert_array_equal(observed, features[[2, 0, 1]])

    def test_duplicate_clean_lesion_cannot_silently_omit_another_lesion(self):
        meta = self.metadata()
        wanted = meta.iloc[[0, 0, 2]].reset_index(drop=True)
        with patch.object(study, "only_panderm", return_value=(None, meta, np.zeros((3, 768)))):
            with self.assertRaisesRegex(ValueError, "unique complete identities"):
                study.aligned_panderm("train", wanted)

    def test_label_group_and_image_count_mismatches_fail(self):
        meta = self.metadata()
        for column, value in (("target", 1), ("split_group_id", "different"), ("image_count", 4)):
            wanted = meta.copy()
            wanted.loc[0, column] = value
            with self.subTest(column=column), patch.object(study, "only_panderm", return_value=(None, meta, np.zeros((3, 768)))):
                with self.assertRaises(ValueError):
                    study.aligned_panderm("val", wanted)

    def test_corrupt_feature_values_and_dimensions_fail(self):
        meta = self.metadata()
        for features in (np.zeros((3, 767)), np.full((3, 768), np.nan)):
            with patch.object(study, "only_panderm", return_value=(None, meta, features)):
                with self.assertRaisesRegex(ValueError, "dimensions or values"):
                    study.aligned_panderm("val", meta)

    def test_original_completion_hash_rejects_self_consistent_replaced_head(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            head = directory / "nested" / "head.npz"
            head.parent.mkdir()
            head.write_bytes(b"original")
            (directory / "complete.json").write_text(json.dumps({"outputs_sha256": {
                "nested\\head.npz": file_hash(head)}}))
            study.verify_completed_artifacts(directory, ["nested/head.npz"])
            head.write_bytes(b"replacement")
            with self.assertRaisesRegex(ValueError, "original completion"):
                study.verify_completed_artifacts(directory, ["nested/head.npz"])

    def test_blend_comes_from_first_best_epoch_and_verified_log(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            run = directory / "training" / "run_synthetic"
            run.mkdir(parents=True)
            (run / "best_finetune_model.keras").write_bytes(b"synthetic-checkpoint")
            (run / "run_config.json").write_text(json.dumps({"checkpoint_metric": "lesion_pr_auc"}))
            history = pd.DataFrame({"epoch": [0, 1, 2], "val_lesion_pr_auc": [.3, .5, .5],
                "val_lesion_direct_weight": [.8, .05, .9]})
            history.to_csv(run / "training_finetune.csv", index=False)
            files = list(run.iterdir())
            (directory / "complete.json").write_text(json.dumps({"outputs_sha256": {
                str(path.relative_to(directory)).replace("/", "\\"): file_hash(path) for path in files}}))
            selected = study.saved_checkpoint_blend(run)
            self.assertEqual(selected["epoch_zero_based"], 1)
            self.assertEqual(selected["direct_weight"], .05)
            history.loc[1, "val_lesion_direct_weight"] = .7
            history.to_csv(run / "training_finetune.csv", index=False)
            with self.assertRaisesRegex(ValueError, "original completion"):
                study.saved_checkpoint_blend(run)


if __name__ == "__main__":
    unittest.main()
