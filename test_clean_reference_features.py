"""Guard feature identity, missing-metadata policy, and forbidden test access."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from compare_model_runs import file_hash
from extract_clean_reference import (aggregate_checkpoint_values, load_extracted, rows_for,
                                     unknown_metadata, verify_run)
from foundation_common import aggregate_features, save_chunk


class CleanFeatureTests(unittest.TestCase):
    def test_completed_run_binding_and_cross_platform_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "training" / "run_example"
            run.mkdir(parents=True)
            config = {"binary_target": "melanoma", "warm_start_model": None, "weights": "imagenet"}
            (run / "run_config.json").write_text(json.dumps(config))
            for name in ("best_finetune_model.keras", "train_split.csv", "val_split.csv", "metadata_schema.json"):
                (run / name).write_text("fixture")
            recorded = {str(path.relative_to(root)).replace("/", "\\"): file_hash(path) for path in run.iterdir()}
            (root / "complete.json").write_text(json.dumps({"outputs_sha256": recorded}))
            self.assertEqual(verify_run(run), config)
            (run / "val_split.csv").write_text("changed")
            with self.assertRaisesRegex(ValueError, "changed since completion"):
                verify_run(run)

    def test_no_test_split_entry_points(self):
        with self.assertRaisesRegex(ValueError, "Only training"):
            rows_for(Path("unused"), "test")
        with self.assertRaisesRegex(ValueError, "Test feature"):
            load_extracted(Path("unused"), "test")

    def test_unknown_metadata_matches_inference_semantics(self):
        names = ["age_scaled", "age_missing", "sex_female", "sex_unknown", "site_torso", "site_unknown"]
        np.testing.assert_allclose(unknown_metadata({"age_median": 55., "feature_names": names}),
                                  [.55, 1., 0., 1., 0., 1.])

    def test_average_outputs_and_features_by_lesion_not_photo_count(self):
        frame = pd.DataFrame({"source": ["S"] * 3, "image_id": ["i1", "i2", "i3"],
            "lesion_id": ["S::a", "S::a", "S::b"], "split_group_id": ["S::a", "S::a", "S::b"],
            "target": [1, 1, 0], "diagnosis": ["MEL", "MEL", "NV"], "image_path": ["a", "b", "c"],
            "sha256": ["1", "2", "3"]})
        meta, values = aggregate_features(frame, np.array([[1., .4], [3., .8], [7., .1]]))
        self.assertEqual(meta.lesion_id.tolist(), ["S::a", "S::b"])
        np.testing.assert_allclose(values, [[2., .6], [7., .1]])

    def test_corrupted_completed_cache_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            pd.DataFrame({"source": ["S"], "lesion_id": ["a"]}).to_csv(output / "val_lesions.csv", index=False)
            save_chunk(output / "val_lesion_features.npz", np.zeros((1, 1280)), "plan")
            names = ["val_lesions.csv", "val_lesion_features.npz"]
            (output / "val_complete.json").write_text(json.dumps({"plan_fingerprint": "plan",
                "outputs_sha256": {name: file_hash(output / name) for name in names}}))
            load_extracted(output, "val")
            (output / "val_lesions.csv").write_text("modified")
            with self.assertRaisesRegex(ValueError, "changed"):
                load_extracted(output, "val")

    def test_checkpoint_direct_scores_use_mean_logits_and_keep_other_means(self):
        frame = pd.DataFrame({"source": ["S"] * 3, "image_id": ["i1", "i2", "i3"],
            "lesion_id": ["S::a", "S::a", "S::b"], "split_group_id": ["a", "a", "b"],
            "target": [1, 1, 0], "diagnosis": ["MEL", "MEL", "NV"],
            "image_path": ["a", "b", "c"], "sha256": ["1", "2", "3"]})
        values = np.zeros((3, 1289), dtype="float32")
        values[:, 0] = [1., 3., 7.]
        values[:, 1280] = [.1, .8, .7]
        values[:, 1285] = [.2, .6, .1]
        values[:, 1286] = [.8, .4, .9]
        _, arithmetic = aggregate_features(frame, values)
        meta, pooled = aggregate_checkpoint_values(frame, values)
        self.assertEqual(meta.lesion_id.tolist(), ["S::a", "S::b"])
        # Geometric mean odds sqrt((1/9)*4)=2/3 gives probability .4, not .45.
        np.testing.assert_allclose(pooled[:, 1280], [.4, .7], atol=1e-7)
        np.testing.assert_array_equal(pooled[:, :1280], arithmetic[:, :1280])
        np.testing.assert_array_equal(pooled[:, 1281:], arithmetic[:, 1281:])
        self.assertEqual(pooled.dtype, np.dtype("float32"))
        permuted, observed = aggregate_checkpoint_values(frame.iloc[[2, 0, 1]], values[[2, 0, 1]])
        self.assertEqual(permuted.lesion_id.tolist(), meta.lesion_id.tolist())
        np.testing.assert_array_equal(observed, pooled)

    def test_checkpoint_direct_scores_clip_zero_and_one_before_logits(self):
        frame = pd.DataFrame({"source": ["S"] * 4, "image_id": ["i1", "i2", "i3", "i4"],
            "lesion_id": ["a", "a", "b", "c"], "split_group_id": ["a", "a", "b", "c"],
            "target": [1, 1, 0, 0], "diagnosis": ["MEL", "MEL", "NV", "NV"],
            "image_path": ["a", "b", "c", "d"], "sha256": ["1", "2", "3", "4"]})
        values = np.zeros((4, 1289), dtype="float32")
        values[:, 1280] = [0., 1., 0., 1.]
        values[:, 1286] = 1.
        _, pooled = aggregate_checkpoint_values(frame, values)
        self.assertTrue(np.isfinite(pooled).all())
        np.testing.assert_allclose(pooled[:, 1280], [.5, 1e-6, 1. - 1e-6], atol=1e-8)


if __name__ == "__main__":
    unittest.main()
