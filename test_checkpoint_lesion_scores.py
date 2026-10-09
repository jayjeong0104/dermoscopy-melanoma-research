from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from checkpoint_lesion_scores import DIAGNOSES, load_checkpoint_scores
from compare_model_runs import file_hash
from foundation_common import aggregate_features, fingerprint, save_chunk


class CheckpointLesionScoreTests(unittest.TestCase):
    def fixture(self, root):
        run, cache = root / "run", root / "features"
        run.mkdir(); cache.mkdir()
        frame = pd.DataFrame({"source": ["S"] * 3, "image_id": ["i1", "i2", "i3"],
            "lesion_id": ["S::a", "S::a", "S::b"], "split_group_id": ["a", "a", "b"],
            "target": [1, 1, 0], "diagnosis": ["MEL", "MEL", "NV"],
            "image_path": ["absent1.jpg", "absent2.jpg", "absent3.jpg"], "sha256": ["h1", "h2", "h3"]})
        frame.to_csv(run / "val_split.csv", index=False)
        (run / "best_finetune_model.keras").write_bytes(b"synthetic checkpoint; never executed")
        (run / "run_config.json").write_text(json.dumps({"binary_target": "melanoma", "diagnoses": DIAGNOSES, "image_size": 300}))
        plan = {"version": 3, "model_sha256": file_hash(run / "best_finetune_model.keras"),
            "splits_sha256": {"val": file_hash(run / "val_split.csv")}, "chunk_images": 2,
            "metadata": "unknown", "tta_transforms": 1, "image_size": 300}
        (cache / "plan.json").write_text(json.dumps(plan))
        values = np.zeros((3, 1289), dtype="float32")
        values[:, 1280] = [.1, .8, .7]
        values[:, 1285] = [.2, .6, .1]
        values[:, 1286] = [.8, .4, .9]
        for start in (0, 2):
            rows = frame.iloc[start:start + 2]
            key = fingerprint({"plan": plan, "split": "val", "start": start,
                "rows": rows[["source", "image_id", "sha256"]].to_dict("records")})
            save_chunk(cache / f"val_{start:06d}.npz", values[start:start + 2], key)
        meta, means = aggregate_features(frame, values)
        # Exercise explicit alignment to the existing cache's order.
        meta.iloc[::-1].to_csv(cache / "val_lesions.csv", index=False)
        save_chunk(cache / "val_lesion_features.npz", means[::-1, :1280], fingerprint(plan))
        names = ["plan.json", "val_lesions.csv", "val_lesion_features.npz"]
        (cache / "val_complete.json").write_text(json.dumps({"plan_fingerprint": fingerprint(plan),
            "outputs_sha256": {name: file_hash(cache / name) for name in names}}))
        return run, cache

    def test_multiview_matches_training_logit_mean_and_preserves_diagnosis_mean_and_order(self):
        with TemporaryDirectory(dir=Path(__file__).parent) as directory:
            run, cache = self.fixture(Path(directory))
            before = {p.name: file_hash(p) for p in cache.iterdir()}
            meta = load_checkpoint_scores(cache, run)
            self.assertEqual(meta.lesion_id.tolist(), ["S::b", "S::a"])
            # Odds(.1)=1/9, odds(.8)=4; geometric mean odds=2/3 -> probability=.4.
            np.testing.assert_allclose(meta.direct_probability, [.7, .4], atol=1e-7)
            np.testing.assert_allclose(meta.diagnosis_probability, [.1, .4], atol=1e-7)
            self.assertNotAlmostEqual(float(meta.direct_probability.iloc[1]), .45)
            self.assertEqual(meta.direct_probability.dtype, np.dtype("float32"))
            self.assertEqual(before, {p.name: file_hash(p) for p in cache.iterdir()})

    def test_rejects_different_checkpoint_or_split(self):
        for name, message in (("best_finetune_model.keras", "checkpoint binding"), ("val_split.csv", "split binding")):
            with self.subTest(name=name), TemporaryDirectory(dir=Path(__file__).parent) as directory:
                run, cache = self.fixture(Path(directory))
                with (run / name).open("ab") as stream:
                    stream.write(b"changed")
                with self.assertRaisesRegex(ValueError, message):
                    load_checkpoint_scores(cache, run)

    def test_rejects_corrupt_per_image_chunk(self):
        with TemporaryDirectory(dir=Path(__file__).parent) as directory:
            run, cache = self.fixture(Path(directory))
            path = cache / "val_000000.npz"
            with np.load(path) as stored:
                payload = {name: stored[name].copy() for name in stored.files}
            payload["features"][0, 1280] = .9
            with path.open("wb") as stream:
                np.savez(stream, **payload)
            with self.assertRaisesRegex(ValueError, "integrity failure"):
                load_checkpoint_scores(cache, run)

    def test_rejects_test_before_reading_any_artifact(self):
        with patch("checkpoint_lesion_scores.file_hash", side_effect=AssertionError("Should not read files")):
            with self.assertRaisesRegex(ValueError, "test access"):
                load_checkpoint_scores(Path("missing"), Path("missing"), "test")


if __name__ == "__main__":
    unittest.main()
