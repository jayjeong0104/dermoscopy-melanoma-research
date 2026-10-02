import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from compare_model_runs import compare_frames, compare_runs, file_hash, operating_metrics


def predictions(scores):
    frame = pd.DataFrame({
        "source": ["fixture"] * 6, "lesion_id": [str(i) for i in range(6)],
        "target": [0, 0, 0, 1, 1, 1], "probability": scores,
    })
    return frame.set_index(["source", "lesion_id"])


class PromotionTests(unittest.TestCase):
    def test_worse_candidate_keeps_incumbent(self):
        baseline = predictions([.1, .2, .3, .7, .8, .9])
        candidate = predictions([.8, .7, .6, .3, .2, .1])
        result = compare_frames(baseline, candidate)
        self.assertEqual(result["decision"], "retain_baseline")
        self.assertFalse(result["candidate_passes"])

    def test_better_candidate_is_only_eligible_for_confirmation(self):
        baseline = predictions([.8, .7, .6, .3, .2, .1])
        candidate = predictions([.1, .2, .3, .7, .8, .9])
        result = compare_frames(baseline, candidate)
        self.assertEqual(result["decision"], "candidate_eligible_for_confirmation")
        self.assertTrue(result["candidate_passes"])

    def test_identical_model_does_not_count_as_improvement(self):
        frame = predictions([.1, .2, .3, .7, .8, .9])
        self.assertFalse(compare_frames(frame, frame)["candidate_passes"])

    def test_label_or_identity_mismatch_is_rejected(self):
        baseline = predictions([.1, .2, .3, .7, .8, .9])
        candidate = baseline.copy()
        candidate.iloc[0, candidate.columns.get_loc("target")] = 1
        with self.assertRaisesRegex(ValueError, "labels"):
            compare_frames(baseline, candidate)
        with self.assertRaisesRegex(ValueError, "identities"):
            compare_frames(baseline, baseline.iloc[::-1])

    def test_tied_scores_meet_required_recall(self):
        result = operating_metrics(np.array([0, 0, 1, 1, 1]), np.array([.2, .8, .8, .8, .9]), .95)
        self.assertEqual(result["threshold"], .8)
        self.assertEqual(result["recall"], 1)
        self.assertEqual(result["false_positives"], 1)

    def test_invalid_scores_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            operating_metrics(np.array([0, 1]), np.array([np.nan, .9]), .95)

    def make_run(self, root, name, scores):
        run = root / name
        run.mkdir()
        (run / "best_model.keras").write_bytes(b"fixture model " + name.encode())
        predictions(scores).reset_index().drop(columns="probability").to_csv(run / "val_split.csv", index=False)
        predictions(scores).iloc[::-1].reset_index().to_csv(run / "val_lesion_predictions.csv", index=False)
        (run / "run_config.json").write_text(json.dumps({"binary_target": "melanoma", "image_size": 300}))
        (run / "calibration.json").write_text(json.dumps({"tta_transforms": 8}))
        (run / "finalization_status.json").write_text(json.dumps({
            "status": "complete", "artifact_sha256": {
                "best_model.keras": file_hash(run / "best_model.keras"),
                "val_split.csv": file_hash(run / "val_split.csv"),
            },
        }))
        return run

    def test_runs_need_no_test_or_external_files_and_do_not_mutate_models(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = self.make_run(root, "old", [.1, .2, .3, .7, .8, .9])
            candidate = self.make_run(root, "new", [.8, .7, .6, .3, .2, .1])
            before = file_hash(baseline / "best_model.keras")
            result = compare_runs(baseline, candidate)
            self.assertEqual(result["decision"], "retain_baseline")
            self.assertEqual(file_hash(baseline / "best_model.keras"), before)
            (candidate / "best_model.keras").write_bytes(b"changed model")
            with self.assertRaisesRegex(ValueError, "Model differs"):
                compare_runs(baseline, candidate)

    def test_legacy_and_corrected_tta_cannot_be_compared_as_same_protocol(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = self.make_run(root, "old", [.1, .2, .3, .7, .8, .9])
            candidate = self.make_run(root, "new", [.1, .2, .3, .7, .8, .9])
            (candidate / "calibration.json").write_text(json.dumps({"tta_transforms": 8, "tta_scheme": "d4_v2"}))
            with self.assertRaisesRegex(ValueError, "tta_scheme"):
                compare_runs(baseline, candidate)


if __name__ == "__main__":
    unittest.main()
