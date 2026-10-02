import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from predict_research import aggregate_scores, verify_checkpoint
import train_multitask as training


class ResearchInferenceTests(unittest.TestCase):
    def test_aggregation_matches_archived_training_helper(self):
        direct = np.array([.02, .3, .15], dtype="float32")
        diagnoses = np.random.default_rng(4).dirichlet(np.ones(8), size=3).astype("float32")
        frame = pd.DataFrame({"lesion_id": ["one"] * 3, "target": [0] * 3, "diagnosis": ["NV"] * 3,
            "source": ["synthetic"] * 3, "target_name": ["not_melanoma"] * 3,
            "diagnosis_index": [5] * 3})
        lesions, averaged = training.aggregate_predictions_by_lesion(frame, direct, diagnoses)
        config = {"direct_weight": .35, "temperature": .76}
        expected = training.apply_temperature(.35 * lesions.probability.to_numpy() + .65 * averaged[:, 4], .76)[0]
        self.assertAlmostEqual(aggregate_scores(direct, diagnoses, config), float(expected), places=7)

    def test_changed_weights_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.keras"
            path.write_bytes(b"reference")
            manifest = {"checkpoint_sha256": hashlib.sha256(b"reference").hexdigest()}
            self.assertEqual(verify_checkpoint(path, manifest), manifest["checkpoint_sha256"])
            path.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                verify_checkpoint(path, manifest)

    def test_invalid_scores_are_rejected(self):
        with self.assertRaises(ValueError):
            aggregate_scores(np.array([np.nan]), np.zeros((1, 8)), {"direct_weight": .35, "temperature": .76})


if __name__ == "__main__":
    unittest.main()
