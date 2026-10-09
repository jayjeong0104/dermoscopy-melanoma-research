import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from external_evaluation_contract import (file_hash, recorded_exposure_manifests,
    verify_inference_binding)


class ExternalEvaluationContractTests(unittest.TestCase):
    def test_changed_checkpoint_and_inference_artifacts_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "best_model.keras"
            model.write_bytes(b"model")
            calibration = root / "calibration.json"
            calibration.write_text('{"temperature":1}')
            status = {"status": "complete", "artifact_sha256": {"best_model.keras": file_hash(model), "calibration.json": file_hash(calibration)}}
            self.assertTrue(verify_inference_binding(root, model, status)["checkpoint_hash_verified"])
            calibration.write_text('{"temperature":2}')
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                verify_inference_binding(root, model, status)
            model.write_bytes(b"other")
            with self.assertRaisesRegex(ValueError, "Checkpoint differs"):
                verify_inference_binding(root, model, status)

    def test_legacy_nonbest_cannot_reuse_unbound_calibration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            other = root / "best_head_model.keras"
            other.write_bytes(b"head")
            with self.assertRaisesRegex(ValueError, "non-best"):
                verify_inference_binding(root, other, {"status": "complete"})
            another_run = root / "another_run"
            another_run.mkdir()
            same_filename = another_run / "best_model.keras"
            same_filename.write_bytes(b"unbound")
            with self.assertRaisesRegex(ValueError, "non-best"):
                verify_inference_binding(root, same_filename, {"status": "complete"})

    def test_overlap_manifests_are_actual_run_and_recursive_ancestors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent, child = root / "parent", root / "child"
            for run in (parent, child):
                run.mkdir()
                (run / "best_model.keras").write_bytes(run.name.encode())
                for split in ("train", "val"):
                    pd.DataFrame({"sha256": [run.name + split]}).to_csv(run / f"{split}_split.csv", index=False)
            (parent / "run_config.json").write_text(json.dumps({"weights": "imagenet", "warm_start_model": None}))
            (child / "run_config.json").write_text(json.dumps({"warm_start_model": str(parent / "best_model.keras")}))
            paths, report = recorded_exposure_manifests(child / "best_model.keras", root)
            self.assertEqual(set(paths), {run / f"{split}_split.csv" for run in (parent, child) for split in ("train", "val")})
            self.assertEqual(len(report["local_lineage"]), 2)
            (parent / "run_config.json").write_text(json.dumps({"warm_start_model": str(child / "best_model.keras")}))
            with self.assertRaisesRegex(ValueError, "Cyclic"):
                recorded_exposure_manifests(child / "best_model.keras", root)

    def test_external_hash_overlapping_only_ancestor_is_rejected(self):
        from evaluate_derm12345 import audit_exact_overlap
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "external.jpg"
            image.write_bytes(b"external-image")
            current, ancestor = root / "current.csv", root / "ancestor.csv"
            pd.DataFrame({"sha256": ["different"]}).to_csv(current, index=False)
            pd.DataFrame({"sha256": [file_hash(image)]}).to_csv(ancestor, index=False)
            frame = pd.DataFrame({"image_path": [image], "patient_id": ["external-patient"]})
            with self.assertRaisesRegex(RuntimeError, "exact training-image"):
                audit_exact_overlap(frame, [current, ancestor], workers=1)


if __name__ == "__main__":
    unittest.main()
