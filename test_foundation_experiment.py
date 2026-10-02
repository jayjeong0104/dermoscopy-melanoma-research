from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from foundation_common import (aggregate_features, error_audit, fingerprint, inner_partition,
                               lesion_metadata, read_chunk, save_chunk, save_json)
from run_foundation_experiment import phase, verify_plan
from extract_foundation_features import checked_image_bytes, load_features


def images():
    return pd.DataFrame([
        {"source": "A", "image_id": "i1", "lesion_id": "A::1", "split_group_id": "p1",
         "target": 1, "diagnosis": "MEL", "sha256": "a", "image_path": "a.jpg"},
        {"source": "A", "image_id": "i2", "lesion_id": "A::1", "split_group_id": "p1",
         "target": 1, "diagnosis": "MEL", "sha256": "b", "image_path": "b.jpg"},
        {"source": "A", "image_id": "i3", "lesion_id": "A::2", "split_group_id": "p2",
         "target": 0, "diagnosis": "NV", "sha256": "c", "image_path": "c.jpg"}])


class FoundationIntegrityTests(unittest.TestCase):
    def test_all_photographs_aggregate_to_one_lesion(self):
        meta, features = aggregate_features(images(), np.array([[1., 3.], [3., 5.], [9., 7.]]))
        np.testing.assert_array_equal(features, [[2., 4.], [9., 7.]])
        self.assertEqual(meta.image_count.tolist(), [2, 1])

    def test_row_permutation_keeps_lesion_alignment(self):
        frame = images()
        x = np.array([[1., 3.], [3., 5.], [9., 7.]])
        first = aggregate_features(frame, x)
        second = aggregate_features(frame.iloc[[2, 0, 1]], x[[2, 0, 1]])
        self.assertTrue(first[0][["source", "lesion_id", "target"]].equals(second[0][["source", "lesion_id", "target"]]))
        np.testing.assert_array_equal(first[1], second[1])

    def test_conflicting_lesion_target_fails(self):
        frame = images()
        frame.loc[1, "target"] = 0
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            lesion_metadata(frame)

    def test_crossing_patient_groups_fails(self):
        frame = images()
        frame.loc[1, "split_group_id"] = "other"
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            lesion_metadata(frame)

    def test_nonfinite_and_misaligned_features_fail(self):
        for x in (np.zeros((2, 4)), np.full((3, 4), np.nan)):
            with self.assertRaises(ValueError):
                aggregate_features(images(), x)

    def test_test_feature_loading_is_disabled(self):
        with self.assertRaisesRegex(ValueError, "Test feature extraction"):
            load_features(Path("not-opened"), "panderm_base", "test")

    def test_image_tampering_is_rejected(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.jpg"
            path.write_bytes(b"different image bytes")
            row = SimpleNamespace(image_path=str(path), sha256="wrong", source="A", image_id="i1")
            with self.assertRaisesRegex(ValueError, "Image bytes changed"):
                checked_image_bytes(row)

    def test_inner_partition_is_patient_grouped_and_deterministic(self):
        meta = pd.DataFrame([{"source": "A", "lesion_id": f"l{i}", "target": (i // 2) % 2,
                              "split_group_id": f"p{i // 2}"} for i in range(200)])
        held = inner_partition(meta)
        np.testing.assert_array_equal(held, inner_partition(meta))
        self.assertFalse(set(meta.loc[held, "split_group_id"]) & set(meta.loc[~held, "split_group_id"]))
        self.assertEqual(set(meta.loc[held, "target"]), {0, 1})

    def test_chunk_roundtrip_and_fingerprint_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chunk.npz"
            x = np.ones((3, 4), dtype="float32")
            save_chunk(path, x, "correct")
            np.testing.assert_array_equal(read_chunk(path, "correct", 3, 4), x)
            with self.assertRaisesRegex(ValueError, "mismatched"):
                read_chunk(path, "other", 3, 4)
            with self.assertRaisesRegex(ValueError, "Invalid"):
                read_chunk(path, "correct", 2, 4)

    def test_corrupted_feature_content_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chunk.npz"
            np.savez(path, features=np.ones((3, 4), dtype="float32"), fingerprint="correct", features_sha256="bad")
            with self.assertRaisesRegex(ValueError, "integrity"):
                read_chunk(path, "correct", 3, 4)

    def test_fingerprint_depends_on_preprocessing_and_order(self):
        a = {"size": 224, "images": ["a", "b"]}
        self.assertNotEqual(fingerprint(a), fingerprint({**a, "size": 300}))
        self.assertNotEqual(fingerprint(a), fingerprint({**a, "images": ["b", "a"]}))

    def test_error_audit_does_not_change_labels(self):
        frame = images()
        original = frame.copy(deep=True)
        reference = lesion_metadata(frame)[["source", "lesion_id", "split_group_id", "target"]]
        reference = reference.assign(prediction=[0, 1], probability=[.01, .9], threshold=.1, policy_fold=[0, 1])
        with tempfile.TemporaryDirectory() as directory:
            report = error_audit(reference, frame, Path(directory))
            self.assertEqual(report["metrics"]["false_negatives"], 1)
            self.assertEqual(report["metrics"]["false_positives"], 1)
            self.assertFalse(report["expert_review_completed"])
            self.assertTrue((Path(directory) / "expert_review_queue.csv").exists())
        pd.testing.assert_frame_equal(frame, original)

    def test_error_audit_rejects_unknown_lesion(self):
        frame = images()
        reference = lesion_metadata(frame)[["source", "lesion_id", "split_group_id", "target"]]
        reference.loc[0, "lesion_id"] = "unknown"
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(ValueError):
            error_audit(reference.assign(prediction=0), frame, Path(directory))

    def test_plan_detects_code_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_json(root / "experiment_plan.json", {"code_sha256": {"code.py": "bad"}, "input_sha256": {}})
            (root / "code.py").write_text("changed", encoding="utf-8")
            with patch("run_foundation_experiment.ROOT", root), self.assertRaisesRegex(ValueError, "code changed"):
                verify_plan(root)

    def test_phase_does_not_restart_verified_work(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_json(root / "phase_done/test.json", {"command": ["python", "job.py"], "outputs": {"output": "hash"}})
            with patch("run_foundation_experiment.verify_plan"), patch("run_foundation_experiment.file_hash", return_value="hash"), patch("run_foundation_experiment.subprocess.Popen") as child:
                phase(root, "test", ["python", "job.py"], ["output"])
                child.assert_not_called()

    def test_phase_rejects_changed_completed_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_json(root / "phase_done/test.json", {"command": ["python"], "outputs": {"output": "hash"}})
            with patch("run_foundation_experiment.verify_plan"), patch("run_foundation_experiment.file_hash", return_value="changed"), self.assertRaisesRegex(ValueError, "output changed"):
                phase(root, "test", ["python"], ["output"])

    def test_child_is_stopped_if_status_write_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("run_foundation_experiment.verify_plan"), patch(
                    "run_foundation_experiment.subprocess.Popen") as child, patch(
                    "run_foundation_experiment.save_json", side_effect=PermissionError("locked")):
                with self.assertRaises(PermissionError):
                    phase(root, "test", ["python"], ["output"])
            child.return_value.terminate.assert_called_once()
            child.return_value.wait.assert_called_once()

    def test_io_repair_preserves_plan_and_rejects_subsequent_changes(self):
        from compare_model_runs import file_hash
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old, overrides = {}, {}
            for name in ("foundation_common.py", "train_foundation_head.py", "run_foundation_experiment.py"):
                before = root / "code_snapshot" / name
                after = root / "io_repair_snapshot" / name
                before.parent.mkdir(exist_ok=True)
                after.parent.mkdir(exist_ok=True)
                before.write_text("old")
                after.write_text("patched")
                (root / name).write_text("patched")
                old[name] = file_hash(before)
                overrides[name] = {"before_sha256": old[name], "after_sha256": file_hash(after)}
            (root / "artifact_io.py").write_text("helper")
            (root / "io_repair_snapshot/artifact_io.py").write_text("helper")
            plan = {"code_sha256": old, "input_sha256": {}}
            save_json(root / "experiment_plan.json", plan)
            plan_hash = file_hash(root / "experiment_plan.json")
            record = {"version": 1, "scope": "artifact_io_only", "original_plan_sha256": plan_hash,
                      "code_overrides": overrides,
                      "additional_code_sha256": {"artifact_io.py": file_hash(root / "artifact_io.py")}}
            save_json(root / "io_repair.json", record)
            with patch("run_foundation_experiment.ROOT", root):
                self.assertEqual(verify_plan(root), plan)
                self.assertEqual(file_hash(root / "experiment_plan.json"), plan_hash)
                (root / "artifact_io.py").write_text("unexpected change")
                with self.assertRaisesRegex(ValueError, "code changed"):
                    verify_plan(root)
                (root / "artifact_io.py").write_text("helper")
                (root / "code_snapshot/foundation_common.py").write_text("changed original")
                with self.assertRaisesRegex(ValueError, "provenance changed"):
                    verify_plan(root)


if __name__ == "__main__":
    unittest.main()
