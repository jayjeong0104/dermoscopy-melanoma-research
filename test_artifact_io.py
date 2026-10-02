from __future__ import annotations

import errno
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from artifact_io import atomic_output, replace_with_retry, save_csv, save_model
from foundation_common import save_json


class ArtifactIOTests(unittest.TestCase):
    def test_transient_sharing_error_retried_without_deleting_previous_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "progress.json"
            save_json(path, {"epoch": 1})
            replace = os.replace
            attempts = []

            def locked_then_available(source, destination):
                attempts.append(source)
                self.assertEqual(json.loads(path.read_text()), {"epoch": 1})
                if len(attempts) <= 3:
                    raise PermissionError(errno.EACCES, "simulated sharing lock")
                replace(source, destination)

            with patch("artifact_io.os.replace", side_effect=locked_then_available), patch(
                    "artifact_io.time.sleep") as sleep, self.assertWarns(RuntimeWarning):
                save_json(path, {"epoch": 2})
            self.assertEqual(sleep.call_count, 3)
            self.assertEqual(len(set(attempts)), 1)
            self.assertEqual(json.loads(path.read_text()), {"epoch": 2})
            self.assertEqual(list(root.iterdir()), [path])

    def test_persistent_lock_is_bounded_and_preserves_both_versions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "progress.json"
            save_json(path, {"epoch": 1})
            with patch("artifact_io.os.replace", side_effect=PermissionError(errno.EACCES, "locked")), patch(
                    "artifact_io.time.monotonic", side_effect=[0., 0., 60.]), patch(
                    "artifact_io.time.sleep") as sleep, self.assertWarns(RuntimeWarning):
                with self.assertRaises(PermissionError) as error:
                    save_json(path, {"epoch": 2})
            self.assertEqual(sleep.call_count, 1)
            self.assertEqual(json.loads(path.read_text()), {"epoch": 1})
            prepared = list(root.glob("*.pending.json"))
            self.assertEqual(len(prepared), 1)
            self.assertEqual(json.loads(prepared[0].read_text()), {"epoch": 2})
            self.assertIn("retained", error.exception.__notes__[0])

    def test_unrelated_disk_error_fails_immediately(self):
        with patch("artifact_io.os.replace", side_effect=OSError(errno.ENOSPC, "disk full")), patch(
                "artifact_io.time.sleep") as sleep:
            with self.assertRaises(OSError):
                replace_with_retry(Path("prepared"), Path("previous"))
            sleep.assert_not_called()

    def test_windows_sharing_violation_code_is_retried(self):
        error = OSError("sharing violation")
        error.winerror = 32
        with patch("artifact_io.os.replace", side_effect=[error, None]) as replace, patch(
                "artifact_io.time.sleep"), self.assertWarns(RuntimeWarning):
            replace_with_retry(Path("prepared"), Path("previous"))
        self.assertEqual(replace.call_count, 2)

    def test_serialization_failure_keeps_previous_checkpoint(self):
        class BrokenModel:
            def save(self, path):
                path.write_bytes(b"partial checkpoint")
                raise ValueError("serialization failure")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.keras"
            path.write_bytes(b"last good checkpoint")
            with self.assertRaisesRegex(ValueError, "serialization failure"):
                save_model(BrokenModel(), path)
            self.assertEqual(path.read_bytes(), b"last good checkpoint")
            self.assertEqual(list(path.parent.iterdir()), [path])

    def test_serializers_keep_suffix_and_contents(self):
        class Model:
            def save(self, path):
                if path.suffix != ".keras":
                    raise ValueError("wrong serialization format")
                path.write_bytes(b"checkpoint")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_model(Model(), root / "model.keras")
            save_csv(pd.DataFrame({"a": [1, 2]}), root / "predictions.csv")
            self.assertEqual((root / "model.keras").read_bytes(), b"checkpoint")
            self.assertEqual(pd.read_csv(root / "predictions.csv").to_dict("list"), {"a": [1, 2]})

    def test_nonfinite_json_never_replaces_previous(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "progress.json"
            save_json(path, {"epoch": 1})
            with self.assertRaises(ValueError):
                save_json(path, {"loss": float("nan")})
            self.assertEqual(json.loads(path.read_text()), {"epoch": 1})

    def test_temporary_names_are_unique_and_same_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            with atomic_output(path) as first:
                first.write_text("first")
                with atomic_output(path) as second:
                    self.assertNotEqual(first, second)
                    self.assertEqual(first.parent, path.parent)
                    self.assertEqual(second.parent, path.parent)
                    second.write_text("second")
                self.assertEqual(path.read_text(), "second")
            self.assertEqual(path.read_text(), "first")

    def test_repeated_progress_publication_roundtrips(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "progress.json"
            for epoch in range(200):
                value = {"epoch": epoch, "history": list(range(epoch + 1))}
                save_json(path, value)
                self.assertEqual(json.loads(path.read_text()), value)
            self.assertEqual(list(path.parent.iterdir()), [path])


if __name__ == "__main__":
    unittest.main()
