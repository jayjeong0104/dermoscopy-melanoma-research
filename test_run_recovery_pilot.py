from pathlib import Path
import json
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd

import run_recovery_pilot as pilot


class PilotTests(unittest.TestCase):
    def test_completed_experiment_is_not_restarted(self) -> None:
        with TemporaryDirectory() as temporary, patch.object(pilot, "run_training") as train:
            root = Path(temporary)
            (root / "status.json").write_text(json.dumps({"status": "complete"}))
            (root / "validation_report.json").write_text("{}")
            with patch("sys.argv", ["run_recovery_pilot.py", "--experiment-dir", str(root)]):
                pilot.main()
            train.assert_not_called()

    def test_inconsistent_completed_state_requires_attention(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "status.json").write_text(json.dumps({"status": "complete"}))
            with self.assertRaisesRegex(ValueError, "missing its validation report"):
                pilot.completed_report(root)

    def test_holdout_comparison_ignores_row_order_not_identity(self) -> None:
        frame = pd.DataFrame({
            "source": ["a", "a"], "image_id": ["001", "002"],
            "lesion_id": ["l1", "l2"], "split_group_id": ["p1", "p2"],
            "sha256": ["hash1", "hash2"], "diagnosis": ["NV", "MEL"],
        })
        pilot.assert_same_holdout(frame, frame.iloc[::-1])
        for field in ("image_id", "sha256", "diagnosis", "split_group_id"):
            changed = frame.copy()
            changed.loc[0, field] = "changed"
            with self.assertRaisesRegex(ValueError, "holdout"):
                pilot.assert_same_holdout(frame, changed)

    def test_pilot_has_equal_budgets_no_head_reset_and_no_test_inference(self) -> None:
        plan = {"arms": {
            "control": {"manifest": "old.csv", "folds": "old_folds.csv"},
            "added_siim": {"manifest": "new.csv", "folds": "new_folds.csv"},
        }}
        with TemporaryDirectory() as temporary, patch.object(pilot, "run_child") as child:
            root = Path(temporary)
            pilot.run_training(root, plan)
        self.assertEqual([call.args[1] for call in child.call_args_list],
                         ["control", "added_siim", "validation"])
        for call in child.call_args_list[:2]:
            command = call.args[2]
            self.assertNotIn("--reset-risk-head", command)
            for flag, value in (("--steps-per-epoch", "1000"), ("--head-epochs", "1"),
                                ("--fine-tune-epochs", "2"), ("--finalization-mode", "none")):
                self.assertEqual(command[command.index(flag) + 1], value)
        self.assertIn("--evaluate", child.call_args_list[-1].args[2])

    def test_resume_uses_existing_run_and_ambiguous_runs_fail(self) -> None:
        plan = {"arms": {"control": {"manifest": "old.csv", "folds": "folds.csv"}}}
        with TemporaryDirectory() as temporary, patch.object(pilot, "run_child") as child:
            root = Path(temporary)
            run = root / "control/run_first"
            run.mkdir(parents=True)
            pilot.run_training(root, plan)
            command = child.call_args_list[0].args[2]
            self.assertEqual(command[command.index("--resume-run-dir") + 1], str(run))
            child.reset_mock()
            (root / "control/run_second").mkdir()
            with self.assertRaisesRegex(ValueError, "Ambiguous"):
                pilot.run_training(root, plan)
            child.assert_not_called()


if __name__ == "__main__":
    unittest.main()
