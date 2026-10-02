import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from compare_model_runs import file_hash
from run_targeted_recovery import phase, training_command, verify_plan


class TargetedDriverTests(unittest.TestCase):
    def test_paired_commands_differ_only_by_policy_path_and_weights(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            plan = {"manifest": "manifest.csv", "folds": "folds.csv"}
            def options(command):
                self.assertEqual(command[-1], "--require-gpu")
                command = command[:-1]
                return dict(zip(command[2::2], command[3::2]))
            control = options(training_command(root, plan, "control", 42))
            hard = options(training_command(root, plan, "hard_negative", 42))
            self.assertNotIn("--hard-negative-weights", control)
            hard.pop("--hard-negative-weights")
            control.pop("--output-root")
            hard.pop("--output-root")
            self.assertEqual(control, hard)
            self.assertEqual(control["--checkpoint-metric"], "lesion_pr_auc")
            self.assertEqual(control["--finalization-mode"], "none")

    def test_ambiguous_run_is_rejected(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("run_one", "run_two"):
                (root / "control_seed42" / name).mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "Ambiguous"):
                training_command(root, {"manifest": "m", "folds": "f"}, "control", 42)

    def test_completed_phase_is_idempotent_and_corruption_is_rejected(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "weights.csv"
            output.write_text("fixture")
            (root / "phase_done").mkdir()
            (root / "phase_done/mining.json").write_text(json.dumps({
                "command": ["python", "fixture"], "outputs": {"weights.csv": file_hash(output)}}))
            with patch("run_targeted_recovery.verify_plan"), patch("run_targeted_recovery.run_child") as child:
                phase(root, "mining", ["python", "fixture"], ["weights.csv"])
                child.assert_not_called()
                output.write_text("changed")
                with self.assertRaisesRegex(ValueError, "changed"):
                    phase(root, "mining", ["python", "fixture"], ["weights.csv"])

    def test_changed_code_blocks_resumption(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            code = root / "train.py"
            code.write_text("original")
            (root / "experiment_plan.json").write_text(json.dumps({
                "code_sha256": {"train.py": file_hash(code)}, "input_sha256": {}}))
            code.write_text("changed")
            with patch("run_targeted_recovery.ROOT", root), self.assertRaisesRegex(ValueError, "Code changed"):
                verify_plan(root)


if __name__ == "__main__":
    unittest.main()
