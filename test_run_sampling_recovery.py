import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import train_multitask as training
import run_sampling_recovery as driver
from operating_policy import policy_folds


class SamplingDriverTests(unittest.TestCase):
    def test_arms_only_differ_in_sampling_and_output(self):
        with TemporaryDirectory() as directory:
            root, plan = Path(directory), {"manifest": "m", "folds": "f"}
            options = []
            for unit in ("image", "lesion"):
                command = driver.training_command(root, plan, unit, 137)
                with patch("sys.argv", command[1:]):
                    args = vars(training.parse_args())
                self.assertEqual(args.pop("sampling_unit"), unit)
                self.assertTrue(args["freeze_batch_norm"])
                self.assertTrue(args["require_gpu"])
                self.assertEqual(args["finalization_mode"], "none")
                args.pop("output_root")
                options.append(args)
            self.assertEqual(options[0], options[1])

    def test_resume_only_its_own_sampling_arm(self):
        with TemporaryDirectory() as directory:
            root, plan = Path(directory), {"manifest": "m", "folds": "f"}
            (root / "image/frozen_bn_seed137/run_one").mkdir(parents=True)
            self.assertIn("--resume-run-dir", driver.training_command(root, plan, "image", 137))
            self.assertNotIn("--resume-run-dir", driver.training_command(root, plan, "lesion", 137))

    def test_failed_screen_stops_and_success_requires_confirmation(self):
        with patch.object(driver, "run_seed", return_value={"passes": False}) as runs:
            result = driver.run_study(Path("study"), {})
            self.assertEqual(runs.call_count, 1)
            self.assertEqual(result["decision"], "retain_baseline")
        for passes in (False, True):
            with patch.object(driver, "run_seed", side_effect=[{"passes": True}, {"passes": passes}]) as runs:
                result = driver.run_study(Path("study"), {})
                self.assertEqual([call.args[2] for call in runs.call_args_list], [42, 137])
                self.assertEqual(result["both_seeds_pass"], passes)
                self.assertFalse(result["automatic_promotion"])
                self.assertFalse(result["test_inference"])

    def test_reused_control_never_schedules_new_seed42_image_training(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "seed42_validation_report.json").write_text('{"passes": false}')
            with patch.object(driver, "phase") as phase:
                driver.run_seed(root, {"manifest": "m", "folds": "f"}, 42)
            self.assertEqual([call.args[1] for call in phase.call_args_list], ["lesion_seed42", "validation_seed42"])

    def test_artifact_paths_cannot_escape_artifact_root(self):
        with self.assertRaises(ValueError):
            driver.artifact_path("/tmp/artifacts/../../outside.keras")
        self.assertTrue(driver.artifact_path("/mnt/c/project/artifacts/run/model.keras").is_relative_to(driver.ROOT / "artifacts"))

    def test_evaluation_checks_all_epochs_uses_batch4_and_normalizes_integer_width(self):
        frame = pd.DataFrame({"source": ["S"] * 100, "lesion_id": [f"l{i:03d}" for i in range(100)],
            "split_group_id": [f"g{i // 2:03d}" for i in range(100)], "target": np.asarray([0, 1] * 50, dtype="int32")})
        frame["image_id"], frame["sha256"] = frame.lesion_id, "sha_" + frame.lesion_id
        frame["diagnosis"] = np.where(frame.target.eq(1), "MEL", "NV")
        frame["diagnosis_index"] = frame.diagnosis.map(training.DIAGNOSIS_TO_INDEX)
        frame = training.configure_binary_target(frame, "melanoma")
        risk = np.where(frame.target.eq(1), .8, .2).astype("float32")
        diagnosis = np.zeros((100, 8), dtype="float32")
        diagnosis[:, 4], diagnosis[:, 5] = risk, 1 - risk
        reference = frame.assign(target=frame.target.astype("int64"), policy_fold=policy_folds(frame), probability=risk)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            baseline, run = root / "baseline", root / "lesion/frozen_bn_seed42/run_one"
            baseline.mkdir()
            run.mkdir(parents=True)
            frame.to_csv(baseline / "val_split.csv", index=False)
            config = {"sampling_unit": "lesion", "seed": 42, "freeze_batch_norm": True}
            (run / "run_config.json").write_text(json.dumps(config))
            (run / "metadata_schema.json").write_text(json.dumps({"age_median": 50, "feature_names": ["fixture"]}))
            for name in driver.EPOCHS:
                (run / name).write_bytes(b"fixture")
            plan = {"tensorflow_version": training.tf.__version__, "required_control_config": {"freeze_batch_norm": True}}
            with patch.object(driver, "BASELINE", baseline), patch.object(driver, "verify_plan", return_value=plan), \
                 patch.object(driver, "recorded_predictions", return_value=reference), \
                 patch("train_multitask.configure_runtime", return_value={"gpu_devices": ["fixture"]}), \
                 patch("finalize_multitask_run.load_run_split", return_value=frame), \
                 patch("finalize_multitask_run.predict_checkpoint", return_value=(risk, diagnosis)) as predictor:
                result = driver.evaluate(root, 42)
            self.assertEqual(predictor.call_count, 3)
            self.assertTrue(all(call.args[4] == 4 for call in predictor.call_args_list))
            self.assertEqual(len(result["results"]), 7)
            self.assertFalse(result["passes"])
            self.assertTrue((root / "seed42_validation_report.json").is_file())


if __name__ == "__main__":
    unittest.main()
