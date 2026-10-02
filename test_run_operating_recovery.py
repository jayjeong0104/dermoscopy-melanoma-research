from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import json
import numpy as np

import train_multitask as training
from run_operating_recovery import EPOCHS, choose_checkpoint, evaluate, run_study, training_command
from operating_policy import policy_folds
import test_operating_policy as policy_tests


class OperatingDriverTests(unittest.TestCase):
    def test_evaluator_reads_every_epoch_and_uses_identical_batch_protocol(self):
        frame = policy_tests.OperatingPolicyTests().frame()
        frame["image_id"] = frame.lesion_id
        frame["sha256"] = "hash_" + frame.lesion_id
        frame["diagnosis"] = np.where(frame.target.eq(1), "MEL", "NV")
        frame["diagnosis_index"] = frame.diagnosis.map(training.DIAGNOSIS_TO_INDEX)
        frame = training.configure_binary_target(frame, "melanoma")
        risk = np.where(frame.target.eq(1), .8, .2).astype("float32")
        diagnosis = np.zeros((len(frame), 8), dtype="float32")
        diagnosis[:, 4], diagnosis[:, 5] = risk, 1 - risk
        with TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline_run"
            for name in ("baseline_run", "control_seed42/run_one", "frozen_bn_seed42/run_one"):
                run = root / name
                run.mkdir(parents=True)
                (run / "run_config.json").write_text(json.dumps({"binary_target": "melanoma", "image_size": 300,
                    "binary_inference_metadata": "unknown", "freeze_batch_norm": name.startswith("frozen_bn"),
                    "checkpoint_metric": "lesion_operating", "archive_epoch_checkpoints": True}))
                (run / "metadata_schema.json").write_text(json.dumps({"age_median": 50.0, "feature_names": ["fixture"]}))
                for filename in (("best_model.keras",) if name == "baseline_run" else EPOCHS):
                    (run / filename).write_bytes(b"fixture")
            frame[["source", "lesion_id", "split_group_id", "target"]].assign(policy_fold=policy_folds(frame)).to_csv(
                root / "development_policy_folds.csv", index=False)
            with patch("run_operating_recovery.BASELINE", baseline), \
                 patch("run_operating_recovery.verify_plan", return_value={"limitations": "fixture"}), \
                 patch("train_multitask.configure_runtime", return_value={"gpu_devices": ["fixture"]}), \
                 patch("finalize_multitask_run.load_run_split", return_value=frame), \
                 patch("finalize_multitask_run.predict_checkpoint", return_value=(risk, diagnosis)) as predictor:
                result = evaluate(root, 42)
            self.assertEqual(predictor.call_count, 7)
            self.assertTrue(all(call.args[4] == 4 for call in predictor.call_args_list))
            self.assertFalse(result["passes"])
            self.assertEqual(len(result["results"]), 7)
            self.assertTrue((root / "seed42_validation_report.json").is_file())

    def test_only_normalization_and_output_paths_differ_between_arms(self):
        with TemporaryDirectory() as directory:
            root, plan = Path(directory), {"manifest": "m.csv", "folds": "f.csv"}
            options = []
            for policy in ("control", "frozen_bn"):
                command = training_command(root, plan, policy, 42)
                with patch("sys.argv", command[1:]):
                    args = vars(training.parse_args())
                self.assertEqual(args["checkpoint_metric"], "lesion_operating")
                self.assertTrue(args["archive_epoch_checkpoints"])
                self.assertTrue(args["require_gpu"])
                self.assertEqual(args["finalization_mode"], "none")
                self.assertEqual(args.pop("freeze_batch_norm"), policy == "frozen_bn")
                args.pop("output_root")
                options.append(args)
            self.assertEqual(options[0], options[1])

    def test_completed_control_does_not_resume_into_frozen_arm(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "control_seed42/run_one").mkdir(parents=True)
            plan = {"manifest": "m.csv", "folds": "f.csv"}
            self.assertIn("--resume-run-dir", training_command(root, plan, "control", 42))
            self.assertNotIn("--resume-run-dir", training_command(root, plan, "frozen_bn", 42))

    def test_failed_screening_stops_without_second_seed(self):
        with patch("run_operating_recovery.run_seed", return_value={"passes": False}) as child:
            result = run_study(Path("study"), {"limitations": "development only"})
            self.assertEqual(child.call_count, 1)
            self.assertEqual(result["decision"], "retain_baseline")
            self.assertIsNone(result["confirmation_report"])

    def test_success_requires_both_seeds_and_never_promotes(self):
        for confirmation in (False, True):
            with patch("run_operating_recovery.run_seed", side_effect=[{"passes": True}, {"passes": confirmation}]) as child:
                result = run_study(Path("study"), {"limitations": "development only"})
                self.assertEqual([call.args[2] for call in child.call_args_list], [42, 137])
                self.assertEqual(result["both_seeds_pass"], confirmation)
                self.assertFalse(result["automatic_promotion"])

    def test_feasible_operating_epoch_is_not_discarded_for_tiny_ap_gain(self):
        baseline = {"recall": .95, "specificity": .7, "pr_auc": .56, "roc_auc": .93}
        results = {"useful": {"metrics": {**baseline, "specificity": .72}},
                   "ap_tiny_gain": {"metrics": {**baseline, "specificity": .68, "pr_auc": .5602}}}
        self.assertEqual(choose_checkpoint(list(results), results, baseline), "useful")


if __name__ == "__main__":
    unittest.main()
