from __future__ import annotations

import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd

import assess_market_readiness as readiness


class ReadinessEvidenceTests(unittest.TestCase):
    def gate(self, report, name):
        return next(item for item in report["gates"] if item["id"] == name)

    def test_missing_manifest_and_unknown_source_do_not_pass_rights(self):
        with TemporaryDirectory(dir=Path(__file__).parent) as directory:
            run = Path(directory)
            report = readiness.build_report(run, run / "missing.csv")
            self.assertEqual(self.gate(report, "commercial_data_rights")["status"], "missing")
            pd.DataFrame({"source": ["unreviewed_source"]}).to_csv(run / "train_split.csv", index=False)
            report = readiness.build_report(run, run / "missing.csv")
            self.assertEqual(self.gate(report, "commercial_data_rights")["status"], "missing")

    def test_saved_training_sources_override_unrelated_supplied_manifest(self):
        with TemporaryDirectory(dir=Path(__file__).parent) as directory:
            run = Path(directory)
            pd.DataFrame({"source": ["HAM10000"]}).to_csv(run / "train_split.csv", index=False)
            pd.DataFrame({"source": ["BCN20000"]}).to_csv(run / "other.csv", index=False)
            report = readiness.build_report(run, run / "other.csv")
            self.assertEqual(report["training_sources"], {"HAM10000": 1})
            self.assertEqual(self.gate(report, "commercial_data_rights")["status"], "failed")

    def test_conditional_license_is_not_verified_permission(self):
        with TemporaryDirectory(dir=Path(__file__).parent) as directory:
            run = Path(directory)
            pd.DataFrame({"source": ["BCN20000"]}).to_csv(run / "train_split.csv", index=False)
            report = readiness.build_report(run, run / "missing.csv")
            self.assertEqual(self.gate(report, "commercial_data_rights")["status"], "manual_review_required")

    def test_report_presence_does_not_certify_independence_or_subgroup_validation(self):
        with TemporaryDirectory(dir=Path(__file__).parent) as directory:
            run = Path(directory)
            (run / "finalization_status.json").write_text(json.dumps({"status": "complete", "tta_transforms": 8, "completed_stages": ["split_integrity"]}))
            (run / "derm12345_external_metrics.json").write_text(json.dumps({
                "development_status": "external_development_cohort_previously_inspected_not_fresh_confirmation",
                "exact_training_hash_overlap": 0, "inference_binding": {"checkpoint_hash_verified": True}}))
            (run / "test_subgroup_metrics.json").write_text(json.dumps({"missingness": {"skin_tone": 0, "capture_device": 0}}))
            report = readiness.build_report(run, run / "missing.csv")
            for name in ("independent_external_test", "patient_lesion_hash_separation", "representative_subgroup_validation"):
                self.assertEqual(self.gate(report, name)["status"], "manual_review_required")
            self.assertTrue(report["release_blocked"])
            readiness.write_model_card(run / "card.md", run)
            card = (run / "card.md").read_text()
            self.assertNotIn("Splits are separated by lesion, patient/split group", card)
            self.assertIn("independence unverified", card)
            self.assertIn("Test-time views: 8", card)
            self.assertNotIn("Output: calibrated research probability", card)

    def test_nonfinite_or_impossible_metrics_cannot_pass(self):
        for value in (float("inf"), float("nan"), 1.5, -.1):
            gates = readiness.metric_gates("test", "Test", {name: value for name in readiness.PERFORMANCE_TARGETS})
            self.assertTrue(all(item["status"] == "failed" for item in gates))


class RepairRegistrationTests(unittest.TestCase):
    def test_arbitrary_registered_diff_does_not_claim_protocol_equivalence(self):
        import register_foundation_io_repair as repair
        import run_foundation_experiment as experiment_driver

        with TemporaryDirectory(dir=Path(__file__).parent) as directory:
            project = Path(directory)
            experiment = project / "experiment"
            (project / "artifacts").mkdir()
            (experiment / "code_snapshot").mkdir(parents=True)
            before = {}
            for name in repair.ORIGINAL:
                snapshot = experiment / "code_snapshot" / name
                snapshot.write_text("objective = 'original'\n")
                before[name] = repair.file_hash(snapshot)
                (project / name).write_text("objective = 'arbitrary different objective'\n")
            (project / "artifact_io.py").write_text("# synthetic I/O helper\n")
            (experiment / "experiment_plan.json").write_text(json.dumps({"code_sha256": before, "input_sha256": {}}))
            for name in ("status.json", "driver.log", "train_panderm_base.log"):
                (experiment / name).write_text("{}")
            fake_fcntl = SimpleNamespace(LOCK_EX=1, LOCK_NB=2, flock=lambda *_: None)
            with patch.object(repair, "ROOT", project), patch.object(repair, "ORIGINAL", before), \
                    patch.object(repair, "assert_idle"), patch.object(experiment_driver, "ROOT", project), \
                    patch.dict("sys.modules", {"fcntl": fake_fcntl}):
                repair.register(experiment)
            record = json.loads((experiment / "io_repair.json").read_text())
            self.assertIsNone(record["protocol_changed"])
            self.assertEqual(record["protocol_review_status"], "unverified_by_registration")


class SmokeUpdateTests(unittest.TestCase):
    def test_smoke_executes_real_accumulated_update_with_tiny_backbone(self):
        import tensorflow as tf
        import smoke_test_multitask_v2 as smoke
        def tiny_backbone(**kwargs):
            inputs = tf.keras.Input(kwargs["input_shape"])
            return tf.keras.Model(inputs, tf.keras.layers.Conv2D(4, 1)(inputs))

        previous = tf.keras.mixed_precision.global_policy()
        try:
            with patch.object(smoke.training, "backbone_builder", return_value=tiny_backbone), \
                    patch("tempfile.tempdir", str(Path(__file__).parent)):
                smoke.main()
        finally:
            tf.keras.mixed_precision.set_global_policy(previous)


if __name__ == "__main__":
    unittest.main()
