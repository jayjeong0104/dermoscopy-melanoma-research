from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from training_provenance import (ProvenanceError, assert_same_resume_identities,
    audit_warm_start, portable_path, require_clean_warm_start, resolve_training_metadata_policy)


def identities(prefix: str) -> pd.DataFrame:
    return pd.DataFrame({"source": ["source"], "image_id": [prefix + "_image"],
        "lesion_id": [prefix + "_lesion"], "sha256": [prefix + "_hash"],
        "split_group_id": [prefix + "_patient"]})


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.holdouts = {"validation": identities("validation"), "test": identities("test")}

    def ancestor(self, name="parent", parent=None, train=None, val=None, test=None):
        directory = self.root / name
        directory.mkdir()
        model = directory / "best_model.keras"
        model.write_bytes(b"test checkpoint identity")
        (directory / "run_config.json").write_text(json.dumps(
            {"weights": "imagenet", "warm_start_model": str(parent) if parent else None}))
        for split, frame in (("train", train), ("val", val), ("test", test)):
            if frame is None:
                frame = identities(name + split)
            frame.to_csv(directory / f"{split}_split.csv", index=False)
        return model

    def test_disjoint_supervised_ancestry_is_eligible(self):
        report = audit_warm_start(self.ancestor(), self.holdouts, self.root)
        require_clean_warm_start(report)
        self.assertFalse(report["contaminated"])
        self.assertEqual(report["root_initialization"], "imagenet")
        self.assertEqual(len(report["ancestors"][0]["model_sha256"]), 64)

    def test_each_identity_detects_training_leakage_independently(self):
        for identity in ("image_id", "sha256", "lesion_id", "split_group_id"):
            with self.subTest(identity=identity):
                exposed = identities("otherwise_disjoint")
                exposed[identity] = self.holdouts["test"][identity]
                report = audit_warm_start(self.ancestor(name=identity, train=exposed), self.holdouts, self.root)
                with self.assertRaisesRegex(ProvenanceError, "Resetting only"):
                    require_clean_warm_start(report)

    def test_ancestor_validation_selection_exposure_is_rejected(self):
        report = audit_warm_start(self.ancestor(val=self.holdouts["validation"]), self.holdouts, self.root)
        with self.assertRaises(ProvenanceError):
            require_clean_warm_start(report)

    def test_ancestor_test_reuse_is_reported_separately(self):
        report = audit_warm_start(self.ancestor(test=self.holdouts["test"]), self.holdouts, self.root)
        require_clean_warm_start(report)
        self.assertTrue(report["prior_test_reuse"])

    def test_transitive_exposure_is_rejected(self):
        grandparent = self.ancestor(name="grandparent", train=self.holdouts["test"])
        parent = self.ancestor(parent=grandparent)
        report = audit_warm_start(parent, self.holdouts, self.root)
        self.assertEqual(len(report["ancestors"]), 2)
        with self.assertRaises(ProvenanceError):
            require_clean_warm_start(report)

    def test_missing_ancestor_checkpoint_fails_closed(self):
        parent = self.ancestor(parent=self.root / "missing.keras")
        with self.assertRaisesRegex(ProvenanceError, "Missing ancestor checkpoint"):
            audit_warm_start(parent, self.holdouts, self.root)

    def test_missing_split_or_config_fails_closed(self):
        for name in ("run_config.json", "train_split.csv", "val_split.csv", "test_split.csv"):
            with self.subTest(name=name):
                model = self.ancestor(name=name.replace(".", "_"))
                (model.parent / name).unlink()
                with self.assertRaisesRegex(ProvenanceError, "Missing ancestor"):
                    audit_warm_start(model, self.holdouts, self.root)

    def test_cycle_fails_closed(self):
        model = self.ancestor()
        (model.parent / "run_config.json").write_text(json.dumps({"warm_start_model": str(model)}))
        with self.assertRaisesRegex(ProvenanceError, "Cyclic"):
            audit_warm_start(model, self.holdouts, self.root)

    def test_missing_group_identity_fails_closed(self):
        invalid = identities("bad")
        invalid.loc[0, "split_group_id"] = None
        with self.assertRaisesRegex(ProvenanceError, "missing provenance identities"):
            audit_warm_start(self.ancestor(train=invalid), self.holdouts, self.root)

    def test_image_identity_is_namespaced_by_source(self):
        exposed = identities("otherwise_disjoint")
        exposed["image_id"] = self.holdouts["test"]["image_id"]
        exposed["source"] = "different_source"
        report = audit_warm_start(self.ancestor(train=exposed), self.holdouts, self.root)
        require_clean_warm_start(report)

    def test_portable_windows_and_wsl_paths(self):
        windows = r"D:\data with spaces\run\best.keras"
        self.assertEqual(str(portable_path(windows, self.root, "posix")).replace("\\", "/"),
            "/mnt/d/data with spaces/run/best.keras")
        self.assertEqual(str(portable_path("/mnt/d/data with spaces/run/best.keras", self.root, "nt")), windows)
        self.assertEqual(portable_path("artifacts/run/best.keras", self.root), self.root / "artifacts/run/best.keras")

    def test_metadata_policy_defaults_match_inference_and_preserve_legacy_resume(self):
        self.assertEqual(resolve_training_metadata_policy(None), "unknown")
        self.assertEqual(resolve_training_metadata_policy(None, {}), "provided")
        self.assertEqual(resolve_training_metadata_policy(None, {"training_metadata_policy": "unknown"}), "unknown")
        with self.assertRaisesRegex(ValueError, "cannot change"):
            resolve_training_metadata_policy("unknown", {})

    def test_resume_checks_actual_identities_not_only_counts(self):
        old = identities("same")
        assert_same_resume_identities(old, old, "train")
        changed = old.copy()
        changed["sha256"] = "different_hash"
        with self.assertRaisesRegex(ProvenanceError, "identities changed"):
            assert_same_resume_identities(old, changed, "train")

    def test_legacy_lesion_group_fallback_preserves_same_resume(self):
        old = identities("same").drop(columns="split_group_id")
        new = old.assign(split_group_id=old.lesion_id)
        assert_same_resume_identities(old, new, "train")

    def test_resume_rejects_swapped_labels_with_unchanged_counts(self):
        old = pd.concat([identities("first"), identities("second")], ignore_index=True)
        old = old.assign(target=[0, 1], diagnosis=["NV", "MEL"], diagnosis_index=[5, 4])
        changed = old.assign(target=[1, 0], diagnosis=["MEL", "NV"], diagnosis_index=[4, 5])
        with self.assertRaisesRegex(ProvenanceError, "per-image values changed.*target"):
            assert_same_resume_identities(old, changed, "train")

    def test_resume_rejects_metadata_or_supervision_changes(self):
        old = identities("same").assign(age_approx=50., sex="female", anatom_site_general="back",
                                        diagnosis_label_available=1)
        for column, value in (("age_approx", 60.), ("sex", "male"),
                              ("anatom_site_general", "face"), ("diagnosis_label_available", 0)):
            with self.subTest(column=column), self.assertRaisesRegex(ProvenanceError, "per-image values changed"):
                assert_same_resume_identities(old, old.assign(**{column: value}), "train")

    def test_resume_accepts_equivalent_csv_numeric_types_and_row_order(self):
        old = pd.concat([identities("first"), identities("second")], ignore_index=True)
        old = old.assign(target=[0, 1], age_approx=[None, 50.], diagnosis_label_available=1)
        current = old.iloc[::-1].copy()
        saved = old.astype("string")
        assert_same_resume_identities(saved, current, "train")


if __name__ == "__main__":
    unittest.main()
