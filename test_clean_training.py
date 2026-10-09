"""Regression tests for isolation of the new ImageNet training reference."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_clean_training as clean


class CleanTrainingTests(unittest.TestCase):
    def test_recipe_cannot_inherit_project_checkpoint_or_run_test_inference(self):
        args = clean.training_arguments(Path("output"))
        self.assertNotIn("--warm-start-model", args)
        self.assertEqual(args[args.index("--weights") + 1], "imagenet")
        self.assertEqual(args[args.index("--finalization-mode") + 1], "none")
        self.assertEqual(args[args.index("--training-metadata-policy") + 1], "unknown")
        self.assertNotIn("--steps-per-epoch", args)
        self.assertNotIn("--sampling-mass-unit", args)
        self.assertEqual(args[args.index("--sampling-alpha") + 1], "0.5")
        self.assertEqual(clean.resolve_output(), clean.DEFAULT_OUTPUT.resolve())

    def test_natural_lesion_changes_only_mass_unit_and_exponent(self):
        reference = clean.training_arguments(Path("output"))
        candidate = clean.training_arguments(Path("output"), "natural_lesion")
        self.assertEqual(candidate[-2:], ["--sampling-mass-unit", "lesion"])
        candidate = candidate[:-2]
        self.assertEqual(candidate[candidate.index("--sampling-alpha") + 1], "1")
        candidate[candidate.index("--sampling-alpha") + 1] = "0.5"
        self.assertEqual(candidate, reference)
        for flag, value in (("--positive-sampling-fraction", "0.25"), ("--head-epochs", "4"),
                            ("--fine-tune-epochs", "12"), ("--weights", "imagenet"),
                            ("--finalization-mode", "none")):
            self.assertEqual(candidate[candidate.index(flag) + 1], value)

    def test_natural_lesion_default_output_is_isolated_and_reference_is_forbidden(self):
        self.assertEqual(clean.resolve_output(recipe="natural_lesion"), clean.NATURAL_LESION_OUTPUT.resolve())
        for output in (clean.DEFAULT_OUTPUT, clean.DEFAULT_OUTPUT / "nested"):
            with self.assertRaisesRegex(ValueError, "separate experiment"):
                clean.resolve_output(output, "natural_lesion")
        with patch("sys.argv", ["run_clean_training.py", "--recipe", "natural_lesion", "--prepare-only"]), \
                patch.object(clean, "prepare") as prepare:
            clean.main()
        prepare.assert_called_once_with(clean.NATURAL_LESION_OUTPUT.resolve(), "natural_lesion")

    def test_reference_recipe_remains_cli_default(self):
        with patch("sys.argv", ["run_clean_training.py", "--prepare-only"]), \
                patch.object(clean, "prepare") as prepare:
            clean.main()
        prepare.assert_called_once_with(clean.DEFAULT_OUTPUT.resolve(), "reference")

    def test_recipe_change_refuses_to_replace_existing_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = [root / name for name in ("manifest.csv", "folds.csv", "weights.h5")]
            for path in files:
                path.write_text("original")
            output = root / "experiment"
            with patch.multiple(clean, MANIFEST=files[0], FOLDS=files[1], WEIGHTS=files[2], CODE=()):
                plan = clean.prepare(output, "natural_lesion")
                self.assertEqual(plan["recipe"], "natural_lesion")
                self.assertEqual(plan["comparison"]["fixed_positive_sampling_fraction"], .25)
                before = (output / "experiment_plan.json").read_bytes()
                with self.assertRaisesRegex(ValueError, "including recipe"):
                    clean.prepare(output, "reference")
                self.assertEqual((output / "experiment_plan.json").read_bytes(), before)

    def test_prepare_refuses_changed_input_and_preserves_original_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = [root / name for name in ("manifest.csv", "folds.csv", "weights.h5")]
            for path in files:
                path.write_text("original")
            output = root / "experiment"
            with patch.multiple(clean, MANIFEST=files[0], FOLDS=files[1], WEIGHTS=files[2], CODE=()):
                clean.prepare(output)
                before = (output / "experiment_plan.json").read_bytes()
                files[0].write_text("changed labels")
                with self.assertRaisesRegex(ValueError, "inputs or code changed"):
                    clean.prepare(output)
                self.assertEqual((output / "experiment_plan.json").read_bytes(), before)

    def test_completed_run_detects_modified_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            model = output / "model.keras"
            model.write_bytes(b"weights")
            (output / "complete.json").write_text(json.dumps({"outputs_sha256": {
                model.name: clean.file_hash(model)}}))
            model.write_bytes(b"modified")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                clean.run(output, {})


if __name__ == "__main__":
    unittest.main()
