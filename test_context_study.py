from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from PIL import Image

from context_study_common import full_frame_view
from converged_probe import fit, save_parameters
from export_converged_head import export
from run_context_study import grouped_folds


class ContextStudyTests(unittest.TestCase):
    def test_wide_image_keeps_all_corners_without_distortion(self):
        pixels = np.zeros((200, 400, 3), dtype="uint8")
        pixels[:100, :200] = [255, 0, 0]
        pixels[:100, 200:] = [0, 255, 0]
        pixels[100:, :200] = [0, 0, 255]
        pixels[100:, 200:] = [255, 255, 255]
        view = np.asarray(full_frame_view(Image.fromarray(pixels)))
        np.testing.assert_array_equal(view[56, 0], [255, 0, 0])
        np.testing.assert_array_equal(view[56, 223], [0, 255, 0])
        np.testing.assert_array_equal(view[167, 0], [0, 0, 255])
        np.testing.assert_array_equal(view[167, 223], [255, 255, 255])
        np.testing.assert_array_equal(view[0, 0], [124, 116, 104])

    def test_square_224_image_is_unchanged(self):
        pixels = np.random.default_rng(2).integers(0, 256, (224, 224, 3), dtype="uint8")
        np.testing.assert_array_equal(full_frame_view(Image.fromarray(pixels)), pixels)

    def test_tall_and_thin_images_are_valid(self):
        for size in ((200, 400), (1, 1000), (1000, 1), (1, 1)):
            self.assertEqual(full_frame_view(Image.new("RGB", size)).size, (224, 224))

    def test_inner_folds_are_grouped_deterministic_and_cover_all_training_rows(self):
        frame = pd.DataFrame({"source": ["A"] * 120, "target": [(i // 2) % 2 for i in range(120)],
                              "split_group_id": [f"p{i // 2}" for i in range(120)]})
        folds = grouped_folds(frame)
        np.testing.assert_array_equal(folds, grouped_folds(frame))
        self.assertEqual(set(folds), {0, 1, 2})
        self.assertEqual(frame.assign(fold=folds).groupby("split_group_id").fold.nunique().max(), 1)

    def test_tensorflow_export_roundtrip_matches_converged_classifier(self):
        rng = np.random.default_rng(5)
        x = rng.normal(size=(80, 6)).astype("float32")
        y = (x[:, 0] > 0).astype(int)
        parameters, _ = fit(x, y, .01)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_parameters(root / "parameters.npz", parameters)
            report = export(root / "parameters.npz", root / "head.keras", x)
            self.assertLess(report["maximum_probability_difference"], 1e-8)
            self.assertFalse(report["automatic_promotion"])


if __name__ == "__main__":
    unittest.main()
