import unittest

import numpy as np
import pandas as pd

from high_recall_selection import evaluate_training_rank_quality
from run_clean_panderm_review import selection_key


class HighRecallSelectionTests(unittest.TestCase):
    def frame(self):
        records = []
        for source in ("A", "B"):
            for fold in range(2):
                for row in range(40):
                    target = row < 20
                    records.append({"source": source, "inner_fold": fold,
                        "lesion_id": f"{source}_{fold}_{row}", "split_group_id": f"{source}_{fold}_{row}",
                        "target": int(target), "useful": .8 if target else .2, "constant": .5})
        return pd.DataFrame(records)

    def test_recall_first_fallback_prefers_constant_but_rank_quality_rejects_it(self):
        constant = {"by_source": {"A": {"recall": 1., "specificity": 0.}}, "pr_auc": .1}
        useful = {"by_source": {"A": {"recall": .94, "specificity": .8}}, "pr_auc": .8}
        self.assertGreater(selection_key(constant), selection_key(useful))
        result = evaluate_training_rank_quality(self.frame(), ("constant", "useful"))
        self.assertEqual(result["selected_candidate"], "useful")
        self.assertEqual(result["candidates"]["constant"]["macro_specificity_at_target"], 0.)
        self.assertEqual(result["candidates"]["useful"]["macro_specificity_at_target"], 1.)

    def test_all_uninformative_candidates_produce_no_selection(self):
        result = evaluate_training_rank_quality(self.frame(), ("constant",))
        self.assertIsNone(result["selected_candidate"])

    def test_source_offsets_preserve_within_source_ranking_quality(self):
        frame = self.frame()
        frame["shifted"] = np.where(frame.source.eq("A"), frame.useful * .1, .6 + frame.useful * .3)
        result = evaluate_training_rank_quality(frame, ("useful", "shifted"))
        self.assertEqual(result["candidates"]["useful"]["macro_specificity_at_target"],
                         result["candidates"]["shifted"]["macro_specificity_at_target"])
        for candidate in result["candidates"].values():
            self.assertTrue(all(cell["recall"] >= .95 for cell in candidate["source_fold_cells"]))

    def test_missing_class_or_crossing_group_fails_closed(self):
        frame = self.frame()
        frame.loc[(frame.source == "A") & (frame.inner_fold == 0), "target"] = 0
        with self.assertRaisesRegex(ValueError, "both target classes"):
            evaluate_training_rank_quality(frame, ("useful",))
        frame = self.frame()
        frame.loc[[0, 40], "split_group_id"] = "shared"
        with self.assertRaisesRegex(ValueError, "crosses"):
            evaluate_training_rank_quality(frame, ("useful",))


if __name__ == "__main__":
    unittest.main()
