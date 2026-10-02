"""Reproducible CPU-only audit of the limited, already explored ensemble recipes."""
import argparse
import json
from pathlib import Path

import numpy as np

from analyze_recovery_errors import paired_bootstrap
from compare_model_runs import file_hash, write_report
from operating_policy import crossfit_policy
from run_sampling_recovery import PREVIOUS, KEYS, recorded_predictions


RECIPES = {"baseline": [1, 0, 0, 0], "frozen_epoch_average": [0, 0, .5, .5],
           "baseline_frozen_average": [.5, 0, .25, .25],
           "control_frozen_average": [0, .5, .25, .25],
           "three_branch_average": [1/3, 1/3, 1/6, 1/6]}
NAMES = ["baseline", "control_seed42_epoch_finetune_02", "frozen_bn_seed42_epoch_finetune_01",
         "frozen_bn_seed42_epoch_finetune_02"]


def audit() -> dict:
    frames = [recorded_predictions(name) for name in NAMES]
    if not all(frame[KEYS].equals(frames[0][KEYS]) for frame in frames):
        raise ValueError("Ensemble prediction identities do not match.")
    scores = np.stack([frame.probability.to_numpy() for frame in frames])
    result = {}
    for name, weights in RECIPES.items():
        metrics, _ = crossfit_policy(frames[0], np.asarray(weights) @ scores,
                                    folds=frames[0].policy_fold.to_numpy())
        result[name] = {"weights": weights, "metrics": metrics}
    uncertainty = paired_bootstrap(frames[0].target.to_numpy(), scores[0], scores[2],
        frames[0].split_group_id.to_numpy(), .5, .5, iterations=1000, seed=20260908)
    # Threshold .5 is not the deployment policy; only ranking intervals are relevant here.
    uncertainty["delta_intervals_95"].pop("specificity")
    return {"members": NAMES, "recipes": result, "frozen_first_finetune_ranking_uncertainty": uncertainty,
        "inputs_sha256": {name: file_hash(PREVIOUS / f"{name}_crossfit_predictions.csv") for name in NAMES},
        "decision": "do_not_promote_explored_ensembles",
        "test_inference": False, "automatic_promotion": False,
        "limitations": "Retrospective exploratory validation. These recipes were inspected before this audit script was saved, not prospectively registered. No ensemble met 95% cross-fitted sensitivity; no new trained weights. Ranking uncertainty is conditional on prior epoch selection and repeated development use."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = audit()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output_dir / "ensemble_audit.json"
    write_report(destination, result)
    print(json.dumps({"report": str(destination), "decision": result["decision"],
                      "ranking_uncertainty": result["frozen_first_finetune_ranking_uncertainty"]}, indent=2))
