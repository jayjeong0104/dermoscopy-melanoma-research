"""Group-separated development threshold evaluation; never fits on a scoring fold."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from compare_model_runs import operating_metrics

POLICY_VERSION = "fixed_blend_group_crossfit_v1"
DIRECT_WEIGHT = 0.5
FOLD_SEED = 20260907


def validate_lesions(frame: pd.DataFrame) -> None:
    required = ["source", "lesion_id", "split_group_id", "target"]
    if frame[required].isna().any().any() or frame.duplicated(["source", "lesion_id"]).any():
        raise ValueError("Lesion identities must be unique and nonmissing.")
    if set(frame.target.unique()) != {0, 1}:
        raise ValueError("Both binary classes are required.")


def policy_folds(frame: pd.DataFrame, count: int = 5) -> np.ndarray:
    validate_lesions(frame)
    ordered = frame.reset_index(drop=True).sort_values(["source", "lesion_id"])
    strata = ordered.source.astype(str) + "::" + ordered.target.astype(str)
    splitter = StratifiedGroupKFold(n_splits=count, shuffle=True, random_state=FOLD_SEED)
    result = np.full(len(frame), -1, dtype=int)
    for fold, (_, held) in enumerate(splitter.split(ordered, strata, ordered.split_group_id)):
        result[ordered.index.to_numpy()[held]] = fold
    validate_folds(frame, result)
    return result


def validate_folds(frame: pd.DataFrame, folds: np.ndarray) -> None:
    validate_lesions(frame)
    folds = np.asarray(folds)
    if folds.shape != (len(frame),) or not np.issubdtype(folds.dtype, np.integer) or (folds < 0).any():
        raise ValueError("Invalid operating-policy folds.")
    assigned = frame.assign(policy_fold=folds)
    if assigned.groupby("split_group_id").policy_fold.nunique().gt(1).any():
        raise ValueError("A split group crosses calibration and scoring folds.")
    if len(np.unique(folds)) < 2:
        raise ValueError("At least two operating-policy folds are required.")
    for fold in np.unique(folds):
        if set(frame.loc[folds == fold, "target"]) != {0, 1}:
            raise ValueError("Every scoring fold must contain both classes.")
        if set(frame.loc[folds != fold, "target"]) != {0, 1}:
            raise ValueError("Every calibration partition must contain both classes.")


def decision_metrics(truth: np.ndarray, decisions: np.ndarray) -> dict:
    truth, decisions = np.asarray(truth), np.asarray(decisions, dtype=bool)
    if truth.ndim != 1 or truth.shape != decisions.shape or set(np.unique(truth)) != {0, 1}:
        raise ValueError("Decision metrics need aligned binary labels with both classes.")
    tp = int(np.sum((truth == 1) & decisions))
    fp = int(np.sum((truth == 0) & decisions))
    fn = int(np.sum((truth == 1) & ~decisions))
    tn = int(np.sum((truth == 0) & ~decisions))
    return {"recall": tp / (tp + fn), "specificity": tn / (tn + fp),
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "accuracy": (tp + tn) / len(truth), "false_positives": fp,
            "false_negatives": fn, "true_positives": tp, "true_negatives": tn}


def crossfit_policy(frame: pd.DataFrame, probabilities: np.ndarray,
                    minimum_recall: float = 0.95, folds: np.ndarray | None = None) -> tuple[dict, pd.DataFrame]:
    folds = policy_folds(frame) if folds is None else np.asarray(folds)
    validate_folds(frame, folds)
    probabilities = np.asarray(probabilities, dtype=float)
    truth = frame.target.to_numpy(dtype=int)
    # Validate the entire score vector without using its fitted threshold for scoring.
    full = operating_metrics(truth, probabilities, minimum_recall)
    decisions, thresholds = np.zeros(len(frame), dtype=bool), np.zeros(len(frame))
    fold_reports = []
    for fold in np.unique(folds):
        scoring, calibration = folds == fold, folds != fold
        fitted = operating_metrics(truth[calibration], probabilities[calibration], minimum_recall)
        thresholds[scoring] = fitted["threshold"]
        decisions[scoring] = probabilities[scoring] >= fitted["threshold"]
        fold_reports.append({"fold": int(fold), "threshold": fitted["threshold"],
                             "calibration_lesions": int(calibration.sum()),
                             "scoring_lesions": int(scoring.sum()),
                             **decision_metrics(truth[scoring], decisions[scoring])})
    result = {"policy_version": POLICY_VERSION, "minimum_calibration_recall": minimum_recall,
              **decision_metrics(truth, decisions),
              "pr_auc": float(average_precision_score(truth, probabilities)),
              "roc_auc": float(roc_auc_score(truth, probabilities)), "folds": fold_reports,
              "same_data_threshold_metrics_for_diagnosis_only": full,
              "limitation": "Development cross-fitting separates threshold calibration from scoring groups, not model development from validation. Repeatedly inspected validation is not an independent test. The requested calibration sensitivity is not guaranteed on scoring folds."}
    output = frame[["source", "lesion_id", "split_group_id", "target"]].copy()
    output["probability"], output["policy_fold"] = probabilities, folds
    output["threshold"], output["prediction"] = thresholds, decisions.astype(int)
    return result, output


def selection_score(metrics: dict, minimum_recall: float = 0.95) -> float:
    # Infeasible candidates cannot outrank any candidate meeting the recall constraint.
    if metrics["recall"] < minimum_recall:
        return float(metrics["recall"] - 1.0)
    return float(metrics["specificity"])


def promotion_gate(baseline: dict, candidate: dict, minimum_gain: float = 0.01) -> dict:
    criteria = {"crossfit_sensitivity_at_least_95_percent": candidate["recall"] >= 0.95,
                "sensitivity_does_not_decrease": candidate["recall"] >= baseline["recall"],
                "specificity_gain_at_least_one_point": candidate["specificity"] >= baseline["specificity"] + minimum_gain,
                "average_precision_does_not_decrease": candidate["pr_auc"] >= baseline["pr_auc"],
                "roc_auc_does_not_decrease": candidate["roc_auc"] >= baseline["roc_auc"]}
    return {"criteria": criteria, "passes": all(criteria.values()), "automatic_promotion": False,
            "limitation": "A predeclared development screen, not independent confirmation or clinical approval. The older AP-improvement experiment remains failed under its original criteria."}


def paired_decision_intervals(reference: pd.DataFrame, candidate: pd.DataFrame,
                              iterations: int = 1000, seed: int = FOLD_SEED) -> dict:
    keys = ["source", "lesion_id", "split_group_id", "target", "policy_fold"]
    if not reference[keys].reset_index(drop=True).equals(candidate[keys].reset_index(drop=True)):
        raise ValueError("Paired comparison identities, labels, or policy folds differ.")
    if iterations < 100:
        raise ValueError("Use at least 100 bootstrap iterations.")
    codes, names = pd.factorize(reference.split_group_id)
    if (codes < 0).any():
        raise ValueError("Missing uncertainty-analysis groups.")
    members = [np.flatnonzero(codes == i) for i in range(len(names))]
    truth = reference.target.to_numpy()
    old, new = reference.prediction.to_numpy(dtype=bool), candidate.prediction.to_numpy(dtype=bool)
    rng, deltas = np.random.default_rng(seed), []
    for _ in range(iterations):
        sample = np.concatenate([members[i] for i in rng.integers(0, len(members), len(members))])
        y = truth[sample]
        if set(np.unique(y)) != {0, 1}:
            continue
        deltas.append([float(np.mean(~new[sample][y == 0]) - np.mean(~old[sample][y == 0])),
                       float(np.mean(new[sample][y == 1]) - np.mean(old[sample][y == 1]))])
    if len(deltas) < 0.8 * iterations:
        raise ValueError("Too few valid bootstrap samples.")
    bounds = np.quantile(deltas, [0.025, 0.975], axis=0)
    return {"iterations": len(deltas), "split_groups": len(names),
            "delta_intervals_95": {key: {"lower": float(bounds[0, i]), "upper": float(bounds[1, i])}
                                   for i, key in enumerate(("specificity", "recall"))},
            "limitation": "Conditional paired group bootstrap of fixed cross-fitted decisions. Does not account for epoch selection, threshold refitting, or repeated development-set inspection; not an independent significance test."}
