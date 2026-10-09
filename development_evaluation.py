"""CPU-only, group-disjoint development evaluation of cached lesion scores.

This module reads no files and performs no model inference or model promotion.
Cross-fitting does not undo selection of a checkpoint on the same development set.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

VERSION = "development_blend_calibration_crossfit_v1"
IDENTITIES = ("source", "lesion_id", "split_group_id", "target")
LIMITATION = (
    "Reused development data: checkpoint/architecture selection already used this cohort. "
    "Scoring groups are excluded from each fold's blend and threshold fitting, but this does "
    "not make the checkpoint independently selected or establish generalization. Fixed "
    "candidates are reported separately; selecting a winner from these results introduces "
    "further development selection. Blend and threshold share the calibration partition. "
    "Scores are not clinically calibrated probabilities. No test or HIBA data are accessed."
)


def _validate(frame, score_columns):
    required = list(IDENTITIES) + list(score_columns)
    missing = set(required) - set(frame)
    if missing:
        raise ValueError(f"Missing lesion evaluation columns: {sorted(missing)}")
    if frame[required].isna().any().any() or frame.duplicated(["source", "lesion_id"]).any():
        raise ValueError("Lesion identities/scores must be nonmissing and lesions unique.")
    if set(frame.target.unique()) != {0, 1}:
        raise ValueError("The development cohort needs both binary target classes.")
    for column in score_columns:
        values = frame[column].to_numpy(dtype=float)
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError(f"{column} must contain finite probabilities in [0, 1].")


def grouped_folds(frame, n_splits=5, seed=20261006):
    if not isinstance(n_splits, int) or n_splits < 2 or frame.split_group_id.nunique() < n_splits:
        raise ValueError("Need at least two folds and enough independent split groups.")
    # Stable assignment even if a cached dataframe arrives in a different order.
    ordered = frame.reset_index(drop=True).sort_values(["source", "lesion_id"])
    strata = ordered.source.astype(str) + "::" + ordered.target.astype(str)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = np.full(len(frame), -1, dtype=int)
    for fold, (_, scored) in enumerate(splitter.split(ordered, strata, ordered.split_group_id)):
        folds[ordered.index.to_numpy()[scored]] = fold
    validate_folds(frame, folds)
    return folds


def validate_folds(frame, folds):
    folds = np.asarray(folds)
    if folds.shape != (len(frame),) or not np.issubdtype(folds.dtype, np.integer) or (folds < 0).any():
        raise ValueError("Invalid aligned fold assignments.")
    if len(np.unique(folds)) < 2:
        raise ValueError("At least two scoring folds are required.")
    if frame.assign(fold=folds).groupby("split_group_id").fold.nunique().gt(1).any():
        raise ValueError("A split group crosses scoring/calibration folds.")
    for fold in np.unique(folds):
        if set(frame.loc[folds != fold, "target"].unique()) != {0, 1}:
            raise ValueError("Every calibration partition must contain both target classes.")


def confusion_metrics(truth, decisions, scores=None):
    truth, decisions = np.asarray(truth), np.asarray(decisions)
    if truth.ndim != 1 or truth.shape != decisions.shape or not set(np.unique(truth)) <= {0, 1}:
        raise ValueError("Truth and decisions must be aligned binary vectors.")
    if not set(np.unique(decisions)) <= {0, 1, False, True}:
        raise ValueError("Decisions must be binary.")
    positive, calls = truth == 1, decisions.astype(bool)
    tp, fn = int((positive & calls).sum()), int((positive & ~calls).sum())
    fp, tn = int((~positive & calls).sum()), int((~positive & ~calls).sum())
    recall = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    result = {"support": len(truth), "positive_support": tp + fn, "negative_support": tn + fp,
        "true_positives": tp, "false_negatives": fn, "false_positives": fp, "true_negatives": tn,
        "recall": recall, "sensitivity": recall, "specificity": specificity,
        "precision": tp / (tp + fp) if tp + fp else None,
        "accuracy": (tp + tn) / len(truth) if len(truth) else None,
        "balanced_accuracy": (recall + specificity) / 2 if recall is not None and specificity is not None else None,
        "confusion_matrix": [[tn, fp], [fn, tp]]}
    if scores is not None:
        scores = np.asarray(scores, dtype=float)
        if scores.shape != truth.shape or not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
            raise ValueError("Scores must be aligned finite probabilities.")
        both_classes = len(np.unique(truth)) == 2
        result.update(pr_auc=float(average_precision_score(truth, scores)) if both_classes else None,
            roc_auc=float(roc_auc_score(truth, scores)) if both_classes else None,
            brier_score=float(np.mean((scores - truth) ** 2)) if len(truth) else None)
    return result


def fit_recall_threshold(truth, scores, minimum_recall=.95):
    truth, scores = np.asarray(truth), np.asarray(scores, dtype=float)
    if not 0 < minimum_recall <= 1 or truth.shape != scores.shape or truth.ndim != 1:
        raise ValueError("Need aligned vectors and recall target in (0, 1].")
    if set(np.unique(truth)) != {0, 1}:
        raise ValueError("Threshold calibration needs both target classes.")
    confusion_metrics(truth, np.zeros(len(truth), dtype=int), scores)
    positive_scores = np.sort(scores[truth == 1])[::-1]
    required = int(np.ceil(minimum_recall * len(positive_scores)))
    return float(positive_scores[required - 1])


def select_blend(truth, first, second, weights, minimum_recall):
    choices = []
    for weight in weights:
        scores = weight * first + (1 - weight) * second
        threshold = fit_recall_threshold(truth, scores, minimum_recall)
        metrics = confusion_metrics(truth, scores >= threshold, scores)
        choices.append((metrics["specificity"], metrics["pr_auc"], -abs(weight - .5), -weight,
            float(weight), threshold))
    chosen = max(choices)
    return chosen[-2], chosen[-1]


def _slices(frame):
    result = {}
    for column in ("source", "diagnosis"):
        if column in frame:
            result[column] = {str(name): confusion_metrics(group.target.to_numpy(), group.prediction.to_numpy(),
                group.probability.to_numpy()) for name, group in frame.groupby(column, dropna=False, sort=True)}
    return result


def descriptive_tradeoffs(truth, scores, targets=(.80, .90, .95, .97, .99)):
    rows = []
    for target in targets:
        threshold = fit_recall_threshold(truth, scores, target)
        rows.append({"fitted_recall_target": target, "threshold": threshold,
            **confusion_metrics(truth, scores >= threshold)})
    return {"uses_same_cohort_labels_for_threshold_and_metrics": True,
        "for_model_selection_or_deployment": False,
        "interpretation": "Descriptive PR tradeoffs only; cross-fitted blend scores use different fold-specific weights.",
        "rows": rows}


def evaluate_development(frame, *, score_columns=("direct_probability", "diagnosis_probability"),
        blend_columns=("direct_probability", "diagnosis_probability"), blend_weights=None,
        minimum_recall=.95, n_splits=5, seed=20261006, folds=None):
    """Return (JSON-safe report, long-form lesion decisions) without file access.

    Each fixed score column is a separate candidate. For ``crossfit_selected_blend``,
    weight selection maximizes calibration specificity at target recall, then AP;
    only that fold's calibration rows select its weight and threshold. No candidate
    is promoted or selected using the resulting held-out scoring metrics.
    """
    frame = frame.reset_index(drop=True).copy()
    score_columns = tuple(score_columns)
    if not score_columns or len(set(score_columns)) != len(score_columns):
        raise ValueError("Need unique fixed-candidate score columns.")
    if "crossfit_selected_blend" in score_columns:
        raise ValueError("Reserved candidate name: crossfit_selected_blend")
    if blend_columns is not None and (len(blend_columns) != 2 or blend_columns[0] == blend_columns[1]):
        raise ValueError("Blend needs two distinct score columns, or None to disable.")
    required_scores = tuple(dict.fromkeys(score_columns + tuple(blend_columns or ())))
    _validate(frame, required_scores)
    if not 0 < minimum_recall <= 1:
        raise ValueError("Recall target must be in (0, 1].")
    weights = np.linspace(0., 1., 21) if blend_weights is None else np.asarray(blend_weights, dtype=float)
    if weights.ndim != 1 or not len(weights) or not np.isfinite(weights).all() or ((weights < 0) | (weights > 1)).any():
        raise ValueError("Blend weights must be a finite nonempty vector in [0, 1].")
    folds = grouped_folds(frame, n_splits, seed) if folds is None else np.asarray(folds)
    validate_folds(frame, folds)
    truth = frame.target.to_numpy(dtype=int)
    candidates = score_columns + (("crossfit_selected_blend",) if blend_columns is not None else ())
    reports, output = {}, []
    for candidate in candidates:
        scores, thresholds, used_weights = np.full(len(frame), np.nan), np.full(len(frame), np.nan), np.full(len(frame), np.nan)
        fit_reports = []
        for fold in sorted(np.unique(folds)):
            scored, calibration = folds == fold, folds != fold
            if candidate == "crossfit_selected_blend":
                first, second = (frame[column].to_numpy(dtype=float) for column in blend_columns)
                weight, threshold = select_blend(truth[calibration], first[calibration], second[calibration], weights, minimum_recall)
                scores[scored] = weight * first[scored] + (1 - weight) * second[scored]
                used_weights[scored] = weight
            else:
                values = frame[candidate].to_numpy(dtype=float)
                threshold = fit_recall_threshold(truth[calibration], values[calibration], minimum_recall)
                scores[scored] = values[scored]
                weight = None
            thresholds[scored] = threshold
            fit_reports.append({"fold": int(fold), "calibration_lesions": int(calibration.sum()),
                "scoring_lesions": int(scored.sum()), "threshold": threshold,
                "first_component_weight": weight,
                "scoring_metrics": confusion_metrics(truth[scored], scores[scored] >= threshold, scores[scored])})
        calls = (scores >= thresholds).astype(int)
        metadata_columns = list(IDENTITIES) + (["diagnosis"] if "diagnosis" in frame else [])
        decisions = frame[metadata_columns].copy()
        decisions["candidate"], decisions["policy_fold"] = candidate, folds
        decisions["probability"], decisions["threshold"], decisions["prediction"] = scores, thresholds, calls
        decisions["first_component_weight"] = used_weights
        decisions["error_type"] = np.select([(truth == 0) & (calls == 1), (truth == 1) & (calls == 0)],
            ["false_positive", "false_negative"], default="correct")
        slices = _slices(decisions)
        reports[candidate] = {"metrics": confusion_metrics(truth, calls, scores), "folds": fit_reports,
            "by_source": slices.get("source", {}), "by_diagnosis": slices.get("diagnosis", {}),
            "descriptive_tradeoffs": descriptive_tradeoffs(truth, scores)}
        output.append(decisions)
    report = {"policy_version": VERSION, "evaluation_unit": "lesion", "calibration_target": minimum_recall,
        "fold_seed": seed, "fold_count": len(np.unique(folds)), "lesions": len(frame),
        "split_groups": int(frame.split_group_id.nunique()), "fixed_candidates": list(score_columns),
        "blend_components": list(blend_columns) if blend_columns else None,
        "blend_grid": weights.tolist() if blend_columns else None,
        "blend_selection": "Calibration-only specificity at target recall, then AP; threshold fitted on same calibration rows.",
        "model_selection_on_scoring_groups": False, "checkpoint_selection_already_used_development": True,
        "test_or_hiba_access": False, "automatic_promotion": False, "results": reports, "limitations": LIMITATION}
    return report, pd.concat(output, ignore_index=True)
