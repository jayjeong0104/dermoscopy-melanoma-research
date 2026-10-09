"""Training-only ranking selection, separated from calibrated-policy eligibility.

Within-source specificity at target recall measures ranking in the intended
operating region. Thresholds here use scoring labels solely for hyperparameter
selection within training data; they are neither deployable thresholds nor an
unbiased estimate of calibrated-policy performance.
"""
from __future__ import annotations

import numpy as np

from development_evaluation import confusion_metrics, fit_recall_threshold


def evaluate_training_rank_quality(frame, score_columns, *, minimum_recall=.95):
    """Evaluate fixed candidates on their group-held-out training predictions.

    The caller must supply training-only out-of-fold scores from heads fitted
    without each row's split group. No validation/test data or file IO is used.
    Each source and inner scoring fold gets its own descriptive order-statistic
    threshold so source probability offsets do not masquerade as ranking quality.
    """
    score_columns = tuple(score_columns)
    required = ["source", "lesion_id", "split_group_id", "target", "inner_fold", *score_columns]
    if not score_columns or len(set(score_columns)) != len(score_columns):
        raise ValueError("Need unique training candidate score columns.")
    if set(required) - set(frame) or frame[required].isna().any().any():
        raise ValueError("Incomplete training out-of-fold identities or scores.")
    if frame.duplicated(["source", "lesion_id"]).any():
        raise ValueError("Training lesions must be unique.")
    folds = frame.inner_fold.to_numpy()
    if not np.issubdtype(folds.dtype, np.integer) or len(np.unique(folds)) < 2 or (folds < 0).any():
        raise ValueError("Need at least two valid inner scoring folds.")
    if frame.groupby("split_group_id").inner_fold.nunique().gt(1).any():
        raise ValueError("A training split group crosses scoring folds.")
    reports = {}
    for column in score_columns:
        cells = []
        for (source, fold), rows in frame.groupby(["source", "inner_fold"], sort=True):
            truth, scores = rows.target.to_numpy(), rows[column].to_numpy(dtype=float)
            threshold = fit_recall_threshold(truth, scores, minimum_recall)
            cells.append({"source": str(source), "inner_fold": int(fold),
                "descriptive_threshold": threshold,
                **confusion_metrics(truth, scores >= threshold, scores)})
        sources = sorted(frame.source.astype(str).unique())
        by_source = {}
        for source in sources:
            chosen = [cell for cell in cells if cell["source"] == source]
            if len(chosen) != len(np.unique(folds)):
                raise ValueError("Every source needs both classes in every scoring fold.")
            by_source[source] = {
                "specificity_at_target": float(np.mean([cell["specificity"] for cell in chosen])),
                "average_precision": float(np.mean([cell["pr_auc"] for cell in chosen]))}
        reports[column] = {"macro_specificity_at_target": float(np.mean([
                values["specificity_at_target"] for values in by_source.values()])),
            "worst_source_specificity_at_target": min(values["specificity_at_target"] for values in by_source.values()),
            "macro_average_precision": float(np.mean([values["average_precision"] for values in by_source.values()])),
            "by_source": by_source, "source_fold_cells": cells}
    selected = max(score_columns, key=lambda name: (
        reports[name]["macro_specificity_at_target"], reports[name]["worst_source_specificity_at_target"],
        reports[name]["macro_average_precision"]))
    if reports[selected]["macro_specificity_at_target"] == 0:
        selected = None
    return {"version": 1, "selected_candidate": selected, "minimum_recall": minimum_recall,
        "selection_role": "Training-only hyperparameter ranking; thresholds fitted on inner scoring labels.",
        "calibrated_policy_eligibility": "Not determined. Evaluate thresholds fitted only on disjoint calibration groups; unmet recall constraints remain unmet.",
        "outer_validation_used": False, "test_or_hiba_access": False,
        "automatic_promotion": False, "candidates": reports,
        "limitation": "Selection scores are optimistic after candidate selection and conditional on any encoder training exposure. Source-specific descriptive thresholds are not a deployed source-specific policy."}
