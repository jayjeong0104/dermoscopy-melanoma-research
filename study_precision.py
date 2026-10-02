"""Reproduce development precision, error composition and paired candidate changes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from compare_model_runs import file_hash
from foundation_common import BASELINE, KEYS, REFERENCE, ROOT, lesion_metadata, save_json, split_frame
from operating_policy import crossfit_policy, decision_metrics


def plot_precision(output: Path, reference: pd.DataFrame, candidates: dict[str, pd.DataFrame]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve

    figure, axis = plt.subplots(figsize=(9, 5.5))
    colors = ["#154c5e", "#cb7437", "#8d4b21", "#429c8b", "#276653"]
    for index, (name, frame) in enumerate({"Retained model": reference, **candidates}.items()):
        precision, recall, _ = precision_recall_curve(frame.target, frame.probability)
        metrics = decision_metrics(frame.target.to_numpy(), frame.prediction.to_numpy())
        color = colors[index % len(colors)]
        axis.plot(recall, precision, color=color, linewidth=1.8, label=name)
        axis.scatter(metrics["recall"], metrics["precision"], color=color, edgecolor="white", s=60, zorder=3)
    axis.axvline(.95, color="#777777", linestyle="--", linewidth=1)
    axis.set(xlim=(.80, 1), ylim=(0, 1), xlabel="Sensitivity (fraction of melanomas detected)",
             ylabel="Precision (fraction of positive flags that are melanoma)",
             title="Precision in the high-sensitivity region")
    axis.grid(alpha=.2)
    axis.legend(loc="upper right", fontsize=8)
    figure.text(.11, .01, "Reused development set: 4,177 lesions. Dots use group-cross-fitted thresholds; curves are descriptive.", fontsize=8)
    figure.tight_layout(rect=(0, .035, 1, 1))
    figure.savefig(output / "precision_recall.png", dpi=160)
    plt.close(figure)


def paired_precision(reference: pd.DataFrame, candidate: pd.DataFrame,
                     iterations: int = 1000, seed: int = 20260930) -> dict:
    keys = KEYS + ["policy_fold"]
    if not reference[keys].equals(candidate[keys]):
        raise ValueError("Paired precision requires identical lesions, labels and calibration groups.")
    codes, groups = pd.factorize(reference.split_group_id)
    if (codes < 0).any():
        raise ValueError("Missing patient/lesion group identities.")
    old = reference.prediction.to_numpy(dtype=bool)
    new = candidate.prediction.to_numpy(dtype=bool)
    y = reference.target.to_numpy(dtype=int)
    group_counts = np.zeros((len(groups), 6), dtype=int)
    for column, values in enumerate((y == 1, old & (y == 1), old, new & (y == 1), new, y == 0)):
        np.add.at(group_counts[:, column], codes, values.astype(int))
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(iterations):
        positives, old_tp, old_calls, new_tp, new_calls, negatives = group_counts[
            rng.integers(0, len(groups), len(groups))].sum(axis=0)
        if min(positives, negatives, old_calls, new_calls) == 0:
            continue
        deltas.append([new_tp / new_calls - old_tp / old_calls,
                       new_tp / positives - old_tp / positives])
    if len(deltas) < .8 * iterations:
        raise ValueError("Insufficient valid paired bootstrap replicates.")
    bounds = np.quantile(deltas, [.025, .975], axis=0)
    return {"groups": len(groups), "iterations": len(deltas),
        "delta_intervals_95": {name: {"lower": float(bounds[0, i]), "upper": float(bounds[1, i])}
                               for i, name in enumerate(("precision", "sensitivity"))},
        "limitation": "Conditional group bootstrap of fixed cross-fitted decisions. Does not repeat "
                      "model selection or threshold fitting; reused validation is not independent confirmation."}


def load_verified_predictions(directory: Path, name: str, report_name: str = "seed42_validation_report.json"):
    report = json.loads((directory / report_name).read_text(encoding="utf-8"))
    path = directory / f"{name}_crossfit_predictions.csv"
    expected = report["results"][name]["predictions_sha256"]
    if file_hash(path) != expected:
        raise ValueError(f"Archived predictions changed: {path}")
    frame = pd.read_csv(path)
    metrics, decisions = crossfit_policy(frame, frame.probability.to_numpy(), folds=frame.policy_fold.to_numpy())
    return metrics, decisions


def study(output: Path, foundation: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    baseline, reference = load_verified_predictions(REFERENCE.parent, "baseline")
    train = split_frame("train")
    val = split_frame("val")
    train_lesions = lesion_metadata(train)
    val_lesions = lesion_metadata(val)
    full = reference.merge(val_lesions, on=KEYS, validate="one_to_one", how="outer", indicator=True)
    if not full._merge.eq("both").all():
        raise ValueError("Validation metadata differs from reference predictions.")
    full["fp"] = (full.target == 0) & (full.prediction == 1)
    full["fn"] = (full.target == 1) & (full.prediction == 0)
    errors = []
    for column in ("source", "diagnosis"):
        for name, group in full.groupby(column):
            negative = int(group.target.eq(0).sum())
            positive = int(group.target.eq(1).sum())
            fp, fn = int(group.fp.sum()), int(group.fn.sum())
            errors.append({"dimension": column, "name": str(name), "lesions": len(group),
                "melanomas": positive, "non_melanomas": negative, "false_positives": fp,
                "false_negatives": fn, "false_positive_rate": fp / negative if negative else None,
                "precision": (positive - fn) / (positive - fn + fp) if positive - fn + fp else None})
    config = json.loads((BASELINE / "run_config.json").read_text(encoding="utf-8"))
    historical = {}
    for name, directory, key in (
        ("frozen_bn_finetune1", "operating_recovery_20260907", "frozen_bn_seed42_epoch_finetune_01"),
        ("lesion_sampling_finetune2", "sampling_recovery_20260908", "lesion_seed42_epoch_finetune_02")):
        metrics, frame = load_verified_predictions(ROOT / "artifacts" / directory, key)
        historical[name] = {"metrics": metrics, "paired_precision": paired_precision(reference, frame)}
    thresholds = []
    for target in (.90, .95, .97, .99):
        metrics, _ = crossfit_policy(reference, reference.probability.to_numpy(), minimum_recall=target,
                                     folds=reference.policy_fold.to_numpy())
        thresholds.append({"calibration_sensitivity_target": target,
                           **{key: metrics[key] for key in ("recall", "precision", "specificity", "false_positives", "false_negatives")}})
    old_precision = baseline["precision"]
    requirements = [{"desired_precision": target,
                     "maximum_false_positives_at_same_275_true_positives": int(np.floor(baseline["true_positives"] * (1 / target - 1))),
                     "illustration_only": True} for target in (.3, .5, .8)]
    results, candidate_frames = {}, {}
    for encoder in ("panderm_base", "incumbent"):
        directory = foundation / "heads" / encoder
        path = directory / "validation_report.json"
        if not path.exists():
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        for seed, saved in report["results"].items():
            predictions = directory / f"seed{seed}_validation_predictions.csv"
            if file_hash(predictions) != saved["predictions_sha256"]:
                raise ValueError("Foundation predictions changed.")
            frame = pd.read_csv(predictions)
            metrics, decisions = crossfit_policy(frame, frame.probability.to_numpy(), folds=frame.policy_fold.to_numpy())
            results[f"{encoder}_seed{seed}"] = {"metrics": metrics,
                "precision_gain_percentage_points": 100 * (metrics["precision"] - old_precision),
                "false_alarms_avoided": baseline["false_positives"] - metrics["false_positives"],
                "additional_melanomas_detected": metrics["true_positives"] - baseline["true_positives"],
                "paired_precision": paired_precision(reference, decisions)}
            candidate_frames[f"{encoder} seed {seed}"] = decisions
    report = {"baseline": baseline,
        "data": {"training_images": len(train), "training_lesions": len(train_lesions),
                 "training_melanoma_images": int(train.target.sum()),
                 "training_melanoma_lesions": int(train_lesions.target.sum()),
                 "validation_lesions": len(reference), "validation_melanomas": int(reference.target.sum()),
                 "validation_melanoma_prevalence": float(reference.target.mean()),
                 "training_positive_sampling_fraction": config["positive_sampling_fraction"]},
        "error_composition": errors, "historical_comparisons": historical,
        "threshold_tradeoffs_exploratory_only": thresholds, "precision_requirements": requirements,
        "foundation_results": results, "source_prediction_sha256": file_hash(REFERENCE),
        "interpretation": {
            "established": ["Low precision reflects many false alarms among numerous non-melanomas.",
                "Most recent experiments reused the same disease-trained encoder and short budgets.",
                "Sampling changes did not improve the high-sensitivity decision rule in completed tests.",
                "Some ranking metrics improved while high-sensitivity decision metrics did not."],
            "plausible_not_proven": ["Visual representations may not adequately distinguish difficult lookalikes.",
                "Independent melanoma diversity and source/case-mix differences may limit generalization.",
                "The old multitask head and loss may contribute; the matched frozen-feature classifier tests this."],
            "not_established": ["A GPU-imposed accuracy ceiling", "That more epochs or more images must improve precision",
                                "That automatic relabeling or deleting difficult cases is justified"]},
        "limitations": "Development analysis with repeatedly inspected validation. Prior/threshold correction "
            "alone cannot improve the ranking of examples. New pretraining overlap is incompletely auditable. "
            "A non-melanoma target includes other cancers; it must not be called healthy or benign."}
    save_json(output / "precision_study.json", report)
    pd.DataFrame(errors).to_csv(output / "error_composition.csv", index=False)
    plot_precision(output, reference, candidate_frames)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/precision_study_20260930")
    parser.add_argument("--foundation-dir", type=Path, default=ROOT / "artifacts/foundation_recovery_20260909")
    args = parser.parse_args()
    report = study(args.output_dir, args.foundation_dir)
    print(json.dumps({"data": report["data"], "baseline_precision": report["baseline"]["precision"],
                      "foundation_results": list(report["foundation_results"]), "output": str(args.output_dir)}, indent=2))
