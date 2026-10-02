from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import train_multitask as training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate an average or rank ensemble on a shared held-out split."
    )
    parser.add_argument("--run-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--method", choices=("probability", "rank"), default="probability")
    parser.add_argument("--unit", choices=("lesion", "image"), default="lesion")
    parser.add_argument("--minimum-recall", type=float, default=0.80)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_predictions(
    run_dir: Path, split: str, binary_target: str, unit: str
) -> pd.DataFrame:
    path = run_dir / (
        f"{split}_lesion_predictions.csv" if unit == "lesion" else f"{split}_predictions.csv"
    )
    if not path.exists():
        raise FileNotFoundError(f"Predictions not found: {path}")
    frame = pd.read_csv(path)
    probability_column = "probability" if unit == "lesion" else f"probability_{binary_target}"
    if unit == "image" and probability_column not in frame:
        probability_column = "probability_high_risk"
    identifier_column = "lesion_id" if unit == "lesion" else "image_id"
    required = {"source", identifier_column, "target", probability_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    result = frame[["source", identifier_column, "target", probability_column]].copy()
    result["key"] = (
        result["source"].astype(str) + "::" + result[identifier_column].astype(str)
    )
    if result["key"].duplicated().any():
        raise ValueError(f"Duplicate prediction keys in {path}")
    return result.set_index("key").sort_index().rename(
        columns={probability_column: "probability"}
    )


def combine(
    frames: list[pd.DataFrame], method: str
) -> tuple[pd.DataFrame, np.ndarray]:
    reference = frames[0]
    values = []
    for frame in frames:
        if not frame.index.equals(reference.index):
            raise ValueError("Ensemble runs do not use the same held-out images.")
        if not np.array_equal(frame["target"].to_numpy(), reference["target"].to_numpy()):
            raise ValueError("Ensemble runs disagree on held-out labels.")
        probabilities = frame["probability"].to_numpy(dtype="float64")
        if method == "rank":
            probabilities = pd.Series(probabilities).rank(method="average", pct=True).to_numpy()
        values.append(probabilities)
    return reference, np.mean(np.vstack(values), axis=0).astype("float32")


def main() -> None:
    args = parse_args()
    run_dirs = [path.resolve() for path in args.run_dirs]
    configs = [
        json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
        for run_dir in run_dirs
    ]
    targets = {config.get("binary_target", "high_risk") for config in configs}
    if len(targets) != 1:
        raise ValueError(f"Ensemble runs use different binary targets: {sorted(targets)}")
    binary_target = targets.pop()

    val_reference, val_probabilities = combine(
        [load_predictions(run_dir, "val", binary_target, args.unit) for run_dir in run_dirs],
        args.method,
    )
    test_reference, test_probabilities = combine(
        [load_predictions(run_dir, "test", binary_target, args.unit) for run_dir in run_dirs],
        args.method,
    )
    val_truth = val_reference["target"].to_numpy(dtype="int32")
    test_truth = test_reference["target"].to_numpy(dtype="int32")
    balanced, operating, rows = training.select_thresholds(
        val_truth, val_probabilities, args.minimum_recall
    )
    threshold = float(operating["threshold"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "binary_target": binary_target,
        "evaluation_unit": args.unit,
        "method": args.method,
        "run_dirs": [str(path) for path in run_dirs],
        "validation_probability_metrics": training.probability_metrics(
            val_truth, val_probabilities
        ),
        "validation_balanced_threshold_metrics": balanced,
        "validation_operating_threshold_metrics": operating,
        "test_probability_metrics": training.probability_metrics(
            test_truth, test_probabilities
        ),
        "test_operating_threshold_metrics": training.threshold_metrics(
            test_truth, test_probabilities, threshold
        ),
    }
    training.save_json(args.output_dir / "ensemble_metrics.json", results)
    pd.DataFrame(rows).drop(columns=["confusion_matrix"]).to_csv(
        args.output_dir / "threshold_sweep.csv", index=False
    )
    output_predictions = test_reference.reset_index()
    output_predictions["ensemble_probability"] = test_probabilities
    output_predictions["ensemble_prediction"] = (test_probabilities >= threshold).astype("int32")
    output_predictions.to_csv(args.output_dir / "test_predictions.csv", index=False)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
