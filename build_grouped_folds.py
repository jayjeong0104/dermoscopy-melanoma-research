from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create leakage-safe source/diagnosis-stratified lesion folds."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data") / "combined_ham_bcn" / "combined_manifest.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data") / "combined_ham_bcn" / "grouped_folds.csv",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.folds < 3:
        raise ValueError("At least three folds are required for train/validation/test roles.")

    frame = pd.read_csv(args.manifest)
    required = {"lesion_id", "source", "diagnosis", "sha256"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")

    if "split_group_id" not in frame:
        frame["split_group_id"] = frame["lesion_id"]
    if frame.groupby("split_group_id")["source"].nunique().gt(1).any():
        raise ValueError("A split group crosses dataset sources.")
    frame["stratum"] = frame["source"].astype(str) + "::" + frame["diagnosis"].astype(str)
    counts = frame.groupby("stratum")["split_group_id"].nunique()
    if counts.min() < args.folds:
        raise ValueError(
            f"Every stratum needs at least {args.folds} split groups; smallest: "
            f"{counts.nsmallest(5).to_dict()}"
        )

    fold_by_group: dict[str, int] = {}
    splitter = StratifiedGroupKFold(
        n_splits=args.folds, shuffle=True, random_state=args.seed
    )
    for fold, (_, test_indices) in enumerate(
        splitter.split(frame, frame["stratum"], groups=frame["split_group_id"])
    ):
        for group_id in frame.iloc[test_indices]["split_group_id"].unique():
            group_id = str(group_id)
            if group_id in fold_by_group:
                raise RuntimeError("A split group was assigned to multiple folds.")
            fold_by_group[group_id] = fold
    expected_groups = set(frame["split_group_id"].astype(str))
    if set(fold_by_group) != expected_groups:
        raise RuntimeError("At least one split group was not assigned to a fold.")

    assignments = pd.DataFrame(
        {
            "split_group_id": list(fold_by_group),
            "fold": list(fold_by_group.values()),
        }
    )

    image_folds = frame[
        ["lesion_id", "split_group_id", "sha256", "source", "diagnosis"]
    ].merge(
        assignments, on="split_group_id", validate="many_to_one"
    )
    cross_fold_hashes = image_folds.groupby("sha256")["fold"].nunique()
    if (cross_fold_hashes > 1).any():
        raise RuntimeError("Exact image hashes cross folds; deduplicate before training.")

    output = assignments.sort_values("split_group_id")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    summary = {
        "manifest": str(args.manifest.resolve()),
        "output": str(args.output.resolve()),
        "seed": args.seed,
        "folds": args.folds,
        "split_groups": int(len(output)),
        "lesions": int(frame["lesion_id"].nunique()),
        "fold_split_groups": {
            str(key): int(value) for key, value in output["fold"].value_counts().sort_index().items()
        },
        "fold_lesions": {
            str(key): int(value)
            for key, value in image_folds.groupby("fold")["lesion_id"].nunique().items()
        },
        "fold_source_diagnosis_images": [
            {
                "fold": int(key[0]),
                "source": str(key[1]),
                "diagnosis": str(key[2]),
                "images": int(value),
            }
            for key, value in image_folds.groupby(["fold", "source", "diagnosis"]).size().items()
        ],
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
