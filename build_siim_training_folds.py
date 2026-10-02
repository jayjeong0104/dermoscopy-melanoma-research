from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SIIM_SOURCE = "SIIM_ISIC_2020"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preserve the established validation/test folds and assign SIIM patients only "
            "to training folds for a controlled improvement run."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data")
        / "combined_ham_bcn_derm12345_siim2020"
        / "combined_manifest.csv",
    )
    parser.add_argument(
        "--base-folds",
        type=Path,
        default=Path("data")
        / "combined_ham_bcn_derm12345"
        / "grouped_folds.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data")
        / "combined_ham_bcn_derm12345_siim2020"
        / "training_folds.csv",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def assign_training_folds(groups: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pieces: list[pd.DataFrame] = []
    for target, subset in groups.groupby("patient_melanoma", sort=True):
        subset = subset.copy()
        order = rng.permutation(len(subset))
        subset = subset.iloc[order].reset_index(drop=True)
        subset["fold"] = np.asarray([2, 3, 4], dtype="int32")[
            np.arange(len(subset)) % 3
        ]
        pieces.append(subset)
    return pd.concat(pieces, ignore_index=True)


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.manifest, low_memory=False)
    base_folds = pd.read_csv(args.base_folds)
    required = {"split_group_id", "source", "target", "sha256"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    if not {"split_group_id", "fold"}.issubset(base_folds.columns):
        raise ValueError("Base folds must contain split_group_id and fold.")
    if base_folds["split_group_id"].duplicated().any():
        raise ValueError("Base fold assignments contain duplicate groups.")
    if frame["sha256"].duplicated().any():
        raise ValueError("Augmented manifest still contains exact duplicate hashes.")

    siim = frame[frame["source"].eq(SIIM_SOURCE)].copy()
    if siim.empty:
        raise ValueError("Augmented manifest contains no SIIM-ISIC 2020 rows.")
    base = frame[~frame["source"].eq(SIIM_SOURCE)].copy()
    expected_base_groups = set(base["split_group_id"].astype(str))
    assigned_base_groups = set(base_folds["split_group_id"].astype(str))
    if expected_base_groups != assigned_base_groups:
        raise ValueError("Base manifest groups no longer match the established fold file.")

    patient_groups = (
        siim.groupby("split_group_id", as_index=False)["target"]
        .max()
        .rename(columns={"target": "patient_melanoma"})
    )
    siim_folds = assign_training_folds(patient_groups, args.seed)
    output = pd.concat(
        [
            base_folds[["split_group_id", "fold"]],
            siim_folds[["split_group_id", "fold"]],
        ],
        ignore_index=True,
    )
    if output["split_group_id"].duplicated().any():
        raise RuntimeError("A split group was assigned more than once.")
    expected_groups = set(frame["split_group_id"].astype(str))
    if set(output["split_group_id"].astype(str)) != expected_groups:
        raise RuntimeError("Fold assignments do not cover the augmented manifest.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.sort_values("split_group_id").to_csv(args.output, index=False)
    summary = {
        "manifest": str(args.manifest.resolve()),
        "base_folds": str(args.base_folds.resolve()),
        "output": str(args.output.resolve()),
        "seed": args.seed,
        "validation_fold": 1,
        "test_fold": 0,
        "siim_training_only": True,
        "total_groups": int(len(output)),
        "siim_groups": int(len(siim_folds)),
        "siim_melanoma_patient_groups": int(siim_folds["patient_melanoma"].sum()),
        "fold_group_counts": {
            str(key): int(value)
            for key, value in output["fold"].value_counts().sort_index().items()
        },
        "siim_fold_group_counts": {
            str(key): int(value)
            for key, value in siim_folds["fold"].value_counts().sort_index().items()
        },
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
