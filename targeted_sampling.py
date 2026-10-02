"""Training-only, capped hard-negative sampling. Never modifies source images."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

IDENTITY = ["source", "image_id", "lesion_id", "split_group_id", "sha256", "diagnosis", "target"]


def canonical(frame: pd.DataFrame) -> pd.DataFrame:
    if frame[IDENTITY].isna().any().any() or frame.duplicated(["source", "image_id"]).any():
        raise ValueError("Missing or duplicate training identities.")
    return frame[IDENTITY].astype("string").sort_values(["source", "image_id"]).reset_index(drop=True)


def assert_training_only(train: pd.DataFrame, holdouts: list[pd.DataFrame]) -> None:
    canonical(train)
    for held_out in holdouts:
        for column in ("sha256", "lesion_id", "split_group_id"):
            if train[column].isna().any() or held_out[column].isna().any():
                raise ValueError(f"Missing leakage-check identity: {column}")
            if set(train[column].astype(str)) & set(held_out[column].astype(str)):
                raise ValueError(f"Training/holdout overlap: {column}")


def mine_weights(train: pd.DataFrame, lesion_scores: pd.DataFrame,
                 fraction: float = 0.2, multiplier: float = 2.0) -> pd.DataFrame:
    """Rank negative lesions within source/diagnosis, not across datasets."""
    if not 0 < fraction <= 0.25 or multiplier != 2.0:
        raise ValueError("Mining fraction must be <= 0.25 and multiplier must be 2.")
    canonical(train)
    grouped = train.groupby(["source", "lesion_id"])
    if (grouped[["target", "diagnosis"]].nunique() != 1).any().any():
        raise ValueError("Conflicting labels within a training lesion.")
    expected = grouped[["target", "diagnosis"]].first().sort_index()
    scores = lesion_scores.set_index(["source", "lesion_id"]).sort_index()
    if scores.index.has_duplicates or not expected.index.equals(scores.index):
        raise ValueError("Mining predictions must cover exactly the training lesions.")
    if not expected.astype("string").equals(scores[["target", "diagnosis"]].astype("string")):
        raise ValueError("Mining labels differ from training labels.")
    values = scores.probability.to_numpy(dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Invalid mining probabilities.")
    scores = scores.reset_index()
    scores["sampling_weight"] = 1.0
    negative = scores.target.astype(int).eq(0) & scores.diagnosis.ne("UNKNOWN")
    for _, group in scores.loc[negative].groupby(["source", "diagnosis"], sort=True):
        # Skip tiny strata so the cap holds even for rare diagnoses.
        count = math.floor(len(group) * fraction)
        ranked = group.sort_values(["probability", "lesion_id"], ascending=[False, True])
        scores.loc[ranked.index[:count], "sampling_weight"] = multiplier
    result = train[IDENTITY].merge(
        scores[["source", "lesion_id", "probability", "sampling_weight"]],
        on=["source", "lesion_id"], validate="many_to_one", how="left", sort=False,
    )
    return result.rename(columns={"probability": "mining_probability"})


def attach_weights(train: pd.DataFrame, path: Path) -> pd.DataFrame:
    weights = pd.read_csv(path, dtype={column: "string" for column in IDENTITY}, low_memory=False)
    if not canonical(train).equals(canonical(weights)):
        raise ValueError("Sampling file must match exactly the training identities and labels.")
    values = pd.to_numeric(weights["sampling_weight"], errors="raise")
    if not np.isfinite(values).all() or not values.isin([1.0, 2.0]).all():
        raise ValueError("Sampling weights must be either 1 or 2.")
    if (values.loc[weights.target.astype(int).eq(1)] != 1).any():
        raise ValueError("Positive sampling weights must remain unchanged.")
    weights["sampling_weight"] = values
    return train.merge(weights[["source", "image_id", "sampling_weight"]],
                       on=["source", "image_id"], how="left", validate="one_to_one", sort=False)
