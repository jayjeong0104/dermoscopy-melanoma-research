"""CPU-only integrity and evaluation helpers for the frozen-encoder study."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from compare_model_runs import file_hash
from operating_policy import decision_metrics
from targeted_sampling import assert_training_only, canonical
from artifact_io import atomic_output

ROOT = Path(__file__).resolve().parent
BASELINE = ROOT / "artifacts/melanoma_v1/run_20260822_111527"
REFERENCE = ROOT / "artifacts/sampling_recovery_20260908/baseline_crossfit_predictions.csv"
ASSETS = ROOT / "artifacts/pretrained/panderm_base"
ENCODERS = ("incumbent", "panderm_base")
KEYS = ["source", "lesion_id", "split_group_id", "target"]
LIMITATION = (
    "Development comparison, not independent confirmation. Validation has been reused. "
    "PanDerm's private pretraining identities are unavailable for exhaustive overlap checks; "
    "the smaller Base checkpoint is not the paper's Large model. The incumbent encoder "
    "already saw all original training lesions, including the classifier's inner holdout. "
    "Inner scores select head settings only, and are not an unbiased encoder comparison. "
    "Patient separation is only as complete as available identities. Target is melanoma "
    "versus non-melanoma, not all cancers versus benign. No clinical-use claim or automatic promotion."
)


def save_json(path: Path, value: dict) -> None:
    payload = json.dumps(value, indent=2, allow_nan=False)
    with atomic_output(path) as temporary:
        temporary.write_text(payload, encoding="utf-8")


def fingerprint(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def runtime_path(value: str) -> Path:
    match = re.match(r"^([A-Za-z]):[\\/](.*)$", value)
    if os.name != "nt" and match:
        return Path(f"/mnt/{match[1].lower()}/{match[2].replace(chr(92), '/')}")
    return Path(value)


def split_frame(name: str) -> pd.DataFrame:
    if name not in {"train", "val", "test"}:
        raise ValueError("Unknown data split.")
    frame = pd.read_csv(BASELINE / f"{name}_split.csv", low_memory=False)
    canonical(frame)
    if not frame.target.eq(frame.diagnosis.eq("MEL").astype(int)).all():
        raise ValueError("Melanoma target mismatch.")
    return frame.sort_values(["source", "image_id"]).reset_index(drop=True)


def lesion_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    canonical(frame)
    grouped = frame.groupby(["source", "lesion_id"], sort=True)
    if grouped[["target", "diagnosis", "split_group_id"]].nunique(dropna=False).ne(1).any().any():
        raise ValueError("Conflicting labels or split groups within a lesion.")
    result = grouped[["split_group_id", "target", "diagnosis", "image_path", "image_id"]].first()
    result["image_count"] = grouped.size()
    return result.reset_index()


def aggregate_features(frame: pd.DataFrame, features: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    if features.ndim != 2 or features.shape[0] != len(frame) or not np.isfinite(features).all():
        raise ValueError("Feature rows must be finite and aligned with every image.")
    meta = lesion_metadata(frame)
    codes = pd.MultiIndex.from_frame(meta[["source", "lesion_id"]]).get_indexer(
        pd.MultiIndex.from_frame(frame[["source", "lesion_id"]]))
    output = np.zeros((len(meta), features.shape[1]), dtype=np.float64)
    np.add.at(output, codes, features)
    output /= np.bincount(codes)[:, None]
    return meta, output.astype("float32")


def inner_partition(meta: pd.DataFrame) -> np.ndarray:
    strata = meta.source.astype(str) + "::" + meta.target.astype(str)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20260909)
    _, held = next(splitter.split(meta, strata, meta.split_group_id))
    result = np.zeros(len(meta), dtype=bool)
    result[held] = True
    if set(meta.loc[result, "split_group_id"]) & set(meta.loc[~result, "split_group_id"]):
        raise ValueError("Inner training/calibration group overlap.")
    if any(set(meta.loc[mask, "target"]) != {0, 1} for mask in (result, ~result)):
        raise ValueError("Both classes are needed in each inner partition.")
    return result


def validate_splits() -> dict[str, pd.DataFrame]:
    splits = {name: split_frame(name) for name in ("train", "val", "test")}
    assert_training_only(splits["train"], [splits["val"], splits["test"]])
    assert_training_only(splits["val"], [splits["test"]])
    for frame in splits.values():
        lesion_metadata(frame)
    return splits


def read_chunk(path: Path, expected: str, rows: int, dimensions: int) -> np.ndarray:
    with np.load(path, allow_pickle=False) as saved:
        features = saved["features"]
        if str(saved["fingerprint"].item()) != expected:
            raise ValueError(f"Stale or mismatched feature cache: {path}")
        if features.shape != (rows, dimensions) or not np.isfinite(features).all():
            raise ValueError(f"Invalid feature cache: {path}")
        if str(saved["features_sha256"].item()) != hashlib.sha256(features.tobytes()).hexdigest():
            raise ValueError(f"Feature cache integrity failure: {path}")
    return features


def save_chunk(path: Path, features: np.ndarray, key: str) -> None:
    if not np.isfinite(features).all():
        raise ValueError("Refusing to cache invalid features.")
    with atomic_output(path) as temporary:
        with temporary.open("wb") as stream:
            np.savez(stream, features=features, fingerprint=key,
                     features_sha256=hashlib.sha256(features.tobytes()).hexdigest())


def error_audit(reference: pd.DataFrame, images: pd.DataFrame, output: Path) -> dict:
    meta = lesion_metadata(images)
    frame = reference.merge(meta, on=KEYS, how="outer", validate="one_to_one", indicator=True)
    if not frame._merge.eq("both").all():
        raise ValueError("Error review identities differ from validation images.")
    frame["error"] = np.select(
        [(frame.target == 1) & (frame.prediction == 0), (frame.target == 0) & (frame.prediction == 1)],
        ["false_negative", "false_positive"], default="correct")
    errors = frame.loc[frame.error.ne("correct")].drop(columns="_merge").copy()
    errors["review_status"] = "not_reviewed"
    errors["expert_notes"] = ""
    errors.sort_values(["error", "source", "probability"]).to_csv(output / "expert_review_queue.csv", index=False)
    breakdown = {}
    for column in ("source", "diagnosis"):
        rows = []
        for name, group in frame.groupby(column):
            rows.append({"name": str(name), "lesions": len(group),
                "melanomas": int(group.target.sum()),
                "false_negatives": int(group.error.eq("false_negative").sum()),
                "false_positives": int(group.error.eq("false_positive").sum())})
        breakdown[column] = rows
    report = {"metrics": decision_metrics(frame.target.to_numpy(), frame.prediction.to_numpy()),
              "breakdown": breakdown, "labels_changed": False, "images_removed": False,
              "expert_review_completed": False,
              "warning": "Automated error inventory, not expert adjudication or proof of label errors. " + LIMITATION}
    save_json(output / "error_audit.json", report)
    return report
