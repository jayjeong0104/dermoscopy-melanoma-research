"""Rebuild canonical checkpoint scores from verified image chunks, without inference."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from compare_model_runs import file_hash
from foundation_common import KEYS, aggregate_features, fingerprint, read_chunk

DIAGNOSES = ("AK", "BCC", "BKL", "DF", "MEL", "NV", "SCC", "VASC")
FEATURE_DIMENSIONS = 1280


def load_checkpoint_scores(featuresdir: Path, run: Path, split: str = "val") -> pd.DataFrame:
    """Return cached lesion order, mean-logit direct scores and mean diagnosis scores.

    Only train/validation manifests and existing chunks are read. No source images,
    TensorFlow, model execution, cache mutation, or test data access is involved.
    """
    if split not in {"train", "val"}:
        raise ValueError("Only train/val checkpoint scores are supported; test access is forbidden.")
    featuresdir, run = Path(featuresdir), Path(run)
    plan = json.loads((featuresdir / "plan.json").read_text(encoding="utf-8"))
    marker = json.loads((featuresdir / f"{split}_complete.json").read_text(encoding="utf-8"))
    required_outputs = {"plan.json", f"{split}_lesions.csv", f"{split}_lesion_features.npz"}
    recorded = marker.get("outputs_sha256", {})
    if not required_outputs.issubset(recorded):
        raise ValueError("Completed features are missing required output bindings.")
    for name, expected in recorded.items():
        path = featuresdir / name
        if not path.resolve().is_relative_to(featuresdir.resolve()):
            raise ValueError("Feature output binding escapes its cache directory.")
        if file_hash(path) != expected:
            raise ValueError(f"Completed feature output changed: {name}")
    if marker.get("plan_fingerprint") != fingerprint(plan):
        raise ValueError("Feature completion is not bound to its extraction plan.")
    if plan.get("model_sha256") != file_hash(run / "best_finetune_model.keras"):
        raise ValueError("Feature checkpoint binding does not match the named run.")
    if plan.get("splits_sha256", {}).get(split) != file_hash(run / f"{split}_split.csv"):
        raise ValueError("Feature split binding does not match the named run.")
    if plan.get("metadata") != "unknown" or plan.get("tta_transforms") != 1:
        raise ValueError("Canonical training scores require unknown metadata and one image view.")
    if "metadata_schema_sha256" in plan and plan["metadata_schema_sha256"] != file_hash(run / "metadata_schema.json"):
        raise ValueError("Feature metadata schema does not match the named run.")
    config = json.loads((run / "run_config.json").read_text(encoding="utf-8"))
    if config.get("binary_target") != "melanoma" or tuple(config.get("diagnoses", ())) != DIAGNOSES:
        raise ValueError("Unsupported checkpoint target or diagnosis score order.")
    if plan.get("image_size") != config.get("image_size"):
        raise ValueError("Feature image size does not match the named run.")

    frame = pd.read_csv(run / f"{split}_split.csv", low_memory=False)
    required = ["source", "image_id", "lesion_id", "split_group_id", "sha256", "image_path", "target", "diagnosis"]
    if frame.empty or frame[required].isna().any().any() or frame.duplicated(["source", "image_id"]).any():
        raise ValueError("Invalid image identities in the bound split.")
    if not frame.target.eq(frame.diagnosis.eq("MEL").astype(int)).all():
        raise ValueError("Bound split has inconsistent melanoma targets.")
    frame = frame.sort_values(["source", "image_id"]).reset_index(drop=True)
    chunk_images = plan.get("chunk_images", 256)  # Legacy version-2 extraction size.
    if not isinstance(chunk_images, int) or isinstance(chunk_images, bool) or chunk_images < 1:
        raise ValueError("Invalid extraction chunk size.")
    pieces = []
    for start in range(0, len(frame), chunk_images):
        rows = frame.iloc[start:start + chunk_images]
        key = fingerprint({"plan": plan, "split": split, "start": start,
            "rows": rows[["source", "image_id", "sha256"]].to_dict("records")})
        values = read_chunk(featuresdir / f"{split}_{start:06d}.npz", key, len(rows), FEATURE_DIMENSIONS + 9)
        pieces.append(values[:, FEATURE_DIMENSIONS:].copy())
    scores = np.concatenate(pieces)
    if ((scores < 0) | (scores > 1)).any() or not np.allclose(scores[:, 1:].sum(axis=1), 1., atol=1e-5):
        raise ValueError("Cached checkpoint probabilities are invalid.")
    meta, means = aggregate_features(frame, scores)
    clipped = np.clip(scores[:, 0].astype("float64"), 1e-6, 1. - 1e-6)
    working = frame[["source", "lesion_id"]].assign(logit=np.log(clipped / (1. - clipped)))
    mean_logits = working.groupby(["source", "lesion_id"], sort=True).logit.mean()
    mean_logits = mean_logits.reindex(pd.MultiIndex.from_frame(meta[["source", "lesion_id"]])).to_numpy()
    meta["direct_probability"] = (1. / (1. + np.exp(-mean_logits))).astype("float32")
    for index, name in enumerate(DIAGNOSES):
        meta[f"diagnosis_probability_{name}"] = means[:, 1 + index]
    meta["diagnosis_probability"] = meta["diagnosis_probability_MEL"]

    cached_meta = pd.read_csv(featuresdir / f"{split}_lesions.csv", low_memory=False)
    if cached_meta[KEYS].isna().any().any() or cached_meta.duplicated(["source", "lesion_id"]).any():
        raise ValueError("Completed lesion identities are invalid.")
    locations = pd.MultiIndex.from_frame(meta[KEYS]).get_indexer(pd.MultiIndex.from_frame(cached_meta[KEYS]))
    if len(meta) != len(cached_meta) or (locations < 0).any():
        raise ValueError("Completed lesion labels/groups do not match the bound image chunks.")
    meta = meta.iloc[locations].reset_index(drop=True)
    for name in ("diagnosis", "image_count"):
        if not np.array_equal(meta[name].to_numpy(), cached_meta[name].to_numpy()):
            raise ValueError(f"Completed lesion {name} does not match the bound image chunks.")
    return meta
