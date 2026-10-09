"""Check supervised checkpoint ancestry before reusing weights on new holdouts.

Within-run split disjointness cannot undo examples learned by a parent model.
Validation exposure counts too: it influences checkpoint and model selection.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import re

import pandas as pd


IDENTITY_COLUMNS = ("source", "image_id", "lesion_id", "sha256")


class ProvenanceError(ValueError):
    """A checkpoint is not eligible for a new unbiased holdout evaluation."""

    def __init__(self, message: str, report: dict | None = None):
        super().__init__(message)
        self.report = report


def portable_path(value: str | Path, base_dir: Path, platform: str | None = None) -> Path:
    """Resolve recorded Windows/WSL paths without guessing a replacement artifact."""
    value = str(value)
    platform = os.name if platform is None else platform
    windows = re.match(r"^([A-Za-z]):[\\/](.*)$", value)
    wsl = re.match(r"^/mnt/([A-Za-z])/(.*)$", value)
    if platform == "nt" and wsl:
        value = str(PureWindowsPath(wsl[1].upper() + ":/" + wsl[2]))
    elif platform != "nt" and windows:
        value = "/mnt/" + windows[1].lower() + "/" + windows[2].replace("\\", "/")
    path = Path(value)
    # PureWindowsPath also recognizes drive paths in cross-platform unit tests.
    if not value.startswith("/") and not path.is_absolute() and not PureWindowsPath(value).is_absolute():
        path = base_dir / path
    return path


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def identity_frame(frame: pd.DataFrame, context: str) -> pd.DataFrame:
    missing = set(IDENTITY_COLUMNS) - set(frame)
    if missing:
        raise ProvenanceError(f"Missing provenance identities in {context}: {sorted(missing)}")
    columns = [*IDENTITY_COLUMNS, *(["split_group_id"] if "split_group_id" in frame else [])]
    result = frame[columns].astype("string").copy()
    if result.empty or result.isna().any().any():
        raise ProvenanceError(f"Empty or missing provenance identities in {context}.")
    for column in columns:
        result[column] = result[column].str.strip()
        if result[column].eq("").any():
            raise ProvenanceError(f"Blank provenance identity {column} in {context}.")
    result["sha256"] = result["sha256"].str.lower()
    if result.duplicated(["source", "image_id"]).any():
        raise ProvenanceError(f"Duplicate image identities in {context}.")
    return result


def overlap_counts(holdout: pd.DataFrame, exposed: pd.DataFrame) -> dict[str, int]:
    """Count affected holdout rows by each independently useful identity."""
    columns = ["lesion_id", "sha256"]
    if "split_group_id" in holdout and "split_group_id" in exposed:
        columns.append("split_group_id")
    counts = {name: int(holdout[name].isin(set(exposed[name])).sum()) for name in columns}
    counts["source_image_id"] = int(pd.MultiIndex.from_frame(holdout[["source", "image_id"]]).isin(
        pd.MultiIndex.from_frame(exposed[["source", "image_id"]])).sum())
    return counts


def audit_warm_start(model_path: Path, holdouts: dict[str, pd.DataFrame],
                     project_root: Path) -> dict:
    """Return complete ancestry evidence, or fail closed on missing provenance.

    A split used only as a prior test set is reported separately. Its reuse still
    requires disclosure, but alone does not prove gradient/selection exposure.
    """
    checked = {name: identity_frame(frame, f"current {name}") for name, frame in holdouts.items()}
    if not checked:
        raise ProvenanceError("At least one current holdout is required.")
    report = {"version": 1, "ancestors": [], "contaminated": False,
              "prior_test_reuse": False, "basis": "Recorded local supervised train and validation exposure"}
    visited = set()
    path = portable_path(model_path, project_root)
    while True:
        path = path.resolve()
        if path in visited:
            raise ProvenanceError(f"Cyclic warm-start ancestry: {path}", report)
        visited.add(path)
        if not path.is_file():
            raise ProvenanceError(f"Missing ancestor checkpoint: {path}", report)
        run_dir = path.parent
        config_path = run_dir / "run_config.json"
        if not config_path.is_file():
            raise ProvenanceError(f"Missing ancestor run_config.json for {path}", report)
        config = json.loads(config_path.read_text(encoding="utf-8-sig"))
        ancestor = {"model": str(path), "model_sha256": _digest(path),
                    "run_config_sha256": _digest(config_path), "splits": {}}
        for split in ("train", "val", "test"):
            split_path = run_dir / f"{split}_split.csv"
            if not split_path.is_file():
                raise ProvenanceError(f"Missing ancestor split provenance: {split_path}", report)
            exposed = identity_frame(pd.read_csv(split_path, dtype="string"), str(split_path))
            matches = {name: overlap_counts(frame, exposed) for name, frame in checked.items()}
            overlap = any(any(values.values()) for values in matches.values())
            report["contaminated" if split in {"train", "val"} else "prior_test_reuse"] |= overlap
            ancestor["splits"][split] = {"sha256": _digest(split_path), "rows": len(exposed),
                                         "holdout_overlap_rows": matches}
        report["ancestors"].append(ancestor)
        parent = config.get("warm_start_model")
        if not parent:
            if config.get("weights") not in {"imagenet", "none"}:
                raise ProvenanceError(f"Unknown root weight provenance in {config_path}", report)
            report["root_initialization"] = config["weights"]
            return report
        path = portable_path(parent, project_root)


def require_clean_warm_start(report: dict) -> None:
    if report["contaminated"]:
        raise ProvenanceError(
            "Warm-start ancestry overlaps current validation/test identities in ancestor training "
            "or validation splits. Resetting only the classifier does not remove this exposure. "
            "Use clean fixed splits with ImageNet initialization, or an audited disjoint checkpoint.", report)


def resolve_training_metadata_policy(requested: str | None, saved_config: dict | None = None) -> str:
    if requested is not None and requested not in {"unknown", "provided"}:
        raise ValueError("Unknown training metadata policy.")
    if saved_config is not None:
        saved = saved_config.get("training_metadata_policy", "provided")
        if requested is not None and requested != saved:
            raise ValueError("Training metadata policy cannot change when resuming a run.")
        return saved
    return requested or "unknown"


def assert_same_resume_identities(saved: pd.DataFrame, current: pd.DataFrame, name: str) -> None:
    saved, current = saved.copy(), current.copy()
    for frame in (saved, current):
        if "split_group_id" not in frame and "lesion_id" in frame:
            frame["split_group_id"] = frame["lesion_id"]
    old = identity_frame(saved, f"saved {name}")
    new = identity_frame(current, f"current {name}")
    if set(old.columns) != set(new.columns):
        raise ProvenanceError(f"Resume identity schema changed for {name}.")
    columns = list(old.columns)
    old = old.sort_values(["source", "image_id"]).reset_index(drop=True)
    new = new[columns].sort_values(["source", "image_id"]).reset_index(drop=True)
    if not old.equals(new):
        raise ProvenanceError(f"Resume split identities changed for {name}.")
    for frame in (saved, current):
        for column in ("source", "image_id"):
            frame[column] = frame[column].astype("string").str.strip()
    saved = saved.sort_values(["source", "image_id"]).reset_index(drop=True)
    current = current.sort_values(["source", "image_id"]).reset_index(drop=True)
    # Aggregate split counts cannot detect swapped labels, and an unchanged age
    # median cannot detect changed per-example metadata. Bind both to each image.
    label_columns = ("target", "diagnosis", "diagnosis_index", "diagnosis_label_available", "melanoma_label")
    metadata_columns = ("age_approx", "sex", "anatom_site_general")
    for column in (*label_columns, *metadata_columns):
        if column == "diagnosis_label_available":
            for frame in (saved, current):
                if column not in frame:
                    frame[column] = 1  # Legacy runs supervised every diagnosis.
        if column not in saved and column not in current:
            continue
        if column not in saved or column not in current:
            raise ProvenanceError(f"Resume data schema changed for {name}: {column}.")
        previous, present = saved[column], current[column]
        if column in {"target", "diagnosis_index", "diagnosis_label_available", "melanoma_label", "age_approx"}:
            previous = pd.to_numeric(previous, errors="raise").astype("Float64")
            present = pd.to_numeric(present, errors="raise").astype("Float64")
        else:
            previous, present = previous.astype("string"), present.astype("string")
        if not previous.equals(present):
            raise ProvenanceError(f"Resume per-image values changed for {name}: {column}.")
