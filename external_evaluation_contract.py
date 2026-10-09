"""Bind external evaluation to a saved inference recipe and local model lineage."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from training_provenance import portable_path


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_inference_binding(run_dir, model_path, finalization):
    run_dir, model_path = Path(run_dir), Path(model_path)
    if finalization.get("status") != "complete":
        raise ValueError("External evaluation requires completed finalization.")
    recorded = finalization.get("artifact_sha256", {})
    expected = recorded.get("best_model.keras")
    actual = file_hash(model_path)
    if expected and actual != expected:
        raise ValueError("Checkpoint differs from the model bound to saved calibration/thresholds.")
    if not expected and model_path.resolve() != (run_dir / "best_model.keras").resolve():
        raise ValueError("No saved checkpoint binding: a non-best model cannot reuse this calibration/threshold.")
    verified = {}
    for name in ("run_config.json", "metadata_schema.json", "model_selection.json", "train_split.csv", "val_split.csv", "calibration.json",
                 "selected_threshold.json", "probability_blend.json"):
        if name in recorded:
            if file_hash(run_dir / name) != recorded[name]:
                raise ValueError(f"Saved inference artifact changed: {name}")
            verified[name] = recorded[name]
    return {"checkpoint_sha256": actual, "checkpoint_hash_verified": expected is not None,
        "verified_inference_artifacts": verified,
        "limitation": "Legacy unrecorded artifact hashes cannot be retrospectively verified; passing this check does not establish cohort independence."}


def recorded_exposure_manifests(model_path, project_root):
    """Return actual train/validation manifests for this model and all warm-start parents.

    Validation is included because it selects checkpoints. Never substitute a
    generic dataset manifest when a run's own recorded lineage is available.
    """
    path = portable_path(model_path, Path(project_root)).resolve()
    visited, manifests, lineage = set(), [], []
    while True:
        if path in visited:
            raise ValueError("Cyclic checkpoint ancestry in external overlap audit.")
        visited.add(path)
        if not path.is_file():
            raise ValueError(f"Missing ancestor checkpoint: {path}")
        config_path = path.parent / "run_config.json"
        if not config_path.is_file():
            raise ValueError(f"Missing ancestor configuration: {config_path}")
        config = json.loads(config_path.read_text(encoding="utf-8-sig"))
        finalization_path = path.parent / "finalization_status.json"
        recorded = (json.loads(finalization_path.read_text(encoding="utf-8-sig")).get("artifact_sha256", {})
            if finalization_path.exists() else {})
        if "run_config.json" in recorded and file_hash(config_path) != recorded["run_config.json"]:
            raise ValueError(f"Ancestor configuration changed: {config_path}")
        if path.name in recorded and file_hash(path) != recorded[path.name]:
            raise ValueError(f"Ancestor checkpoint changed: {path}")
        item = {"model": str(path), "model_sha256": file_hash(path),
            "config_sha256": file_hash(config_path), "exposure_manifests": []}
        for split in ("train", "val"):
            manifest = path.parent / f"{split}_split.csv"
            if not manifest.is_file():
                raise ValueError(f"Missing recorded exposure manifest: {manifest}")
            if manifest.name in recorded and file_hash(manifest) != recorded[manifest.name]:
                raise ValueError(f"Ancestor exposure manifest changed: {manifest}")
            manifests.append(manifest)
            item["exposure_manifests"].append({"path": str(manifest), "split": split, "sha256": file_hash(manifest)})
        lineage.append(item)
        parent = config.get("warm_start_model")
        if not parent:
            if config.get("weights") not in {"imagenet", "none"}:
                raise ValueError("Unknown root checkpoint provenance; cannot certify local overlap audit.")
            break
        path = portable_path(parent, Path(project_root)).resolve()
    return manifests, {"local_lineage": lineage, "root_initialization": config["weights"],
        "audit_scope": "Selected model and recursive local ancestors' train and validation identities/hashes.",
        "limitation": "Recorded local exposure only. Private pretraining, unavailable cross-dataset patient linkage, and historical reuse of external evaluations remain unresolved."}
