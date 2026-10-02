"""CPU-only audit of completed prediction caches. Never modifies historical reports."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re

import numpy as np
import pandas as pd

from compare_model_runs import file_hash, write_report
from operating_policy import DIRECT_WEIGHT, crossfit_policy, policy_folds, promotion_gate

ROOT = Path(__file__).resolve().parent
EXPERIMENT = ROOT / "artifacts/targeted_recovery_20260907_v1"


def local_path(value: str) -> Path:
    if os.name == "nt" and re.match(r"^/mnt/[a-z]/", value):
        return Path(value[5].upper() + ":/" + value[7:])
    return Path(value)


def wsl_path(value: str) -> str:
    match = re.match(r"^([A-Za-z]):[\\/](.*)$", value)
    if not match:
        return value
    return "/mnt/" + match[1].lower() + "/" + match[2].replace("\\", "/")


def cached_lesions(experiment: Path, name: str) -> tuple[pd.DataFrame, dict]:
    report = json.loads((experiment / "validation_report.json").read_text(encoding="utf-8"))
    result = report["results"][name]
    run = local_path(result["run_dir"])
    model = run / result["model_name"]
    if file_hash(model) != result["model_sha256"]:
        raise ValueError("Model differs from completed experiment.")
    frame = pd.read_csv(run / "val_split.csv", dtype="string", low_memory=False)
    schema = json.loads((run / "metadata_schema.json").read_text(encoding="utf-8"))
    cache = experiment / "prediction_cache" / f"{name}_val_d4_v2_tta1.npz"
    metadata = json.loads(cache.with_suffix(".json").read_text(encoding="utf-8"))
    ordered = frame[["source", "image_id", "sha256"]].copy()
    ordered["runtime_image_path"] = frame.image_path.map(wsl_path)
    digest = hashlib.sha256(ordered.astype("string").to_csv(index=False, lineterminator="\n").encode("utf-8"))
    digest.update(json.dumps(schema, sort_keys=True).encode("utf-8"))
    expected = {"cache_version": 2, "input_sha256": digest.hexdigest(), "tta_scheme": "d4_v2",
                "model_sha256": result["model_sha256"], "row_count": len(frame), "image_size": 300,
                "transforms": 1, "masked_metadata": True}
    if metadata != expected:
        raise ValueError(f"Cache provenance mismatch for {name}.")
    with np.load(cache, allow_pickle=False) as values:
        direct, diagnosis = values["direct"], values["diagnosis"]
    if direct.shape != (len(frame),) or diagnosis.shape != (len(frame), 8):
        raise ValueError("Invalid cached prediction shapes.")
    for values in (direct, diagnosis):
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError("Invalid cached probabilities.")
    frame["target"] = frame.target.astype(int)
    clipped = np.clip(direct.astype(float), 1e-6, 1 - 1e-6)
    frame["direct_logit"] = np.log(clipped / (1 - clipped))
    frame["diagnosis_probability"] = diagnosis[:, 4]
    grouped = frame.groupby(["source", "lesion_id"], sort=True)
    if grouped[["target", "split_group_id", "diagnosis"]].nunique(dropna=False).ne(1).any().any():
        raise ValueError("Conflicting validation lesion identities/labels.")
    lesions = grouped[["target", "split_group_id", "diagnosis"]].first()
    lesions["direct_probability"] = (1 / (1 + np.exp(-grouped.direct_logit.mean()))).astype("float32")
    lesions["diagnosis_probability"] = grouped.diagnosis_probability.mean().astype("float32")
    original = pd.read_csv(experiment / f"{name}_validation_predictions.csv",
                           dtype={"source": "string", "lesion_id": "string"}).set_index(["source", "lesion_id"]).sort_index()
    weight = result["blend"]["direct_weight"]
    reproduced = weight * lesions.direct_probability + (1 - weight) * lesions.diagnosis_probability
    if (not original.index.equals(lesions.index) or not np.array_equal(original.target, lesions.target)
            or not np.allclose(original.probability, reproduced, rtol=0, atol=2e-7)):
        raise ValueError("Cached components do not reproduce the completed validation predictions.")
    return lesions.reset_index(), {"model_sha256": file_hash(model), "cache_sha256": file_hash(cache),
                                  "validation_split_sha256": file_hash(run / "val_split.csv")}


def audit(experiment: Path, output: Path) -> dict:
    status = json.loads((experiment / "status.json").read_text(encoding="utf-8"))
    if status["status"] != "complete":
        raise ValueError("Only completed experiments can be audited.")
    report = json.loads((experiment / "validation_report.json").read_text(encoding="utf-8"))
    results, baseline, folds, reference = {}, None, None, None
    output.mkdir(parents=True, exist_ok=True)
    for name in report["results"]:
        lesions, provenance = cached_lesions(experiment, name)
        identity = lesions[["source", "lesion_id", "split_group_id", "target"]]
        if reference is None:
            if name != "baseline":
                raise ValueError("Baseline must be evaluated first.")
            reference, folds = identity, policy_folds(lesions)
            identity.assign(policy_fold=folds).to_csv(output / "development_policy_folds.csv", index=False)
        elif not identity.equals(reference):
            raise ValueError("Candidate validation identities differ from baseline.")
        values = DIRECT_WEIGHT * lesions.direct_probability + (1 - DIRECT_WEIGHT) * lesions.diagnosis_probability
        metrics, predictions = crossfit_policy(lesions, values.to_numpy(), folds=folds)
        predictions.to_csv(output / f"{name}_crossfit_predictions.csv", index=False)
        if baseline is None:
            baseline = metrics
        results[name] = {"metrics": metrics, "provenance": provenance,
                         "new_policy_gate": promotion_gate(baseline, metrics)}
        print(f"{name}: crossfit sensitivity={metrics['recall']:.4f}, specificity={metrics['specificity']:.4f}, "
              f"AP={metrics['pr_auc']:.4f}, false alarms={metrics['false_positives']}", flush=True)
    summary = {"results": results, "fixed_direct_weight": DIRECT_WEIGHT,
               "purpose": "Retrospective development diagnostic, not a replacement for the completed experiment's failed result.",
               "source_report_sha256": file_hash(experiment / "validation_report.json"),
               "test_inference": False, "automatic_promotion": False}
    write_report(output / "operating_policy_audit.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, default=EXPERIMENT)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    audit(arguments.experiment_dir.resolve(), arguments.output_dir.resolve())
