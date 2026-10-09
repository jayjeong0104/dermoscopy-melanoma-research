"""Read-only split, provenance, and warm-start ancestry audit; no model inference."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
BASELINE = ROOT / "artifacts/melanoma_v1/run_20260822_111527"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def local_path(value: str) -> Path:
    """Locate this repository's historical Windows/WSL artifact paths."""
    normalized = value.replace("\\", "/")
    marker = "/archive/"
    if marker in normalized:
        return ROOT / normalized.split(marker, 1)[1]
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def summary(frame: pd.DataFrame) -> dict:
    lesions = frame.drop_duplicates("lesion_id")
    patient = frame.get("patient_id", pd.Series(pd.NA, index=frame.index)).astype("string")
    known = patient.notna() & ~patient.str.strip().str.lower().isin({"", "nan", "unknown", "none"})
    return {
        "images": len(frame), "lesions": len(lesions),
        "split_groups": int(frame.split_group_id.nunique()),
        "melanoma_images": int(frame.target.sum()),
        "melanoma_lesions": int(lesions.target.sum()),
        "image_melanoma_prevalence": float(frame.target.mean()) if len(frame) else None,
        "lesion_melanoma_prevalence": float(lesions.target.mean()) if len(lesions) else None,
        "images_with_patient_id": int(known.sum()),
        "patient_id_coverage": float(known.mean()) if len(frame) else None,
        "known_patients_with_source_namespace": int(
            (frame.loc[known, "source"].astype(str) + "::" + patient[known]).nunique()
        ),
    }


def breakdown(frame: pd.DataFrame) -> list[dict]:
    return [
        {"split": str(role), "source": str(source), **summary(group)}
        for (role, source), group in frame.groupby(["audit_split", "source"], sort=True)
    ]


def overlap_breakdown(current: pd.DataFrame, prior: pd.DataFrame) -> dict:
    result = {}
    for key in ("sha256", "lesion_id", "split_group_id"):
        if key not in prior:
            result[key] = {"unavailable": True}
            continue
        exposed = current.loc[current[key].isin(set(prior[key].dropna()))]
        result[key] = {
            **summary(exposed),
            "source_and_target_images": [
                {"source": str(source), "melanoma_target": int(target), "images": int(count)}
                for (source, target), count in exposed.groupby(["source", "target"]).size().items()
            ],
        }
    return result


def ancestry_audit(run: Path, frames: dict[str, pd.DataFrame]) -> dict:
    seen, ancestors, missing = set(), [], []
    prior_frames = {"train": [], "val": []}
    config_path = run / "run_config.json"
    while config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8-sig"))
        parent_model = config.get("warm_start_model")
        if not parent_model:
            break
        parent = local_path(str(parent_model)).parent
        if str(parent) in seen:
            raise ValueError("Cycle in warm-start model ancestry.")
        seen.add(str(parent))
        entry = {"run": str(parent), "model": str(parent_model),
                 "referenced_by_config": str(config_path),
                 "referenced_by_config_sha256": digest(config_path), "exposures": {}}
        for role in ("train", "val"):
            path = parent / f"{role}_split.csv"
            if not path.exists():
                missing.append(str(path))
                continue
            prior = pd.read_csv(path, low_memory=False)
            if "split_group_id" not in prior and "lesion_id" in prior:
                prior["split_group_id"] = prior["lesion_id"]
            prior_frames[role].append(prior)
            entry["exposures"][role] = {
                "split_file": str(path), "sha256": digest(path), "images": len(prior),
                "current_holdout_overlap": {
                    held: overlap_breakdown(frames[held], prior) for held in ("val", "test")
                },
            }
        ancestors.append(entry)
        config_path = parent / "run_config.json"
        if not config_path.exists():
            missing.append(str(config_path))
    union = {}
    for role, prior in prior_frames.items():
        if prior:
            union[role] = {
                held: overlap_breakdown(frames[held], pd.concat(prior, ignore_index=True))
                for held in ("val", "test")
            }
    return {
        "ancestors": ancestors, "union_of_ancestor_exposures": union,
        "missing_lineage_files": missing,
        "interpretation": (
            "Ancestor training exposure is holdout leakage even when the current risk head was reset. "
            "Ancestor validation exposure means the encoder/checkpoint was selected using those cases. "
            "Within-run split disjointness does not establish independence from either exposure."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=BASELINE)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/pipeline_review_20261005")
    args = parser.parse_args()
    frames = {}
    inputs = []
    for role in ("train", "val", "test"):
        path = args.run_dir / f"{role}_split.csv"
        frames[role] = pd.read_csv(path, low_memory=False).assign(audit_split=role)
        inputs.append({"path": str(path), "sha256": digest(path)})
    frame = pd.concat(frames.values(), ignore_index=True)
    checks = {}
    for column in ("sha256", "image_id", "lesion_id", "split_group_id", "dhash"):
        grouped = frame.groupby(column).agg(
            splits=("audit_split", "nunique"), images=("audit_split", "size"),
            melanoma_labels=("target", "nunique"),
        )
        crossing = grouped[grouped.splits.gt(1)]
        checks[column] = {
            "cross_split_groups": len(crossing), "images_in_cross_split_groups": int(crossing.images.sum()),
            "cross_split_groups_with_conflicting_targets": int(crossing.melanoma_labels.gt(1).sum()),
            "missing_values": int(frame[column].isna().sum()),
        }
    candidates = []
    crossing_hashes = frame.groupby("dhash").audit_split.nunique().loc[lambda v: v.gt(1)].index
    for value, group in frame.loc[frame.dhash.isin(crossing_hashes)].groupby("dhash"):
        for (_, left), (_, right) in itertools.combinations(group.iterrows(), 2):
            if left.audit_split == right.audit_split:
                continue
            candidates.append({
                "dhash": value, "status": "unreviewed_similarity_candidate_not_duplicate_proof",
                **{f"{key}_{side}": row[key] for side, row in (("left", left), ("right", right))
                   for key in ("image_id", "audit_split", "source", "diagnosis", "target", "lesion_id", "image_path")},
            })
    derm = frame.loc[frame.source.eq("DERM12345")]
    inconsistent = derm.loc[derm.original_malignancy.eq("malignant") & derm.risk_group.eq("benign")]
    report = {
        "script_sha256": digest(Path(__file__)), "run_dir": str(args.run_dir),
        "inputs": inputs, "totals": summary(frame), "by_split_source": breakdown(frame),
        "by_split": {role: summary(subset) for role, subset in frames.items()},
        "within_run_cross_split_checks": checks,
        "source_label_flaws": {
            "original_malignant_mapped_benign_images": len(inconsistent),
            "source_labels": inconsistent.derm_label.value_counts().to_dict(),
            "image_ids": inconsistent.image_id.tolist(),
            "interpretation": "Wrong broad high-risk labels/auxiliary ontology; these are correctly negative for the distinct melanoma-vs-rest endpoint.",
        },
        "warm_start_ancestry": ancestry_audit(args.run_dir, frames),
        "perceptual_review": {"candidate_pairs": len(candidates), "method": "Identical 64-bit dHash across distinct current splits", "images_decoded_by_this_script": 0},
        "limitations": [
            "No new model inference, relabeling, source image edits, or automatic duplicate removals.",
            "Stored SHA values are audited; image bytes are not rehashed by this script.",
            "dHash equality is only a review queue and does not prove duplicate identity.",
            "HAM10000 and BCN20000 patient IDs are absent, so lesion separation cannot establish patient independence.",
            "This audit uses archived holdout identities and source labels, not new holdout predictions.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(candidates).to_csv(args.output_dir / "near_duplicate_candidates.csv", index=False)
    (args.output_dir / "data_audit.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"report": str(args.output_dir / "data_audit.json"), "images": len(frame), "ancestor_count": len(report["warm_start_ancestry"]["ancestors"]), "candidate_pairs": len(candidates)}))


if __name__ == "__main__":
    main()
