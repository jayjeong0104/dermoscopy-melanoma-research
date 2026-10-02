from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from prepare_combined_dataset import (
    IMAGE_SUFFIXES,
    add_integrity_columns,
    load_integrity_cache,
)


SOURCE = "SIIM_ISIC_2020"
UNKNOWN_DIAGNOSIS = "UNKNOWN"
SIIM_DIAGNOSIS_MAP = {
    "melanoma": "MEL",
    "nevus": "NV",
    "seborrheic keratosis": "BKL",
    "lentigo nos": "BKL",
    "lichenoid keratosis": "BKL",
    "solar lentigo": "BKL",
}


def default_image_dir() -> Path:
    if os.name == "nt":
        return Path(r"D:\SkinCancerData\SIIM_ISIC_2020\train")
    return Path("/mnt/d/SkinCancerData/SIIM_ISIC_2020/train")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Add SIIM-ISIC 2020 to the existing exact-deduplicated dermoscopy manifest."
        )
    )
    parser.add_argument(
        "--base-manifest",
        type=Path,
        default=Path("data")
        / "combined_ham_bcn_derm12345"
        / "combined_manifest.csv",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data")
        / "siim_isic2020"
        / "ISIC_2020_Training_GroundTruth_v2.csv",
    )
    parser.add_argument(
        "--official-duplicates",
        type=Path,
        default=Path("data")
        / "siim_isic2020"
        / "ISIC_2020_Training_Duplicates.csv",
    )
    parser.add_argument("--image-dir", type=Path, default=default_image_dir())
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data") / "combined_ham_bcn_derm12345_siim2020",
    )
    parser.add_argument(
        "--hash-workers",
        type=int,
        default=max(1, min(4, os.cpu_count() or 1)),
    )
    return parser.parse_args()


def map_diagnosis(value: object) -> tuple[str, int]:
    normalized = str(value).strip().lower()
    diagnosis = SIIM_DIAGNOSIS_MAP.get(normalized)
    if diagnosis is None:
        return UNKNOWN_DIAGNOSIS, 0
    return diagnosis, 1


def index_images(image_dir: Path) -> dict[str, str]:
    if not image_dir.exists():
        raise FileNotFoundError(f"SIIM-ISIC 2020 image directory not found: {image_dir}")
    image_index: dict[str, str] = {}
    for path in image_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        key = path.stem.upper()
        if key in image_index:
            raise ValueError(f"Duplicate SIIM image ID on disk: {key}")
        image_index[key] = str(path.resolve())
    return image_index


def validate_official_duplicate_metadata(
    metadata: pd.DataFrame, duplicate_pairs: pd.DataFrame
) -> None:
    required = {"image_name_1", "image_name_2"}
    missing = required - set(duplicate_pairs.columns)
    if missing:
        raise ValueError(f"Official duplicate list is missing columns: {sorted(missing)}")
    lookup = metadata.set_index("image_name")
    pair_ids = set(duplicate_pairs["image_name_1"]) | set(
        duplicate_pairs["image_name_2"]
    )
    missing_ids = sorted(pair_ids - set(lookup.index))
    if missing_ids:
        raise ValueError(f"Official duplicate IDs are absent from metadata: {missing_ids[:5]}")
    left = lookup.loc[duplicate_pairs["image_name_1"]].reset_index(drop=True)
    right = lookup.loc[duplicate_pairs["image_name_2"]].reset_index(drop=True)
    if not left["patient_id"].astype(str).equals(right["patient_id"].astype(str)):
        raise ValueError("An official duplicate pair crosses patient IDs.")
    if not left["target"].astype(int).equals(right["target"].astype(int)):
        raise ValueError("An official duplicate pair has conflicting melanoma labels.")


def load_siim(
    metadata_path: Path,
    duplicate_path: Path,
    image_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata = pd.read_csv(metadata_path)
    required = {
        "image_name",
        "patient_id",
        "lesion_id",
        "sex",
        "age_approx",
        "anatom_site_general_challenge",
        "diagnosis",
        "benign_malignant",
        "target",
    }
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"SIIM metadata is missing columns: {sorted(missing)}")
    if metadata["image_name"].duplicated().any():
        raise ValueError("SIIM metadata contains duplicate image IDs.")
    if metadata[["patient_id", "lesion_id"]].isna().any().any():
        raise ValueError("SIIM metadata contains missing patient or lesion IDs.")
    metadata["target"] = pd.to_numeric(metadata["target"], errors="raise").astype("int32")
    if not metadata["target"].isin([0, 1]).all():
        raise ValueError("SIIM target must contain only 0 or 1.")
    expected_target = metadata["diagnosis"].astype(str).str.lower().eq("melanoma").astype("int32")
    if not np.array_equal(metadata["target"].to_numpy(), expected_target.to_numpy()):
        raise ValueError("SIIM melanoma targets disagree with diagnosis labels.")

    duplicate_pairs = pd.read_csv(duplicate_path)
    validate_official_duplicate_metadata(metadata, duplicate_pairs)
    image_index = index_images(image_dir)
    expected_ids = set(metadata["image_name"].astype(str).str.upper())
    if set(image_index) != expected_ids:
        missing_images = sorted(expected_ids - set(image_index))
        extra_images = sorted(set(image_index) - expected_ids)
        raise FileNotFoundError(
            "SIIM image files do not exactly match metadata. "
            f"Missing sample: {missing_images[:5]}; extra sample: {extra_images[:5]}"
        )

    frame = metadata.copy()
    mapped = frame["diagnosis"].map(map_diagnosis)
    frame["diagnosis"] = mapped.map(lambda value: value[0])
    frame["diagnosis_label_available"] = mapped.map(lambda value: value[1]).astype("int32")
    frame["image_path"] = frame["image_name"].astype(str).str.upper().map(image_index)
    frame["image_id"] = frame["image_name"].astype(str)
    frame["source"] = SOURCE
    frame["source_lesion_id"] = frame["lesion_id"].astype(str)
    frame["lesion_id"] = SOURCE + "::" + frame["source_lesion_id"]
    frame["split_group_id"] = SOURCE + "::" + frame["patient_id"].astype(str)
    frame["patient_id"] = SOURCE + "::" + frame["patient_id"].astype(str)
    frame["target_name"] = frame["target"].map({0: "not_melanoma", 1: "melanoma"})
    frame["risk_group"] = np.where(frame["target"].eq(1), "malignant", "benign")
    frame["anatom_site_general"] = frame["anatom_site_general_challenge"]
    frame["capture_date"] = pd.NA
    frame["diagnosis_method"] = np.where(
        frame["target"].eq(1), "histopathology", "challenge_ground_truth"
    )
    frame["image_type"] = "dermoscopic"
    frame["copyright-license"] = "CC BY-NC 4.0"
    return frame, duplicate_pairs


def melanoma_targets(frame: pd.DataFrame) -> pd.Series:
    supervised = frame.get(
        "diagnosis_label_available",
        pd.Series(1, index=frame.index, dtype="int32"),
    ).astype(int)
    result = frame["diagnosis"].astype(str).str.upper().eq("MEL").astype("int32")
    result.loc[supervised.eq(0)] = frame.loc[supervised.eq(0), "target"].astype("int32")
    return result


def official_duplicate_audit(
    siim: pd.DataFrame, duplicate_pairs: pd.DataFrame
) -> pd.DataFrame:
    lookup = siim.set_index("image_id")
    audit = duplicate_pairs.copy()
    for side in (1, 2):
        selected = lookup.loc[audit[f"image_name_{side}"]]
        audit[f"sha256_{side}"] = selected["sha256"].to_numpy()
        audit[f"patient_id_{side}"] = selected["patient_id"].to_numpy()
        audit[f"target_{side}"] = selected["target"].to_numpy()
    audit["byte_identical"] = audit["sha256_1"].eq(audit["sha256_2"])
    audit["same_patient"] = audit["patient_id_1"].eq(audit["patient_id_2"])
    audit["same_target"] = audit["target_1"].eq(audit["target_2"])
    return audit


def main() -> None:
    args = parse_args()
    for path in (args.base_manifest, args.metadata, args.official_duplicates):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.hash_workers < 1:
        raise ValueError("--hash-workers must be at least 1.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base = pd.read_csv(args.base_manifest)
    if "diagnosis_label_available" not in base:
        base["diagnosis_label_available"] = 1
    siim, duplicate_pairs = load_siim(
        args.metadata, args.official_duplicates, args.image_dir
    )
    cache_path = args.output_dir / "siim_integrity_cache.csv"
    cached = load_integrity_cache(siim, cache_path)
    if cached is None:
        siim = add_integrity_columns(siim, args.hash_workers, include_dhash=True)
        siim.to_csv(cache_path, index=False)
    else:
        siim = cached

    official_audit = official_duplicate_audit(siim, duplicate_pairs)
    official_audit.to_csv(args.output_dir / "siim_official_duplicate_audit.csv", index=False)

    combined = pd.concat([base, siim], ignore_index=True, sort=False)
    combined["melanoma_target"] = melanoma_targets(combined)
    duplicate_mask = combined.duplicated("sha256", keep=False)
    exact_duplicates = combined.loc[duplicate_mask].sort_values(
        ["sha256", "source", "image_id"]
    )
    target_conflicts = exact_duplicates.groupby("sha256")["melanoma_target"].nunique()
    conflicting_hashes = set(target_conflicts[target_conflicts > 1].index)
    supervised_duplicates = exact_duplicates[
        exact_duplicates["diagnosis_label_available"].eq(1)
    ]
    diagnosis_conflicts = supervised_duplicates.groupby("sha256")["diagnosis"].nunique()
    conflicting_hashes.update(diagnosis_conflicts[diagnosis_conflicts > 1].index)
    conflicts = exact_duplicates[exact_duplicates["sha256"].isin(conflicting_hashes)]
    conflicts.to_csv(args.output_dir / "exact_duplicate_label_conflicts.csv", index=False)
    if conflicting_hashes:
        raise ValueError(
            f"Found {len(conflicting_hashes)} exact duplicate groups with conflicting labels."
        )

    exact_duplicates = exact_duplicates.copy()
    exact_duplicates["kept"] = ~exact_duplicates.duplicated("sha256", keep="first")
    exact_duplicates.to_csv(args.output_dir / "exact_duplicates.csv", index=False)
    deduplicated = combined.drop_duplicates("sha256", keep="first").copy()

    perceptual = deduplicated[
        deduplicated.duplicated("dhash", keep=False)
    ].sort_values(["dhash", "source", "image_id"])
    perceptual.to_csv(args.output_dir / "perceptual_duplicate_candidates_retained.csv", index=False)

    manifest_path = args.output_dir / "combined_manifest.csv"
    deduplicated.to_csv(manifest_path, index=False)
    summary = {
        "manifest": str(manifest_path.resolve()),
        "base_images": int(len(base)),
        "siim_input_images": int(len(siim)),
        "siim_patients": int(siim["patient_id"].nunique()),
        "siim_lesions": int(siim["lesion_id"].nunique()),
        "siim_melanoma_images": int(siim["target"].sum()),
        "siim_diagnosis_supervised_images": int(siim["diagnosis_label_available"].sum()),
        "siim_binary_only_images": int(siim["diagnosis_label_available"].eq(0).sum()),
        "official_duplicate_pairs": int(len(official_audit)),
        "official_byte_identical_pairs": int(official_audit["byte_identical"].sum()),
        "exact_duplicate_rows": int(len(exact_duplicates)),
        "exact_duplicate_rows_removed": int(len(combined) - len(deduplicated)),
        "perceptual_candidates_retained": int(len(perceptual)),
        "output_images": int(len(deduplicated)),
        "output_lesions": int(deduplicated["lesion_id"].nunique()),
        "output_split_groups": int(deduplicated["split_group_id"].nunique()),
        "source_counts": {
            str(key): int(value)
            for key, value in deduplicated["source"].value_counts().items()
        },
        "melanoma_images": int(deduplicated["melanoma_target"].sum()),
    }
    (args.output_dir / "manifest_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
