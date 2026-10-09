from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from prepare_combined_dataset import HIGH_RISK_DIAGNOSES, IMAGE_SUFFIXES, inspect_image


UNKNOWN_DIAGNOSIS = "UNKNOWN"
# These source diagnoses have no valid class in the project's eight-class head.
# Keep their melanoma supervision, but do not teach that they are benign DF/VASC
# or that a cutaneous horn/Paget disease is necessarily SCC.
UNREPRESENTABLE_LABELS = {"ks", "dfsp", "ch", "mpd"}


def default_image_dir() -> Path:
    if os.name == "nt":
        return Path(r"D:\SkinCancerData\DERM12345\train")
    return Path("/mnt/d/SkinCancerData/DERM12345/train")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add the official DERM12345 training split to the deduplicated manifest."
    )
    parser.add_argument(
        "--base-manifest",
        type=Path,
        default=Path("data") / "combined_ham_bcn" / "combined_manifest.csv",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data") / "derm12345" / "derm12345_metadata_train.tab",
    )
    parser.add_argument(
        "--test-metadata",
        type=Path,
        default=Path("data") / "derm12345" / "derm12345_metadata_test.tab",
    )
    parser.add_argument("--image-dir", type=Path, default=default_image_dir())
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data") / "combined_ham_bcn_derm12345",
    )
    parser.add_argument("--hash-workers", type=int, default=2)
    return parser.parse_args()


def index_images(image_dir: Path) -> dict[str, str]:
    if not image_dir.exists():
        raise FileNotFoundError(f"DERM12345 training directory not found: {image_dir}")
    image_index: dict[str, str] = {}
    for path in image_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        key = path.stem.upper()
        if key in image_index:
            raise ValueError(f"Duplicate DERM12345 image ID on disk: {key}")
        image_index[key] = str(path.resolve())
    return image_index


def map_diagnosis(row: pd.Series) -> str:
    super_class = str(row["super_class"]).lower()
    main_class = str(row["main_class_1"]).lower()
    malignancy = str(row["malignancy"]).lower()
    label = str(row["label"]).lower()
    if label in UNREPRESENTABLE_LABELS:
        return UNKNOWN_DIAGNOSIS
    if main_class == "melanoma":
        return "MEL"
    if super_class == "melanocytic":
        return "BKL" if main_class == "lentigo" else "NV"
    if main_class == "keratinocytic":
        if label == "bcc":
            return "BCC"
        if malignancy == "malignant":
            return "SCC"
        if malignancy == "indeterminate":
            return "AK"
        return "BKL"
    if main_class == "fibro_histiocytic":
        return "DF"
    if main_class == "vascular":
        return "VASC"
    raise ValueError(f"Unsupported DERM12345 taxonomy row: {row.to_dict()}")


def correct_derm_taxonomy(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply source taxonomy without replacing raw diagnoses or image identities.

    Accepts either original DERM metadata or its historical manifest columns.
    ``target`` remains the high-risk label here; callers building melanoma
    training data should use ``repair_existing_manifest``.
    """
    output = frame.copy()
    label_column = "derm_label" if "derm_label" in output else "label"
    malignancy_column = (
        "original_malignancy" if "original_malignancy" in output else "malignancy"
    )
    required = {label_column, malignancy_column, "super_class", "main_class_1"}
    missing = required - set(output.columns)
    if missing:
        raise ValueError(f"DERM taxonomy is missing columns: {sorted(missing)}")
    taxonomy = output.rename(
        columns={label_column: "label", malignancy_column: "malignancy"}
    )
    if taxonomy[list({"label", "malignancy", "super_class", "main_class_1"})].isna().any().any():
        raise ValueError("DERM taxonomy has missing labels or malignancy.")
    malignancy = taxonomy["malignancy"].astype(str).str.lower()
    if not malignancy.isin({"benign", "malignant", "indeterminate"}).all():
        raise ValueError("DERM taxonomy has an unsupported malignancy value.")
    output["diagnosis"] = taxonomy.apply(map_diagnosis, axis=1)
    output["diagnosis_label_available"] = output["diagnosis"].ne(UNKNOWN_DIAGNOSIS).astype("int32")
    output["melanoma_label"] = taxonomy["main_class_1"].astype(str).str.lower().eq("melanoma").astype("int32")
    if (output["melanoma_label"].eq(1) & malignancy.ne("malignant")).any():
        raise ValueError("A DERM melanoma diagnosis is not marked malignant.")
    output["high_risk_label"] = malignancy.isin({"malignant", "indeterminate"}).astype("int32")
    output["risk_group"] = malignancy.map({
        "benign": "benign", "malignant": "malignant", "indeterminate": "premalignant"
    })
    output["target"] = output["high_risk_label"]
    output["target_name"] = output["target"].map({0: "benign", 1: "high_risk"})
    return output


def repair_existing_manifest(
    frame: pd.DataFrame, binary_target: str = "melanoma"
) -> pd.DataFrame:
    """Derive corrected labels without rehashing images or changing split IDs.

    Existing SIIM UNKNOWN labels remain binary-only; raw provenance fields are
    preserved. Historical manifests must be written to a new output path.
    """
    if binary_target != "melanoma":
        raise ValueError("This derived-manifest repair supports only melanoma-vs-rest.")
    output = frame.copy()
    if "diagnosis_label_available" not in output:
        output["diagnosis_label_available"] = 1
    supervision = pd.to_numeric(output["diagnosis_label_available"], errors="raise")
    if not supervision.isin([0, 1]).all():
        raise ValueError("Diagnosis supervision flags must contain only 0 or 1.")
    supervised = supervision.eq(1)
    derm_mask = output["source"].eq("DERM12345")
    explicit = output.get("melanoma_label", pd.Series(pd.NA, index=output.index)).copy()
    output["melanoma_label"] = output["diagnosis"].eq("MEL").astype("int32")
    # DERM is repaired from its raw taxonomy below. Its generic high-risk target
    # may correctly be positive for UNKNOWN non-melanoma cancers such as KS.
    unknown_other = ~supervised & ~derm_mask
    use_explicit = unknown_other & explicit.notna()
    if use_explicit.any():
        labels = pd.to_numeric(explicit.loc[use_explicit], errors="raise")
        if not labels.isin([0, 1]).all():
            raise ValueError("Explicit melanoma labels must contain only 0 or 1.")
        output.loc[use_explicit, "melanoma_label"] = labels.astype("int32")
    fallback = unknown_other & explicit.isna()
    if fallback.any():
        # Without an explicit melanoma label, legacy SIIM rows are known negatives.
        if not output.loc[fallback, "target"].eq(0).all():
            raise ValueError("Cannot infer melanoma labels for binary-only positive rows.")
        output.loc[fallback, "melanoma_label"] = 0
    output["high_risk_label"] = pd.Series(pd.NA, index=output.index, dtype="Int64")
    output.loc[supervised, "high_risk_label"] = output.loc[supervised, "diagnosis"].isin(HIGH_RISK_DIAGNOSES).astype(int)
    if derm_mask.any():
        corrected = correct_derm_taxonomy(output.loc[derm_mask])
        for column in ("diagnosis", "diagnosis_label_available", "melanoma_label", "high_risk_label", "risk_group"):
            output.loc[derm_mask, column] = corrected[column]
    output["target"] = output["melanoma_label"].astype("int32")
    output["target_name"] = output["target"].map({0: "not_melanoma", 1: "melanoma"})
    return output


def load_derm(args: argparse.Namespace) -> pd.DataFrame:
    metadata = pd.read_csv(args.metadata, sep="\t")
    required = {
        "image_id",
        "patient_id",
        "image_type",
        "split",
        "super_class",
        "malignancy",
        "main_class_1",
        "main_class_2",
        "sub_class",
        "label",
    }
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"DERM12345 metadata is missing columns: {sorted(missing)}")
    if not metadata["split"].astype(str).str.lower().eq("train").all():
        raise ValueError("DERM12345 training metadata contains non-training rows.")
    if metadata["image_id"].duplicated().any():
        raise ValueError("DERM12345 training metadata contains duplicate image IDs.")

    image_index = index_images(args.image_dir)
    if len(image_index) != len(metadata):
        raise ValueError(
            f"Found {len(image_index)} image files for {len(metadata)} metadata rows."
        )
    if not args.test_metadata.exists():
        raise FileNotFoundError(f"DERM12345 test metadata not found: {args.test_metadata}")
    test_metadata = pd.read_csv(args.test_metadata, sep="\t", usecols=["patient_id"])
    test_patients = set(test_metadata["patient_id"].astype(str))
    overlap_mask = metadata["patient_id"].astype(str).isin(test_patients)
    excluded = metadata.loc[overlap_mask].copy()
    excluded.to_csv(args.output_dir / "excluded_test_patient_overlap.csv", index=False)
    metadata = metadata.loc[~overlap_mask].copy()

    frame = metadata.copy()
    frame["image_path"] = frame["image_id"].astype(str).str.upper().map(image_index)
    missing_images = frame.loc[frame["image_path"].isna(), "image_id"].head(10).tolist()
    if missing_images:
        raise FileNotFoundError(
            f"DERM12345 training images are incomplete; missing examples: {missing_images}"
        )
    archive_labels = frame["image_path"].map(lambda value: Path(value).parent.name.lower())
    label_mismatch = ~archive_labels.eq(frame["label"].astype(str).str.lower())
    if label_mismatch.any():
        examples = frame.loc[label_mismatch, ["image_id", "label", "image_path"]].head(5)
        raise ValueError(f"Archive folder labels disagree with metadata:\n{examples}")
    frame["source"] = "DERM12345"
    frame["source_lesion_id"] = frame["image_id"].astype(str)
    frame["lesion_id"] = "DERM12345::" + frame["source_lesion_id"]
    frame["split_group_id"] = "DERM12345::" + frame["patient_id"].astype(str)
    frame = correct_derm_taxonomy(frame)
    frame["age_approx"] = pd.NA
    frame["sex"] = "unknown"
    frame["anatom_site_general"] = "unknown"
    frame["capture_date"] = pd.NA
    frame["diagnosis_method"] = frame["malignancy"].map(
        lambda value: "histo" if str(value).lower() == "malignant" else "unspecified"
    )
    frame.rename(columns={"label": "derm_label", "malignancy": "original_malignancy"}, inplace=True)
    return frame


def add_integrity(frame: pd.DataFrame, workers: int) -> pd.DataFrame:
    if workers < 1:
        raise ValueError("--hash-workers must be at least 1.")
    jobs = [(path, True) for path in frame["image_path"]]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(inspect_image, jobs))
    output = frame.copy()
    output[["file_size", "sha256", "dhash", "width", "height", "modified_ns"]] = (
        pd.DataFrame(results, index=output.index)
    )
    return output


def main() -> None:
    args = parse_args()
    if not args.base_manifest.exists():
        raise FileNotFoundError(f"Base manifest not found: {args.base_manifest}")
    if not args.metadata.exists():
        raise FileNotFoundError(f"DERM12345 metadata not found: {args.metadata}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base = pd.read_csv(args.base_manifest)
    if "diagnosis_label_available" not in base:
        base["diagnosis_label_available"] = 1
    if "split_group_id" not in base:
        base["split_group_id"] = base["lesion_id"]
    if "patient_id" not in base:
        base["patient_id"] = pd.NA
    derm = add_integrity(load_derm(args), args.hash_workers)
    combined = pd.concat([base, derm], ignore_index=True, sort=False)

    duplicate_mask = combined.duplicated("sha256", keep=False)
    exact_duplicates = combined.loc[duplicate_mask].sort_values(
        ["sha256", "source", "image_id"]
    )
    exact_duplicates.to_csv(args.output_dir / "exact_duplicates.csv", index=False)
    diagnosis_counts = exact_duplicates.groupby("sha256")["diagnosis"].nunique()
    conflicting_hashes = set(diagnosis_counts[diagnosis_counts > 1].index)
    exact_duplicates[exact_duplicates["sha256"].isin(conflicting_hashes)].to_csv(
        args.output_dir / "exact_duplicate_label_conflicts.csv", index=False
    )

    safe = combined[~combined["sha256"].isin(conflicting_hashes)]
    deduplicated = safe.drop_duplicates("sha256", keep="first").copy()
    perceptual = deduplicated[
        deduplicated.duplicated("dhash", keep=False)
    ].sort_values(["dhash", "source", "image_id"])
    perceptual.to_csv(args.output_dir / "perceptual_duplicate_candidates.csv", index=False)

    manifest_path = args.output_dir / "combined_manifest.csv"
    deduplicated.to_csv(manifest_path, index=False)
    summary = {
        "manifest": str(manifest_path.resolve()),
        "base_images": int(len(base)),
        "derm12345_train_images": int(len(derm)),
        "derm12345_train_patients": int(derm["patient_id"].nunique()),
        "test_overlap_patients_excluded": int(
            pd.read_csv(args.output_dir / "excluded_test_patient_overlap.csv")["patient_id"].nunique()
        ),
        "test_overlap_images_excluded": int(
            len(pd.read_csv(args.output_dir / "excluded_test_patient_overlap.csv"))
        ),
        "output_images": int(len(deduplicated)),
        "lesions": int(deduplicated["lesion_id"].nunique()),
        "split_groups": int(deduplicated["split_group_id"].nunique()),
        "same_diagnosis_exact_duplicates_removed": int(len(safe) - len(deduplicated)),
        "conflicting_duplicate_groups": int(len(conflicting_hashes)),
        "conflicting_duplicate_images_removed": int(
            combined["sha256"].isin(conflicting_hashes).sum()
        ),
        "perceptual_candidates_retained": int(len(perceptual)),
        "source_counts": {
            str(key): int(value)
            for key, value in deduplicated["source"].value_counts().items()
        },
        "diagnosis_counts": {
            str(key): int(value)
            for key, value in deduplicated["diagnosis"].value_counts().sort_index().items()
        },
        "melanoma_images": int(deduplicated["diagnosis"].eq("MEL").sum()),
    }
    (args.output_dir / "manifest_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
