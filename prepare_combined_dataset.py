from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps


HAM_DIAGNOSIS_MAP = {
    "akiec": "AK",
    "bcc": "BCC",
    "bkl": "BKL",
    "df": "DF",
    "mel": "MEL",
    "nv": "NV",
    "vasc": "VASC",
}
BCN_DIAGNOSES = {"AK", "BCC", "BKL", "DF", "MEL", "NV", "SCC", "VASC"}
MALIGNANT_DIAGNOSES = {"BCC", "MEL", "SCC"}
PREMALIGNANT_DIAGNOSES = {"AK"}
HIGH_RISK_DIAGNOSES = MALIGNANT_DIAGNOSES | PREMALIGNANT_DIAGNOSES
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a validated, deduplicated HAM10000 + BCN20000 manifest."
    )
    parser.add_argument(
        "--ham-metadata",
        type=Path,
        default=Path("HAM10000_metadata.csv"),
    )
    parser.add_argument(
        "--ham-image-dirs",
        nargs="+",
        type=Path,
        default=[Path("HAM10000_images_part_1"), Path("HAM10000_images_part_2")],
    )
    parser.add_argument(
        "--bcn-metadata",
        type=Path,
        default=Path(r"D:\SkinCancerData\BCN20000\bcn_20k_train.csv"),
    )
    parser.add_argument(
        "--bcn-image-dir",
        type=Path,
        default=Path(r"D:\SkinCancerData\BCN20000\bcn_20k_train"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data") / "combined_ham_bcn",
    )
    parser.add_argument(
        "--hash-workers",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
        help="Parallel workers used for integrity and perceptual hashing.",
    )
    parser.add_argument(
        "--skip-perceptual-hash",
        action="store_true",
        help="Skip dHash generation. SHA-256 exact duplicate detection is always performed.",
    )
    return parser.parse_args()


def validate_inputs(args: argparse.Namespace) -> None:
    required_paths = [args.ham_metadata, args.bcn_metadata, args.bcn_image_dir, *args.ham_image_dirs]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required dataset paths: " + ", ".join(missing))
    if args.hash_workers < 1:
        raise ValueError("--hash-workers must be at least 1.")


def index_images(image_dirs: list[Path], ignore_copy_suffix: bool = False) -> dict[str, str]:
    index: dict[str, str] = {}
    for image_dir in image_dirs:
        for path in image_dir.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            if ignore_copy_suffix and path.stem.endswith(" (2)"):
                continue
            key = path.name.lower()
            if key in index:
                raise ValueError(f"Multiple primary images have the same filename: {path.name}")
            index[key] = str(path.resolve())
    return index


def load_ham(metadata_path: Path, image_dirs: list[Path]) -> pd.DataFrame:
    metadata = pd.read_csv(metadata_path)
    required = {"lesion_id", "image_id", "dx", "dx_type", "age", "sex", "localization"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"HAM10000 metadata is missing columns: {sorted(missing)}")

    image_index = index_images(image_dirs, ignore_copy_suffix=True)
    frame = metadata.copy()
    frame["diagnosis"] = frame["dx"].map(HAM_DIAGNOSIS_MAP)
    if frame["diagnosis"].isna().any():
        unknown = sorted(frame.loc[frame["diagnosis"].isna(), "dx"].unique())
        raise ValueError(f"Unmapped HAM10000 diagnoses: {unknown}")

    frame["image_path"] = frame["image_id"].map(
        lambda image_id: image_index.get(f"{image_id}.jpg".lower())
    )
    missing_images = frame.loc[frame["image_path"].isna(), "image_id"].tolist()
    if missing_images:
        raise FileNotFoundError(
            f"Missing {len(missing_images)} HAM10000 images. Sample: {missing_images[:5]}"
        )

    frame["source"] = "HAM10000"
    frame["source_lesion_id"] = frame["lesion_id"].astype(str)
    frame["lesion_id"] = "HAM10000::" + frame["source_lesion_id"]
    frame["age_approx"] = frame["age"]
    frame["anatom_site_general"] = frame["localization"]
    frame["diagnosis_method"] = frame["dx_type"]
    return frame


def load_bcn(metadata_path: Path, image_dir: Path) -> pd.DataFrame:
    metadata = pd.read_csv(metadata_path)
    required = {"bcn_filename", "lesion_id", "diagnosis"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"BCN20000 metadata is missing columns: {sorted(missing)}")

    frame = metadata.copy()
    frame["diagnosis"] = frame["diagnosis"].astype(str).str.upper()
    unknown = sorted(set(frame["diagnosis"]) - BCN_DIAGNOSES)
    if unknown:
        raise ValueError(f"Unmapped BCN20000 diagnoses: {unknown}")

    image_index = index_images([image_dir])
    frame["image_path"] = frame["bcn_filename"].map(
        lambda filename: image_index.get(str(filename).lower())
    )
    missing_images = frame.loc[frame["image_path"].isna(), "bcn_filename"].tolist()
    if missing_images:
        raise FileNotFoundError(
            f"Missing {len(missing_images)} BCN20000 images. Sample: {missing_images[:5]}"
        )

    frame["image_id"] = frame["bcn_filename"].map(lambda value: Path(str(value)).stem)
    frame["source"] = "BCN20000"
    frame["source_lesion_id"] = frame["lesion_id"].astype(str)
    frame["lesion_id"] = "BCN20000::" + frame["source_lesion_id"]
    frame["dx"] = frame["diagnosis"]
    frame["diagnosis_method"] = "unspecified"
    return frame


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    normalized["target"] = normalized["diagnosis"].isin(HIGH_RISK_DIAGNOSES).astype("int32")
    normalized["target_name"] = normalized["target"].map({0: "benign", 1: "high_risk"})
    normalized["risk_group"] = "benign"
    normalized.loc[
        normalized["diagnosis"].isin(PREMALIGNANT_DIAGNOSES), "risk_group"
    ] = "premalignant"
    normalized.loc[normalized["diagnosis"].isin(MALIGNANT_DIAGNOSES), "risk_group"] = (
        "malignant"
    )

    optional_columns = [
        "age_approx",
        "sex",
        "anatom_site_general",
        "capture_date",
        "diagnosis_method",
    ]
    for column in optional_columns:
        if column not in normalized:
            normalized[column] = pd.NA

    columns = [
        "source",
        "image_id",
        "lesion_id",
        "source_lesion_id",
        "image_path",
        "diagnosis",
        "risk_group",
        "target",
        "target_name",
        "age_approx",
        "sex",
        "anatom_site_general",
        "capture_date",
        "diagnosis_method",
    ]
    return normalized[columns]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def difference_hash_image(image: Image.Image) -> str:
    grayscale = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = list(grayscale.getdata())
    bits = []
    for row in range(8):
        offset = row * 9
        bits.extend(pixels[offset + col] > pixels[offset + col + 1] for col in range(8))
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def difference_hash(path: Path) -> str:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        image.load()
        return difference_hash_image(image)


def inspect_image(item: tuple[str, bool]) -> tuple[int, str, str, int, int, int]:
    path_text, include_dhash = item
    path = Path(path_text)
    data = path.read_bytes()
    with Image.open(io.BytesIO(data)) as image:
        image = ImageOps.exif_transpose(image)
        image.load()
        width, height = image.size
        dhash = difference_hash_image(image) if include_dhash else ""
    stat = path.stat()
    return (
        stat.st_size,
        hashlib.sha256(data).hexdigest(),
        dhash,
        width,
        height,
        stat.st_mtime_ns,
    )


def add_integrity_columns(
    frame: pd.DataFrame,
    workers: int,
    include_dhash: bool,
) -> pd.DataFrame:
    print(f"Hashing and verifying {len(frame):,} images with {workers} workers...")
    jobs = [(path, include_dhash) for path in frame["image_path"]]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(inspect_image, jobs))

    output = frame.copy()
    output[
        ["file_size", "sha256", "dhash", "width", "height", "modified_ns"]
    ] = pd.DataFrame(results, index=output.index)
    return output


def load_integrity_cache(frame: pd.DataFrame, cache_path: Path) -> pd.DataFrame | None:
    if not cache_path.exists():
        return None
    cache = pd.read_csv(cache_path)
    integrity_columns = [
        "image_path",
        "file_size",
        "sha256",
        "dhash",
        "width",
        "height",
        "modified_ns",
    ]
    if not set(integrity_columns).issubset(cache.columns):
        return None
    if set(cache["image_path"]) != set(frame["image_path"]):
        return None

    cache = cache[integrity_columns].copy()
    cached_stats = cache.set_index("image_path")[["file_size", "modified_ns"]].to_dict("index")
    for image_path in frame["image_path"]:
        stat = Path(image_path).stat()
        expected = cached_stats[image_path]
        if int(expected["file_size"]) != stat.st_size or int(expected["modified_ns"]) != stat.st_mtime_ns:
            return None
    print(f"Reusing verified integrity cache: {cache_path}")
    return frame.merge(cache, on="image_path", how="left", validate="one_to_one")


def assert_lesion_labels_are_consistent(frame: pd.DataFrame) -> None:
    target_conflicts = frame.groupby("lesion_id")["target"].nunique()
    conflicts = target_conflicts[target_conflicts > 1]
    if not conflicts.empty:
        raise ValueError(
            f"Found {len(conflicts)} lesions with conflicting benign/high-risk labels."
        )


def nested_counts(frame: pd.DataFrame, group_columns: list[str]) -> list[dict[str, object]]:
    return [
        {**{column: key[index] for index, column in enumerate(group_columns)}, "count": int(count)}
        for key, count in frame.groupby(group_columns, dropna=False).size().items()
    ]


def main() -> None:
    args = parse_args()
    validate_inputs(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ham = normalize_columns(load_ham(args.ham_metadata, args.ham_image_dirs))
    bcn = normalize_columns(load_bcn(args.bcn_metadata, args.bcn_image_dir))
    combined = pd.concat([ham, bcn], ignore_index=True)
    assert_lesion_labels_are_consistent(combined)
    cache_path = args.output_dir / "integrity_cache.csv"
    cached = load_integrity_cache(combined, cache_path)
    if cached is None:
        combined = add_integrity_columns(
            combined,
            workers=args.hash_workers,
            include_dhash=not args.skip_perceptual_hash,
        )
        combined.to_csv(cache_path, index=False)
    else:
        combined = cached

    exact_mask = combined.duplicated("sha256", keep=False)
    exact_duplicates = combined.loc[exact_mask].sort_values(["sha256", "source", "image_id"])
    exact_duplicates.to_csv(args.output_dir / "exact_duplicates.csv", index=False)

    label_conflicts = exact_duplicates.groupby("sha256")["target"].nunique()
    conflicting_hashes = set(label_conflicts[label_conflicts > 1].index)
    conflicting_duplicates = exact_duplicates[
        exact_duplicates["sha256"].isin(conflicting_hashes)
    ].copy()
    conflicting_duplicates.to_csv(
        args.output_dir / "exact_duplicate_label_conflicts.csv", index=False
    )

    safe_rows = combined[~combined["sha256"].isin(conflicting_hashes)]
    deduplicated = safe_rows.drop_duplicates("sha256", keep="first").copy()
    perceptual_candidates = pd.DataFrame(columns=combined.columns)
    if not args.skip_perceptual_hash:
        perceptual_mask = deduplicated.duplicated("dhash", keep=False)
        perceptual_candidates = deduplicated.loc[perceptual_mask].sort_values(
            ["dhash", "source", "image_id"]
        )
    perceptual_candidates.to_csv(
        args.output_dir / "perceptual_duplicate_candidates.csv", index=False
    )

    manifest_path = args.output_dir / "combined_manifest.csv"
    deduplicated.to_csv(manifest_path, index=False)
    summary = {
        "manifest_path": str(manifest_path.resolve()),
        "input_images": int(len(combined)),
        "output_images": int(len(deduplicated)),
        "lesions": int(deduplicated["lesion_id"].nunique()),
        "same_label_exact_duplicates_removed": int(len(safe_rows) - len(deduplicated)),
        "conflicting_duplicate_groups": int(len(conflicting_hashes)),
        "conflicting_duplicate_images_removed": int(len(conflicting_duplicates)),
        "perceptual_candidate_images": int(len(perceptual_candidates)),
        "perceptual_candidate_groups": int(perceptual_candidates["dhash"].nunique()),
        "source_counts": {
            str(key): int(value) for key, value in deduplicated["source"].value_counts().items()
        },
        "target_counts": {
            str(key): int(value)
            for key, value in deduplicated["target_name"].value_counts().items()
        },
        "diagnosis_counts": {
            str(key): int(value)
            for key, value in deduplicated["diagnosis"].value_counts().sort_index().items()
        },
        "source_target_counts": nested_counts(deduplicated, ["source", "target_name"]),
    }
    (args.output_dir / "manifest_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
