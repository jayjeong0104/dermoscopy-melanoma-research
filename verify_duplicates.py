from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parent
    image_paths = list(root.glob("HAM10000_images_part_*/*.jpg"))

    primaries: dict[str, Path] = {}
    duplicates: dict[str, Path] = {}

    for image_path in image_paths:
        stem = image_path.stem
        if stem.endswith(" (2)"):
            duplicates[stem[:-4]] = image_path
        else:
            primaries[stem] = image_path

    missing_primary: list[str] = []
    missing_duplicate: list[str] = []
    size_mismatches: list[str] = []
    hash_mismatches: list[str] = []
    matched_pairs = 0

    for image_id, primary_path in primaries.items():
        duplicate_path = duplicates.get(image_id)
        if duplicate_path is None:
            missing_duplicate.append(image_id)
            continue

        if primary_path.stat().st_size != duplicate_path.stat().st_size:
            size_mismatches.append(image_id)
            continue

        if sha256(primary_path) != sha256(duplicate_path):
            hash_mismatches.append(image_id)
            continue

        matched_pairs += 1

    for image_id in duplicates:
        if image_id not in primaries:
            missing_primary.append(image_id)

    summary = {
        "primary_files": len(primaries),
        "duplicate_files": len(duplicates),
        "matched_pairs": matched_pairs,
        "missing_primary": len(missing_primary),
        "missing_duplicate": len(missing_duplicate),
        "size_mismatches": len(size_mismatches),
        "hash_mismatches": len(hash_mismatches),
        "missing_primary_samples": missing_primary[:5],
        "missing_duplicate_samples": missing_duplicate[:5],
        "size_mismatch_samples": size_mismatches[:5],
        "hash_mismatch_samples": hash_mismatches[:5],
    }

    print(json.dumps(summary, indent=2))

    if any(
        [
            missing_primary,
            missing_duplicate,
            size_mismatches,
            hash_mismatches,
        ]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
