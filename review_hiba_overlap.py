"""Second-stage image-similarity audit; never deletes or relabels an image."""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps
from scipy.fft import dctn

from evaluate_hiba import DATA, OUT, ROOT, digest, write_json


def native_path(value):
    if value.startswith("/mnt/"):
        return Path(value[5].upper() + ":/" + value[7:])
    return Path(value)


@lru_cache(maxsize=4096)
def signatures(path):
    with Image.open(path) as image:
        image.draft("RGB", (256, 256))
        gray = ImageOps.exif_transpose(image).convert("L").resize((64, 64), Image.Resampling.LANCZOS)
        pixels = np.asarray(gray, dtype=float)
    result = []
    for flip in (False, True):
        base = np.fliplr(pixels) if flip else pixels
        for rotations in range(4):
            coefficients = dctn(np.rot90(base, rotations), type=2, norm="ortho")[:16, :16].ravel()[1:]
            result.append(coefficients > np.median(coefficients))
    return np.asarray(result)


def main():
    candidates_path = OUT / "perceptual_candidates.csv"
    candidates = pd.read_csv(candidates_path)
    manifests = [pd.read_csv(p, usecols=["image_id", "image_path"], dtype=str)
                 for p in sorted((ROOT / "data").glob("combined*/combined_manifest.csv"))]
    lookup = pd.concat(manifests).drop_duplicates("image_id").set_index("image_id").image_path.to_dict()
    rows = []
    for row in candidates.itertuples():
        new_path = DATA / "images" / f"{row.isic_id}.jpg"
        old_path = native_path(lookup[row.prior_image_id])
        distance = int(np.count_nonzero(signatures(new_path) != signatures(old_path)[0], axis=1).min())
        rows.append({"isic_id": row.isic_id, "prior_image_id": row.prior_image_id,
                     "dhash_distance": row.dhash_distance, "phash255_min_d4_distance": distance,
                     "prior_path": str(old_path)})
        if len(rows) % 100 == 0:
            print(f"Secondary screen: {len(rows)}/{len(candidates)}", flush=True)
    results = pd.DataFrame(rows).sort_values("phash255_min_d4_distance")
    results.to_csv(OUT / "secondary_similarity_audit.csv", index=False)
    # Contact sheets are local audit evidence, not redistributed dataset assets.
    for start in range(0, min(24, len(results)), 6):
        sheet = Image.new("RGB", (900, 6 * 250), "white")
        draw = ImageDraw.Draw(sheet)
        for offset, row in enumerate(results.iloc[start:start + 6].itertuples()):
            y = offset * 250
            draw.text((8, y + 4), f"{row.isic_id} | {row.prior_image_id} | pHash255 distance {row.phash255_min_d4_distance}", fill="black")
            for x, path in [(0, DATA / "images" / f"{row.isic_id}.jpg"), (450, Path(row.prior_path))]:
                with Image.open(path) as image:
                    thumb = ImageOps.contain(ImageOps.exif_transpose(image).convert("RGB"), (440, 220))
                    sheet.paste(thumb, (x + 5, y + 25))
        sheet.save(OUT / f"similarity_contact_sheet_{start // 6}.jpg")
    write_json(OUT / "secondary_similarity_summary.json", {
        "candidate_file_sha256": digest(candidates_path), "pairs_screened": len(results),
        "method": "255-bit DCT perceptual hash; minimum over eight rotations/reflections; candidates from dHash<=3",
        "minimum_distance": int(results.phash255_min_d4_distance.min()),
        "distance_le_32": int(results.phash255_min_d4_distance.le(32).sum()),
        "distance_le_48": int(results.phash255_min_d4_distance.le(48).sum()),
        "report_sha256": digest(OUT / "secondary_similarity_audit.csv"),
        "limitation": "Similarity screening is not proof of independence; cropped/reencoded duplicates may be missed."})
    print(results.head(30).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
