"""Read-only, fixed-sample training-image preprocessing QA; no label adjudication."""
import hashlib
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
from PIL import Image

from artifact_io import atomic_output, save_csv
from foundation_common import ROOT, split_frame, save_json
from run_convergence_study import local_path


def center_view(image):
    width, height = image.size
    resized = (int(256 * width / height), 256) if width > height else (256, int(256 * height / width))
    scaled = image.resize(resized, Image.Resampling.BILINEAR)
    left, top = round((resized[0] - 224) / 2), round((resized[1] - 224) / 2)
    box = (left * width / resized[0], top * height / resized[1],
           224 * width / resized[0], 224 * height / resized[1])
    return scaled.crop((left, top, left + 224, top + 224)), box, 224 ** 2 / (resized[0] * resized[1])


def main():
    output = ROOT / "artifacts/convergence_recovery_20261001/crop_audit"
    output.mkdir(parents=True, exist_ok=True)
    frame = split_frame("train").drop_duplicates(["source", "lesion_id"])
    selected = []
    for source in sorted(frame.source.unique()):
        for diagnosis in ("MEL", "NV", "BKL"):
            pool = frame.loc[frame.source.eq(source) & frame.diagnosis.eq(diagnosis)]
            selected.append(pool.sample(n=min(2, len(pool)), random_state=20261001))
    cases = pd.concat(selected).reset_index(drop=True)
    records = []
    for start in range(0, len(cases), 6):
        page = cases.iloc[start:start + 6]
        figure, axes = plt.subplots(3, 4, figsize=(14, 11))
        for axis in axes.flat:
            axis.axis("off")
        for index, row in enumerate(page.itertuples()):
            raw = local_path(row.image_path).read_bytes()
            if hashlib.sha256(raw).hexdigest() != row.sha256:
                raise ValueError("Training image content changed.")
            with Image.open(io.BytesIO(raw)) as opened:
                original = opened.convert("RGB")
            view, box, retained = center_view(original)
            pair = axes[index // 2, (index % 2) * 2:(index % 2) * 2 + 2]
            pair[0].imshow(original)
            pair[0].add_patch(Rectangle(box[:2], box[2], box[3], fill=False, edgecolor="#00e5ff", linewidth=1.5))
            pair[0].set_title(f"{row.source} / {row.diagnosis}\n{row.image_id}", fontsize=8)
            pair[1].imshow(view)
            pair[1].set_title(f"Encoder center crop\n{100 * retained:.1f}% of image area retained", fontsize=8)
            records.append({"source": row.source, "diagnosis": row.diagnosis, "image_id": row.image_id,
                "lesion_id": row.lesion_id, "image_path": row.image_path, "sha256": row.sha256,
                "width": original.width, "height": original.height, "retained_image_area_fraction": retained})
        figure.suptitle("Training-only crop QA: cyan box is the retained region; labels are dataset labels, not a medical review.", fontsize=10)
        figure.tight_layout(rect=(0, 0, 1, .96))
        with atomic_output(output / f"training_crops_{start // 6 + 1}.png") as path:
            figure.savefig(path, dpi=130)
        plt.close(figure)
    save_csv(pd.DataFrame(records), output / "sample.csv")
    save_json(output / "audit.json", {"training_only": True, "cases": len(records), "random_seed": 20261001,
        "selection": "Two distinct lesions per source and diagnosis MEL/NV/BKL, fixed seed before visualization",
        "meaning": "Area fraction measures image cropping, not lesion tissue removal; no segmentation was available.",
        "labels_changed": False, "original_images_modified": False, "expert_diagnostic_review": False})
    print(f"Rendered {len(records)} training-only preprocessing comparisons in {output}")


if __name__ == "__main__":
    main()
