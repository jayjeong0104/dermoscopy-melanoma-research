from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import train
import pandas as pd
import tensorflow as tf


PH2_PATTERN = re.compile(
    r"^\|\|\s*(IMD\d{3})\s*\|\|\s*(.*?)\s*\|\|\s*(\d)\s*\|\|\s*(\d)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\|\s*(.*?)\s*\|\|$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved skin-lesion model on the external PH2 dataset."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Path to an artifacts/run_* directory created by train.py.",
    )
    parser.add_argument(
        "--ph2-root",
        type=Path,
        default=Path("PH2Dataset"),
        help="Path to the PH2 dataset root folder.",
    )
    parser.add_argument(
        "--model-name",
        default="best_model.keras",
        choices=["best_model.keras", "final_model.keras"],
        help="Saved model file inside the run directory to evaluate.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional explicit decision threshold. If omitted, uses the run's saved selected threshold when available.",
    )
    return parser.parse_args()


def load_threshold(run_dir: Path, model_name: str, override: float | None) -> float:
    if override is not None:
        return float(override)

    model_stem = Path(model_name).stem
    candidate_paths = [
        run_dir / f"{model_stem}_evaluation.json",
        run_dir / "selected_threshold.json",
        run_dir / "test_metrics.json",
    ]
    for path in candidate_paths:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "selected_threshold" in payload:
            return float(payload["selected_threshold"])

    return 0.5


def load_ph2_dataframe(ph2_root: Path) -> pd.DataFrame:
    label_path = ph2_root / "PH2_dataset.txt"
    image_root = ph2_root / "PH2 Dataset images"
    if not label_path.exists():
        raise FileNotFoundError(f"PH2 label file not found: {label_path}")
    if not image_root.exists():
        raise FileNotFoundError(f"PH2 image root not found: {image_root}")

    rows: list[dict[str, object]] = []
    missing_images: list[str] = []

    for raw_line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = PH2_PATTERN.match(raw_line.strip())
        if not match:
            continue

        (
            image_id,
            histological_diagnosis,
            clinical_diagnosis,
            asymmetry,
            pigment_network,
            dots_globules,
            streaks,
            regression_areas,
            blue_whitish_veil,
            colors,
        ) = match.groups()

        image_path = image_root / image_id / f"{image_id}_Dermoscopic_Image" / f"{image_id}.bmp"
        if not image_path.exists():
            missing_images.append(image_id)
            continue

        clinical_code = int(clinical_diagnosis)
        rows.append(
            {
                "image_id": image_id,
                "image_path": str(image_path.resolve()),
                "histological_diagnosis": histological_diagnosis or "",
                "clinical_diagnosis": clinical_code,
                "clinical_name": {
                    0: "common_nevus",
                    1: "atypical_nevus",
                    2: "melanoma",
                }[clinical_code],
                "target": int(clinical_code == 2),
                "target_name": "malignant" if clinical_code == 2 else "benign",
                "asymmetry": int(asymmetry),
                "pigment_network": pigment_network,
                "dots_globules": dots_globules,
                "streaks": streaks,
                "regression_areas": regression_areas,
                "blue_whitish_veil": blue_whitish_veil,
                "colors": colors.split(),
            }
        )

    if missing_images:
        sample = ", ".join(missing_images[:5])
        raise FileNotFoundError(
            f"Missing {len(missing_images)} PH2 dermoscopic images. Sample: {sample}"
        )

    frame = pd.DataFrame(rows).sort_values("image_id").reset_index(drop=True)
    if len(frame) != 200:
        print(f"Warning: expected 200 PH2 cases, found {len(frame)}")
    return frame


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    ph2_root = args.ph2_root.resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    model_path = run_dir / args.model_name
    if not model_path.exists():
        raise FileNotFoundError(f"Saved model not found: {model_path}")

    run_config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    threshold = load_threshold(run_dir, args.model_name, args.threshold)
    ph2_frame = load_ph2_dataframe(ph2_root)

    dataset = train.build_dataset(
        ph2_frame,
        image_size=int(run_config["image_size"]),
        batch_size=int(run_config["batch_size"]),
        training=False,
    )
    model = tf.keras.models.load_model(model_path)

    probabilities = train.predict_probabilities(model, dataset, verbose=0)
    truth = ph2_frame["target"].to_numpy()
    default_summary = train.compute_threshold_metrics(truth, probabilities, threshold=0.5)
    tuned_summary = train.compute_threshold_metrics(truth, probabilities, threshold=threshold)
    auc_score = float(train.roc_auc_score(truth, probabilities))

    payload = {
        "dataset": "PH2",
        "model_name": args.model_name,
        "sample_count": int(len(ph2_frame)),
        "class_counts": {
            "benign": int((truth == 0).sum()),
            "malignant": int((truth == 1).sum()),
        },
        "roc_auc_score": auc_score,
        "default_threshold_metrics": {
            **default_summary["metrics"],
            "roc_auc_score": auc_score,
        },
        "selected_threshold": float(threshold),
        "selected_threshold_metrics": {
            **tuned_summary["metrics"],
            "roc_auc_score": auc_score,
        },
    }

    model_stem = Path(args.model_name).stem
    train.save_json(run_dir / f"{model_stem}_ph2_evaluation.json", payload)
    train.save_json(
        run_dir / f"{model_stem}_ph2_classification_report.json",
        {
            "default_threshold": {
                "threshold": 0.5,
                "report": default_summary["report"],
            },
            "selected_threshold": {
                "threshold": float(threshold),
                "report": tuned_summary["report"],
            },
        },
    )
    train.save_json(
        run_dir / f"{model_stem}_ph2_confusion_matrix.json",
        {
            "default_threshold": {
                "threshold": 0.5,
                "matrix": default_summary["confusion_matrix"],
            },
            "selected_threshold": {
                "threshold": float(threshold),
                "matrix": tuned_summary["confusion_matrix"],
            },
        },
    )

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
