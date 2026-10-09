"""Uniform lesion exposure within a source/diagnosis stratum, without dropping images."""
from __future__ import annotations

import numpy as np
import pandas as pd


def lesion_rows(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    required = ["source", "image_id", "lesion_id", "split_group_id", "diagnosis", "target"]
    if frame.empty or frame[required].isna().any().any():
        raise ValueError("Missing sampling identities or labels.")
    if frame.duplicated(["source", "image_id"]).any():
        raise ValueError("Duplicate sampling image identity.")
    if any(frame[key].nunique() != 1 for key in ("source", "diagnosis", "target")):
        raise ValueError("Lesion sampler requires one source/diagnosis/target stratum.")
    if frame.groupby("lesion_id").split_group_id.nunique().ne(1).any():
        raise ValueError("A sampling lesion crosses patient/split groups.")
    ordered = frame.reset_index(drop=True).sort_values(["lesion_id", "image_id"])
    counts = ordered.groupby("lesion_id", sort=True).size().to_numpy(dtype=np.int64)
    starts = np.concatenate(([0], np.cumsum(counts)[:-1])).astype(np.int64)
    return ordered.index.to_numpy(dtype=np.int64), starts, counts


def row_dataset(frame: pd.DataFrame, seed: int):
    import tensorflow as tf

    rows, starts, counts = lesion_rows(frame)
    rows, starts, counts = (tf.constant(value, dtype=tf.int64) for value in (rows, starts, counts))
    lesions = tf.data.Dataset.range(len(starts)).shuffle(
        max(16, len(starts)), seed=seed, reshuffle_each_iteration=True).repeat()

    def choose(counter, lesion):
        # Each cycle visits every lesion; each visit chooses one of its original photos.
        offset = tf.random.stateless_uniform((), seed=tf.stack([tf.cast(seed, tf.int64), counter]),
            minval=tf.constant(0, tf.int64), maxval=tf.gather(counts, lesion), dtype=tf.int64)
        return tf.gather(rows, tf.gather(starts, lesion) + offset)

    return lesions.enumerate().map(choose)


def stratum_mass(frame: pd.DataFrame, alpha: float, mass_unit: str = "image") -> float:
    """Count independent lesions or legacy photographs for stratum allocation."""
    if mass_unit not in {"image", "lesion"}:
        raise ValueError("Unsupported sampling mass unit.")
    if not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("Invalid sampling exponent.")
    if frame.empty:
        raise ValueError("Sampling strata cannot be empty.")
    if mass_unit == "lesion":
        if "lesion_id" not in frame or frame.lesion_id.isna().any():
            raise ValueError("Lesion sampling mass requires complete lesion identities.")
        count = frame.lesion_id.nunique()
    else:
        count = len(frame)
    return float(count ** alpha)


def exposure_audit(frame: pd.DataFrame, alpha: float = .5, positive_fraction: float = .45,
                   mass_unit: str = "image") -> dict:
    if not 0 <= alpha <= 1 or not 0 < positive_fraction < 1:
        raise ValueError("Invalid sampling policy.")
    groups = list(frame.groupby(["source", "diagnosis"], sort=True))
    totals = {label: sum(stratum_mass(g, alpha, mass_unit) for _, g in groups if int(g.target.iloc[0]) == label)
              for label in (0, 1)}
    if min(totals.values()) <= 0:
        raise ValueError("Sampling needs both binary classes.")
    report = []
    for (source, diagnosis), group in groups:
        _, _, counts = lesion_rows(group)
        label = int(group.target.iloc[0])
        mass = (positive_fraction if label else 1 - positive_fraction) * stratum_mass(group, alpha, mass_unit) / totals[label]
        report.append({"source": str(source), "diagnosis": str(diagnosis), "target": label,
            "images": len(group), "lesions": len(counts), "source_diagnosis_probability": mass,
            "maximum_photos_per_lesion": int(counts.max()),
            "image_sampling_effective_lesions": float(counts.sum() ** 2 / np.sum(counts ** 2)),
            "lesion_sampling_effective_lesions": len(counts),
            "old_min_lesion_probability": float(mass * counts.min() / counts.sum()),
            "old_max_lesion_probability": float(mass * counts.max() / counts.sum()),
            "new_each_lesion_probability": float(mass / len(counts))})
    return {"training_images": len(frame), "lesions": int(sum(g["lesions"] for g in report)),
        "positive_fraction": positive_fraction, "sampling_alpha": alpha,
        "sampling_mass_unit": mass_unit, "strata": report,
        "source_class_mass_preserved": mass_unit == "image",
        "binary_class_mass_preserved": True, "deleted_images": 0, "changed_labels": 0,
        "limitation": "Exposure is equal within source/diagnosis only, not globally or per patient. All photos remain eligible; a budget need not visit every photograph."}
