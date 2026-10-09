"""One-shot external HIBA evaluation of the frozen, published research model."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import time
import urllib.request

os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data/hiba_external"
OUT = ROOT / "artifacts/hiba_external_20261003"
RELEASE = ROOT / "publication/dermoscopy-melanoma-research"
MODEL = ROOT / "artifacts/melanoma_v1/run_20260822_111527/best_model.keras"
METADATA_URL = "https://api.isic-archive.com/collections/251/metadata/"
MELANOMA = "Malignant melanocytic proliferations (Melanoma)"
ALLOWED_DIAGNOSES = {
    MELANOMA, "Benign epidermal proliferations", "Benign melanocytic proliferations",
    "Benign soft tissue proliferations - Fibro-histiocytic",
    "Benign soft tissue proliferations - Vascular",
    "Indeterminate epidermal proliferations",
    "Malignant adnexal epithelial proliferations - Follicular",
    "Malignant epidermal proliferations",
}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def status(stage, **extra):
    value = {"stage": stage, "utc": datetime.now(timezone.utc).isoformat(),
             "pid": os.getpid(), **extra}
    write_json(OUT / "status.json", value)
    print(json.dumps(value), flush=True)


def fetch(url):
    for attempt in range(4):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "HIBA-research-evaluation/1.0"})
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def select_dermoscopy(frame):
    if frame.isic_id.duplicated().any():
        raise ValueError("Repeated HIBA image IDs")
    selected = frame.loc[frame.image_type.eq("dermoscopic")].copy()
    required = ["isic_id", "lesion_id", "patient_id", "diagnosis_2"]
    if selected[required].isna().any().any():
        raise ValueError("Missing identity or diagnosis; define exclusion before evaluation")
    if not selected.isic_id.str.fullmatch(r"ISIC_\d{7}").all():
        raise ValueError("Invalid ISIC ID")
    if not set(selected.diagnosis_2) <= ALLOWED_DIAGNOSES:
        raise ValueError("Unexpected diagnosis; review target definition before inference")
    if not selected.copyright_license.eq("CC-BY").all():
        raise ValueError("Unexpected license; review before download")
    selected["target"] = selected.diagnosis_2.eq(MELANOMA).astype(int)
    for column in ("patient_id", "target"):
        if selected.groupby("lesion_id")[column].nunique().gt(1).any():
            raise ValueError("Conflicting lesion identity or binary diagnosis")
    return selected.sort_values("isic_id").reset_index(drop=True)


def known_records():
    paths = sorted((ROOT / "data").glob("combined*/combined_manifest.csv"))
    paths += sorted((ROOT / "data").glob("combined*/exact_duplicates.csv"))
    paths += sorted((ROOT / "data").glob("combined*/integrity_cache.csv"))
    paths += sorted((ROOT / "data").glob("combined*/siim_integrity_cache.csv"))
    paths += [MODEL.parent / "derm12345_external_predictions.csv"]
    records, sources = [], []
    columns = {"image_id", "sha256", "dhash", "patient_id", "lesion_id", "source_lesion_id"}
    for path in paths:
        frame = pd.read_csv(path, dtype=str, usecols=lambda x: x in columns)
        records.append(frame)
        sources.append({"path": path.relative_to(ROOT).as_posix(), "sha256": digest(path), "rows": len(frame)})
    known = pd.concat(records, ignore_index=True).fillna("")
    # PH2 was used for earlier external-development checks; screen it as well.
    ph2_paths = sorted((ROOT / "PH2Dataset").rglob("*.bmp"))
    if ph2_paths:
        ph2 = pd.DataFrame({"image_id": [p.stem for p in ph2_paths],
                            "sha256": [digest(p) for p in ph2_paths]})
        known = pd.concat([known, ph2], ignore_index=True).fillna("")
        sources.append({"path": "PH2Dataset/**/*.bmp", "rows": len(ph2),
                        "sha256_list_digest": hashlib.sha256("\n".join(ph2.sha256).encode()).hexdigest()})
    return known, sources


def id_overlap(selected, known):
    def ids(column):
        return set(known[column].str.split("::").str[-1]) - {""}
    return (selected.isic_id.isin(ids("image_id"))
            | selected.patient_id.isin(ids("patient_id"))
            | selected.lesion_id.isin(ids("lesion_id") | ids("source_lesion_id")))


def download_image(isic_id):
    path = DATA / "images" / f"{isic_id}.jpg"
    if not path.exists():
        data = fetch(f"https://isic-archive.s3.amazonaws.com/images/{isic_id}.jpg")
        with Image.open(io.BytesIO(data)) as image:
            if image.format != "JPEG":
                raise ValueError(f"Non-JPEG image: {isic_id}")
            image.verify()
        temporary = path.with_suffix(".part")
        temporary.write_bytes(data)
        temporary.replace(path)
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        image.load()
        pixels = np.asarray(image.convert("L").resize((9, 8), Image.Resampling.LANCZOS))
        bits = (pixels[:, :-1] > pixels[:, 1:]).reshape(-1)
        dhash = f"{int(''.join('1' if bit else '0' for bit in bits), 2):016x}"
        width, height = image.size
    return {"isic_id": isic_id, "sha256": digest(path), "dhash": dhash,
            "width": width, "height": height, "bytes": path.stat().st_size}


def prepare():
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "images").mkdir(exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "results.json").exists():
        raise RuntimeError("Completed test already exists; do not silently rerun or tune")
    manifest_path = RELEASE / "models/model_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if digest(MODEL) != manifest["checkpoint_sha256"]:
        raise ValueError("Retained checkpoint mismatch")
    protocol = {"model": manifest, "endpoint": "melanoma versus all other diagnoses, including other cancers",
                "dataset": "ISIC collection 251 HIBA; dermoscopic only",
                "source": "https://doi.org/10.34970/587329", "primary_unit": "lesion",
                "tuning": "none", "metadata_input": "all unknown",
                "overlap_policy": "exclude affected patients on ID or exact-file overlap; never delete originals",
                "internal_exact_duplicates": "one copy per hash; conflicting identity/labels fail closed",
                "uncertainty": "2000 patient-cluster bootstrap replicates, seed 20261003",
                "limitations": ["No globally complete patient identity linkage across datasets",
                                "Exact hashes and dHash screening do not prove absence of transformed duplicates",
                                "Not a clinical validation or a test on ordinary smartphone photographs"],
                "code_sha256": {"evaluator": digest(__file__), "training_helpers": digest(ROOT / "train_multitask.py"),
                                "released_inference": digest(RELEASE / "predict_research.py")}}
    protocol_path = OUT / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != protocol:
            raise RuntimeError("Frozen protocol changed")
    else:
        write_json(protocol_path, protocol)
    status("metadata_and_overlap")
    metadata_path = DATA / "metadata.csv"
    if not metadata_path.exists():
        metadata_path.write_bytes(fetch(METADATA_URL))
    full = pd.read_csv(metadata_path, dtype=str)
    selected = select_dermoscopy(full)
    known, sources = known_records()
    initial_overlap = id_overlap(selected, known)
    download_rows = selected.loc[~initial_overlap]
    inspected = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(download_image, image_id) for image_id in download_rows.isic_id]
        for future in as_completed(futures):
            inspected.append(future.result())
            if len(inspected) % 100 == 0:
                status("downloading", completed=len(inspected), total=len(download_rows))
    selected = selected.merge(pd.DataFrame(inspected), on="isic_id", how="left", validate="one_to_one")
    exact_overlap = selected.sha256.isin(set(known.sha256) - {""})
    overlap = id_overlap(selected, known) | exact_overlap
    excluded_patients = set(selected.loc[overlap, "patient_id"])
    excluded = selected.loc[selected.patient_id.isin(excluded_patients)]
    excluded.to_csv(OUT / "excluded_overlap.csv", index=False)
    eligible = selected.loc[~selected.patient_id.isin(excluded_patients)].copy()
    if eligible.sha256.isna().any():
        raise ValueError("Incomplete downloaded images")
    repeated = eligible.loc[eligible.sha256.duplicated(keep=False)]
    for _, group in repeated.groupby("sha256"):
        if any(group[c].nunique() != 1 for c in ("patient_id", "lesion_id", "target")):
            raise ValueError("Exact duplicate has conflicting identity or diagnosis")
    repeated.to_csv(OUT / "internal_exact_duplicates.csv", index=False)
    eligible = eligible.drop_duplicates("sha256").reset_index(drop=True)
    # Candidate screening only: visual similarity is never treated as proof of duplication.
    prior = known.loc[known.dhash.str.fullmatch(r"[0-9a-fA-F]{16}"), ["image_id", "dhash"]].drop_duplicates("dhash")
    prior_values = np.array([int(v, 16) for v in prior.dhash], dtype=np.uint64)
    popcount = np.array([i.bit_count() for i in range(256)], dtype=np.uint8)
    candidates = []
    for row in eligible.itertuples():
        xor = np.bitwise_xor(prior_values, np.uint64(int(row.dhash, 16)))
        distances = popcount[xor.view(np.uint8).reshape(-1, 8)].sum(axis=1)
        for index in np.flatnonzero(distances <= 3):
            candidates.append({"isic_id": row.isic_id, "prior_image_id": prior.iloc[index].image_id,
                               "dhash_distance": int(distances[index])})
    pd.DataFrame(candidates, columns=["isic_id", "prior_image_id", "dhash_distance"]).to_csv(OUT / "perceptual_candidates.csv", index=False)
    audit = {"metadata_sha256": digest(metadata_path), "known_sources": sources,
             "all_collection_images": len(full), "dermoscopic_images": int(full.image_type.eq("dermoscopic").sum()),
             "image_or_identity_overlap_rows": int(initial_overlap.sum()), "exact_overlap_rows": int(exact_overlap.sum()),
             "excluded_patient_count": len(excluded_patients), "excluded_image_count": len(excluded),
             "internal_duplicate_extra_copies": len(selected) - len(excluded) - len(eligible),
             "perceptual_candidates_distance_le_3": len(candidates), "evaluation_images": len(eligible),
             "evaluation_lesions": eligible.lesion_id.nunique(), "evaluation_patients": eligible.patient_id.nunique(),
             "melanoma_lesions": eligible.drop_duplicates("lesion_id").target.sum().item(),
             "downloaded_bytes": int(eligible.bytes.sum()),
             "limitation": "Available IDs and SHA-256 only; missing/nonlinked patient IDs and transformed copies cannot be ruled out globally."}
    eligible.to_csv(OUT / "evaluation_manifest.csv", index=False)
    audit["evaluation_manifest_sha256"] = digest(OUT / "evaluation_manifest.csv")
    write_json(OUT / "overlap_audit.json", audit)
    if candidates:
        raise RuntimeError("Perceptual candidates need manual review before test; no files deleted")
    status("ready", **{k: audit[k] for k in ["evaluation_images", "evaluation_lesions", "evaluation_patients", "melanoma_lesions"]})


def measure(truth, scores, threshold):
    from sklearn.metrics import roc_auc_score, average_precision_score
    truth = np.asarray(truth, dtype=int)
    predicted = np.asarray(scores) >= threshold
    tp, tn = int(np.sum(predicted & (truth == 1))), int(np.sum(~predicted & (truth == 0)))
    fp, fn = int(np.sum(predicted & (truth == 0))), int(np.sum(~predicted & (truth == 1)))
    return {"accuracy": (tp + tn) / len(truth), "precision": tp / (tp + fp) if tp + fp else 0.,
            "sensitivity": tp / (tp + fn), "specificity": tn / (tn + fp),
            "roc_auc": float(roc_auc_score(truth, scores)), "pr_auc_average_precision": float(average_precision_score(truth, scores)),
            "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def evaluate(batch_size):
    import sys
    import tensorflow as tf
    import train_multitask as training
    sys.path.insert(0, str(RELEASE))
    from predict_research import aggregate_scores

    if (OUT / "results.json").exists():
        raise RuntimeError("Results already exist; do not overwrite this independent test")
    protocol = json.loads((OUT / "protocol.json").read_text())
    audit = json.loads((OUT / "overlap_audit.json").read_text())
    review = None
    if audit["perceptual_candidates_distance_le_3"]:
        review = json.loads((OUT / "overlap_review.json").read_text())
        if (review["unresolved_suspected_pairs"] != 0
                or review["candidate_file_sha256"] != digest(OUT / "perceptual_candidates.csv")
                or review["secondary_report_sha256"] != digest(OUT / "secondary_similarity_audit.csv")):
            raise RuntimeError("Incomplete or stale possible-overlap review")
    if digest(OUT / "evaluation_manifest.csv") != audit["evaluation_manifest_sha256"]:
        raise ValueError("Evaluation manifest changed after audit")
    if digest(MODEL) != protocol["model"]["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed")
    expected_code = dict(protocol["code_sha256"])
    amendment_path = OUT / "protocol_amendment_01.json"
    if amendment_path.exists():
        amendment = json.loads(amendment_path.read_text())
        if amendment["original_protocol_sha256"] != digest(OUT / "protocol.json"):
            raise ValueError("Protocol amendment references a different protocol")
        expected_code["evaluator"] = amendment["evaluator_sha256"]
    for key, path in [("evaluator", Path(__file__)), ("training_helpers", ROOT / "train_multitask.py"),
                      ("released_inference", RELEASE / "predict_research.py")]:
        if digest(path) != expected_code[key]:
            raise ValueError(f"Code changed after protocol was frozen: {key}")
    config = protocol["model"]["inference"]
    frame = pd.read_csv(OUT / "evaluation_manifest.csv")
    paths = [DATA / "images" / f"{i}.jpg" for i in frame.isic_id]
    if any(digest(p) != h for p, h in zip(paths, frame.sha256)):
        raise ValueError("An image changed after audit")
    devices = tf.config.list_physical_devices("GPU")
    for device in devices:
        tf.config.experimental.set_memory_growth(device, True)
    encoder = training.MetadataEncoder(age_median=config["metadata_age_median"],
                                       feature_names=tuple(config["metadata_feature_names"]))
    unknown = pd.DataFrame({"age_approx": [np.nan] * len(frame), "sex": "unknown", "anatom_site_general": "unknown"})
    # Ground-truth labels never enter the inference dataset.
    dataset = tf.data.Dataset.from_tensor_slices(([str(p) for p in paths], encoder.transform(unknown)))

    def decode(path, metadata):
        image = training.resize_and_normalize_image(
            tf.io.decode_jpeg(tf.io.read_file(path), channels=3), config["image_size"])
        image.set_shape([config["image_size"], config["image_size"], 3])
        return {"image": image, "metadata": metadata}, tf.constant(0., dtype=tf.float32)

    dataset = dataset.map(decode, num_parallel_calls=1).batch(batch_size)
    options = tf.data.Options()
    options.threading.private_threadpool_size = 1
    options.threading.max_intra_op_parallelism = 1
    dataset = dataset.with_options(options).prefetch(1)
    model = tf.keras.models.load_model(MODEL, compile=False, safe_mode=True)
    start = time.monotonic()
    direct_views, diagnosis_views = [], []
    for view in range(config["tta_transforms"]):
        status("inference", view=view + 1, total_views=config["tta_transforms"], gpu_devices=[str(d) for d in devices])
        direct, diagnosis = training.unpack_predictions(model, model.predict(
            training.transform_tta_dataset(dataset, view, config["tta_scheme"]), verbose=0))
        direct_views.append(direct.astype("float64"))
        diagnosis_views.append(diagnosis.astype("float64"))
        np.savez_compressed(OUT / f"predictions_view_{view}.npz", direct=direct, diagnosis=diagnosis)
    direct = np.mean(direct_views, axis=0).astype("float32")
    diagnoses = np.mean(diagnosis_views, axis=0).astype("float32")
    image_predictions = frame[["isic_id", "lesion_id", "patient_id", "target"]].copy()
    image_predictions["direct_raw"] = direct
    for i, name in enumerate(config["diagnosis_order"]):
        image_predictions[f"diagnosis_{name}"] = diagnoses[:, i]
    image_predictions.to_csv(OUT / "image_predictions.csv", index=False)
    lesions = []
    for lesion_id, group in frame.groupby("lesion_id", sort=False):
        indices = group.index.to_numpy()
        score = aggregate_scores(direct[indices], diagnoses[indices], config)
        lesions.append({"lesion_id": lesion_id, "patient_id": group.patient_id.iloc[0],
                        "target": int(group.target.iloc[0]), "image_count": len(group),
                        "research_score": score, "prediction": int(score >= config["threshold"])})
    lesions = pd.DataFrame(lesions)
    lesions.to_csv(OUT / "lesion_predictions.csv", index=False)
    metrics = measure(lesions.target, lesions.research_score, config["threshold"])
    status("confidence_intervals", lesions=len(lesions))
    rng = np.random.default_rng(20261003)
    groups = list(lesions.groupby("patient_id", sort=False).indices.values())
    replicates = []
    for _ in range(2000):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        subset = lesions.iloc[idx]
        if subset.target.nunique() == 2:
            replicates.append(measure(subset.target, subset.research_score, config["threshold"]))
    intervals = {key: [float(x) for x in np.percentile([r[key] for r in replicates], [2.5, 97.5])]
                 for key in ["accuracy", "precision", "sensitivity", "specificity", "roc_auc", "pr_auc_average_precision"]}
    results = {"completed_utc": datetime.now(timezone.utc).isoformat(), "dataset": "HIBA ISIC collection 251 dermoscopy",
               "model_sha256": digest(MODEL), "protocol_sha256": digest(OUT / "protocol.json"),
               "unit": "lesion", "images": len(frame), "lesions": len(lesions), "patients": len(groups),
               "melanoma_lesions": int(lesions.target.sum()), "melanoma_prevalence": float(lesions.target.mean()),
               "fixed_threshold": config["threshold"], "metrics": metrics,
               "patient_cluster_bootstrap_95_ci": intervals, "bootstrap_valid_replicates": len(replicates),
               "all_negative_accuracy": float(1 - lesions.target.mean()), "inference_and_scoring_seconds": time.monotonic() - start,
               "tensorflow_version": tf.__version__, "keras_version": tf.keras.__version__,
               "gpu_devices": [str(d) for d in devices], "clinical_use_validated": False,
               "no_training_or_threshold_tuning": True, "overlap_audit": audit,
               "overlap_review": review,
               "protocol_amendment_sha256": digest(amendment_path) if amendment_path.exists() else None,
               "limitations": protocol["limitations"]}
    write_json(OUT / "results.json", results)
    status("complete", metrics=metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "evaluate"])
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    try:
        prepare() if args.stage == "prepare" else evaluate(args.batch_size)
    except Exception as error:
        if OUT.exists():
            status("failed", error=repr(error))
        raise
