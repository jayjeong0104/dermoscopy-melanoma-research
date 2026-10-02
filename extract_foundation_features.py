"""Resumable, bounded-memory GPU extraction; no training or test inference."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from foundation_common import (ASSETS, BASELINE, ENCODERS, ROOT, fingerprint, read_chunk,
                               runtime_path, save_chunk, save_json, split_frame)
from compare_model_runs import file_hash

CHUNK_SIZE = 256


def checked_image_bytes(row) -> bytes:
    data = runtime_path(row.image_path).read_bytes()
    if hashlib.sha256(data).hexdigest() != row.sha256:
        raise ValueError(f"Image bytes changed: {row.source}/{row.image_id}")
    return data


def panderm_encoder():
    import torch
    import torchvision
    import timm
    import PIL
    from torchvision import transforms
    from PIL import Image
    if not torch.cuda.is_available():
        raise RuntimeError("PanDerm extraction requires the GPU; no silent CPU fallback.")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    source = ASSETS / "modeling_finetune.py"
    spec = importlib.util.spec_from_file_location("pinned_panderm", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    model = module.panderm_base_patch16_224()
    # Never use unrestricted pickle loading for downloaded model artifacts.
    state = torch.load(ASSETS / "panderm_base.pth", map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError("Expected the official flat PanDerm Base tensor state dictionary.")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if set(missing) - {"head.weight", "head.bias"} or unexpected:
        raise ValueError(f"Incomplete encoder weights: missing={missing}, unexpected={unexpected}")
    del state
    model.head = torch.nn.Identity()
    model.requires_grad_(False).eval().to("cuda")
    transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(), transforms.Normalize((.485, .456, .406), (.228, .224, .225))])

    def predict(rows):
        tensors = []
        for row in rows:
            with Image.open(io.BytesIO(checked_image_bytes(row))) as image:
                tensors.append(transform(image.convert("RGB")))
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            result = model(torch.stack(tensors).to("cuda"))
        return result.float().cpu().numpy()

    runtime = {"framework": "pytorch", "version": torch.__version__, "cuda": torch.version.cuda,
        "torchvision": torchvision.__version__, "timm": timm.__version__, "pillow": PIL.__version__,
        "gpu": torch.cuda.get_device_name(0), "precision": "autocast_float16",
        "missing_classifier_keys": missing, "encoder_parameters": sum(p.numel() for p in model.parameters()),
        "input": "RGB PIL Resize(256,bilinear,antialias), CenterCrop(224), official mean/std incl .228"}
    return predict, 768, runtime


def incumbent_encoder():
    import train_multitask as training
    tf = training.tf
    runtime = training.configure_runtime("auto")
    if not runtime["gpu_devices"]:
        raise RuntimeError("Incumbent extraction requires the GPU.")
    original = tf.keras.models.load_model(BASELINE / "best_model.keras", compile=False)
    backbone = original.get_layer("backbone")
    # Use the trained backbone, without its old normalization/projection/classifier.
    inputs = tf.keras.Input((300, 300, 3))
    pooled = tf.keras.layers.GlobalAveragePooling2D(dtype="float32")(backbone(inputs, training=False))
    model = tf.keras.Model(inputs, pooled)

    def predict(rows):
        images = [tf.image.resize(tf.io.decode_jpeg(checked_image_bytes(row), channels=3),
                                 (300, 300), antialias=True) for row in rows]
        return model(tf.stack(images), training=False).numpy().astype("float32")

    runtime.update({"framework": "tensorflow", "input": "RGB decode_jpeg, bilinear 300x300 antialias, [0,255]"})
    return predict, int(model.output_shape[-1]), runtime


def extract(experiment: Path, encoder: str, batch_size: int = 4, smoke: bool = False) -> None:
    from run_foundation_experiment import verify_plan
    plan = verify_plan(experiment)
    predict, dimensions, runtime = panderm_encoder() if encoder == "panderm_base" else incumbent_encoder()
    destination = experiment / ("smoke_features" if smoke else "features") / encoder
    destination.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"pid": os.getpid(), "encoder": encoder, "runtime": runtime}), flush=True)
    started = time.monotonic()
    for split in ("train", "val"):
        frame = split_frame(split)
        if smoke:
            frame = frame.iloc[:batch_size]
        cache = {"version": 1, "encoder": encoder, "runtime": runtime, "dimensions": dimensions,
                 "batch_size": batch_size, "chunk_size": CHUNK_SIZE, "rows": len(frame),
                 "input_sha256": plan["input_sha256"], "code_sha256": plan["code_sha256"],
                 "plan_sha256": file_hash(experiment / "experiment_plan.json"), "smoke_only": smoke}
        manifest_path = destination / f"{split}_manifest.json"
        if manifest_path.exists() and json.loads(manifest_path.read_text()) != cache:
            raise ValueError("Feature cache configuration changed; refusing silent reuse.")
        save_json(manifest_path, cache)
        for start in range(0, len(frame), CHUNK_SIZE):
            chunk = frame.iloc[start:start + CHUNK_SIZE]
            key = fingerprint({"manifest": cache, "start": start,
                               "images": chunk[["source", "image_id", "sha256"]].values.tolist()})
            path = destination / f"{split}_{start:06d}.npz"
            if path.exists():
                read_chunk(path, key, len(chunk), dimensions)
            else:
                values = []
                for index in range(0, len(chunk), batch_size):
                    values.append(predict(list(chunk.iloc[index:index + batch_size].itertuples())))
                features = np.concatenate(values).astype("float32")
                if features.shape != (len(chunk), dimensions):
                    raise ValueError("Unexpected encoder output dimensions.")
                save_chunk(path, features, key)
            progress = {"pid": os.getpid(), "encoder": encoder, "split": split,
                "completed_images": start + len(chunk), "total_images": len(frame),
                "elapsed_seconds": round(time.monotonic() - started, 1), "smoke_only": smoke}
            save_json(destination / "progress.json", progress)
            print(json.dumps(progress), flush=True)
    save_json(destination / "complete.json", {"smoke_only": smoke, "runtime": runtime,
        "outputs_sha256": {path.name: file_hash(path) for path in sorted(destination.glob("*.npz"))}})


def load_features(experiment: Path, encoder: str, split: str):
    if split not in {"train", "val"}:
        raise ValueError("Test feature extraction is deliberately disabled.")
    destination = experiment / "features" / encoder
    complete = json.loads((destination / "complete.json").read_text())
    if complete["smoke_only"]:
        raise ValueError("Smoke features cannot be used for a study.")
    cache = json.loads((destination / f"{split}_manifest.json").read_text())
    if cache["plan_sha256"] != file_hash(experiment / "experiment_plan.json"):
        raise ValueError("Features belong to a different experiment plan.")
    frame = split_frame(split)
    arrays = []
    for start in range(0, len(frame), CHUNK_SIZE):
        rows = frame.iloc[start:start + CHUNK_SIZE]
        key = fingerprint({"manifest": cache, "start": start,
                           "images": rows[["source", "image_id", "sha256"]].values.tolist()})
        path = destination / f"{split}_{start:06d}.npz"
        if file_hash(path) != complete["outputs_sha256"][path.name]:
            raise ValueError("Feature file changed after extraction.")
        arrays.append(read_chunk(path, key, len(rows), cache["dimensions"]))
    return frame, np.concatenate(arrays)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--encoder", choices=ENCODERS, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    extract(args.experiment_dir.resolve(), args.encoder, smoke=args.smoke)
