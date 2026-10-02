"""Frozen PanDerm extraction preserving the complete frame, with resumable chunks."""
import argparse
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from compare_model_runs import file_hash
from context_study_common import full_frame_view, verify_context_plan
from extract_foundation_features import checked_image_bytes, CHUNK_SIZE
from foundation_common import ASSETS, fingerprint, read_chunk, save_chunk, save_json, split_frame


def extract(output):
    plan = verify_context_plan(output)
    import torch
    import torchvision
    import timm
    import PIL
    from PIL import Image
    from torchvision import transforms
    if not torch.cuda.is_available():
        raise RuntimeError("GPU unavailable; no silent CPU fallback.")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    spec = importlib.util.spec_from_file_location("pinned_context_panderm", ASSETS / "modeling_finetune.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    model = module.panderm_base_patch16_224()
    weights = torch.load(ASSETS / "panderm_base.pth", map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(weights, dict) or not all(isinstance(value, torch.Tensor) for value in weights.values()):
        raise ValueError("Unexpected pretrained checkpoint.")
    missing, unexpected = model.load_state_dict(weights, strict=False)
    if set(missing) - {"head.weight", "head.bias"} or unexpected:
        raise ValueError("Incomplete pretrained encoder.")
    del weights
    model.head = torch.nn.Identity()
    model.requires_grad_(False).eval().to("cuda")
    transform = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize((.485, .456, .406), (.228, .224, .225))])
    runtime = {"torch": torch.__version__, "torchvision": torchvision.__version__, "timm": timm.__version__,
        "pillow": PIL.__version__, "gpu": torch.cuda.get_device_name(0), "autocast": "float16",
        "preprocessing": "Full RGB frame; aspect-preserving bilinear resize into 224 square; centered mean-color padding (124,116,104); official mean/std"}
    print(json.dumps({"pid": os.getpid(), "runtime": runtime}), flush=True)
    destination = output / "features/panderm_full_frame"
    destination.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    for split in ("train", "val"):
        frame = split_frame(split)
        cache = {"version": 1, "encoder": "panderm_full_frame", "runtime": runtime,
            "dimensions": 768, "batch_size": 4, "chunk_size": CHUNK_SIZE, "rows": len(frame),
            "input_sha256": plan["input_sha256"], "code_sha256": plan["code_sha256"],
            "plan_sha256": file_hash(output / "experiment_plan.json"), "smoke_only": False}
        manifest = destination / f"{split}_manifest.json"
        if manifest.exists() and json.loads(manifest.read_text()) != cache:
            raise ValueError("Context cache/runtime changed; refusing silent reuse.")
        save_json(manifest, cache)
        for start in range(0, len(frame), CHUNK_SIZE):
            rows = frame.iloc[start:start + CHUNK_SIZE]
            key = fingerprint({"manifest": cache, "start": start,
                               "images": rows[["source", "image_id", "sha256"]].values.tolist()})
            path = destination / f"{split}_{start:06d}.npz"
            if path.exists():
                read_chunk(path, key, len(rows), 768)
            else:
                features = []
                for batch_start in range(0, len(rows), 4):
                    tensors = []
                    for row in rows.iloc[batch_start:batch_start + 4].itertuples():
                        with Image.open(io.BytesIO(checked_image_bytes(row))) as image:
                            tensors.append(transform(full_frame_view(image)))
                    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                        values = model(torch.stack(tensors).to("cuda")).float().cpu().numpy()
                    features.append(values)
                save_chunk(path, np.concatenate(features).astype("float32"), key)
            progress = {"pid": os.getpid(), "split": split, "completed_images": start + len(rows),
                "total_images": len(frame), "elapsed_seconds": round(time.monotonic() - started, 1)}
            save_json(destination / "progress.json", progress)
            print(json.dumps(progress), flush=True)
    save_json(destination / "complete.json", {"smoke_only": False, "runtime": runtime,
        "outputs_sha256": {p.name: file_hash(p) for p in sorted(destination.glob("*.npz"))}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    extract(parser.parse_args().experiment_dir.resolve())
