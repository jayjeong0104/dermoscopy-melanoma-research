"""Full-frame preprocessing and integrity checks for the context follow-up."""
import json
from pathlib import Path

from PIL import Image

from compare_model_runs import file_hash
from foundation_common import ROOT


def full_frame_view(image):
    image = image.convert("RGB")
    scale = 224 / max(image.size)
    size = tuple(max(1, min(224, round(side * scale))) for side in image.size)
    view = Image.new("RGB", (224, 224), (124, 116, 104))
    resized = image.resize(size, Image.Resampling.BILINEAR)
    view.paste(resized, ((224 - size[0]) // 2, (224 - size[1]) // 2))
    return view


def verify_context_plan(output: Path):
    plan = json.loads((output / "experiment_plan.json").read_text())
    for name, expected in plan["code_sha256"].items():
        if file_hash(ROOT / name) != expected:
            raise ValueError(f"Context study source changed: {name}")
    for name, expected in plan["input_sha256"].items():
        if file_hash(Path(name)) != expected:
            raise ValueError(f"Context study input changed: {name}")
    return plan
