from __future__ import annotations

import argparse
import ctypes
import json
import os
import tempfile
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import pandas as pd
import tensorflow as tf

from train_multitask import runtime_path


JPEG_WARNING_MARKERS = (
    "Corrupt JPEG data",
    "Premature end of JPEG",
    "Invalid JPEG",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Decode every JPEG with TensorFlow while associating libjpeg warnings with "
            "the exact manifest row. Source files are never modified."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-images", type=int)
    parser.add_argument(
        "--fail-on-warning",
        action="store_true",
        help="Exit non-zero for recoverable decoder warnings as well as hard failures.",
    )
    return parser.parse_args()


def flush_c_streams() -> None:
    try:
        ctypes.CDLL(None).fflush(None)
    except (AttributeError, OSError):
        pass


def decode_with_captured_stderr(path: str) -> tuple[tuple[int, int, int] | None, str, str]:
    saved_stderr = os.dup(2)
    error = ""
    shape: tuple[int, int, int] | None = None
    with tempfile.TemporaryFile() as captured:
        try:
            os.dup2(captured.fileno(), 2)
            image = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
            shape = tuple(int(value) for value in image.shape)
            _ = image.numpy()
        except Exception as exception:
            error = f"{type(exception).__name__}: {exception}"
        finally:
            flush_c_streams()
            os.dup2(saved_stderr, 2)
            os.close(saved_stderr)
        captured.seek(0)
        raw_stderr = captured.read().decode("utf-8", errors="replace").strip()
        warning = "\n".join(
            line
            for line in raw_stderr.splitlines()
            if any(marker in line for marker in JPEG_WARNING_MARKERS)
        )
    return shape, warning, error


def main() -> None:
    args = parse_args()
    if not args.manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")
    frame = pd.read_csv(args.manifest, low_memory=False)
    if "image_path" not in frame:
        raise ValueError("Manifest must contain image_path.")
    if args.max_images is not None:
        if args.max_images < 1:
            raise ValueError("--max-images must be positive.")
        frame = frame.head(args.max_images)

    try:
        tf.config.set_visible_devices([], "GPU")
    except RuntimeError:
        pass
    _ = tf.constant(0).numpy()

    findings: list[dict[str, object]] = []
    dimension_mismatches = 0
    for position, (_, row) in enumerate(frame.iterrows(), start=1):
        path = runtime_path(str(row["image_path"]))
        shape, warning, error = decode_with_captured_stderr(path)
        expected_width = pd.to_numeric(pd.Series([row.get("width")]), errors="coerce").iloc[0]
        expected_height = pd.to_numeric(pd.Series([row.get("height")]), errors="coerce").iloc[0]
        dimension_mismatch = False
        if shape is not None and pd.notna(expected_width) and pd.notna(expected_height):
            dimension_mismatch = (
                shape[0] != int(expected_height) or shape[1] != int(expected_width)
            )
        if dimension_mismatch:
            dimension_mismatches += 1
        if warning or error or dimension_mismatch:
            findings.append(
                {
                    "row_position": position,
                    "source": row.get("source", ""),
                    "image_id": row.get("image_id", ""),
                    "lesion_id": row.get("lesion_id", ""),
                    "image_path": str(row["image_path"]),
                    "runtime_image_path": path,
                    "sha256": row.get("sha256", ""),
                    "decoded_shape": None if shape is None else list(shape),
                    "expected_width": None if pd.isna(expected_width) else int(expected_width),
                    "expected_height": None if pd.isna(expected_height) else int(expected_height),
                    "dimension_mismatch": dimension_mismatch,
                    "decoder_warning": warning,
                    "decode_error": error,
                }
            )
        if position % 500 == 0:
            print(f"Decoded {position}/{len(frame)} images; findings={len(findings)}")

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(findings).to_csv(output, index=False)
    warning_count = sum(bool(item["decoder_warning"]) for item in findings)
    error_count = sum(bool(item["decode_error"]) for item in findings)
    summary = {
        "manifest": str(args.manifest.resolve()),
        "images_checked": int(len(frame)),
        "decoder_warnings": int(warning_count),
        "decode_errors": int(error_count),
        "dimension_mismatches": int(dimension_mismatches),
        "findings_csv": str(output),
        "source_files_modified": False,
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    if error_count or dimension_mismatches or (args.fail_on_warning and warning_count):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
