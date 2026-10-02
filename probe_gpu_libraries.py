#!/usr/bin/env python3
"""Report which TensorFlow GPU runtime libraries can be loaded."""

from __future__ import annotations

import ctypes


LIBRARIES = (
    "libcuda.so.1",
    "libcudart.so.12",
    "libcublasLt.so.12",
    "libcublas.so.12",
    "libcufft.so.11",
    "libcurand.so.10",
    "libcusolver.so.11",
    "libcusparse.so.12",
    "libnvJitLink.so.12",
    "libnvrtc.so.12",
    "libcupti.so.12",
    "libcudnn.so.9",
    "libnccl.so.2",
)


def main() -> int:
    failed = False
    for library in LIBRARIES:
        try:
            ctypes.CDLL(library, mode=ctypes.RTLD_GLOBAL)
        except OSError as error:
            failed = True
            print(f"FAIL {library}: {error}")
        else:
            print(f"OK   {library}")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
