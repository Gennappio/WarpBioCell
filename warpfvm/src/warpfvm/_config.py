"""Device and precision defaults.

Every mesh carries a Warp device and a scalar type; every variable, coefficient and solver
buffer on that mesh uses them. They come from the mesh constructor (``device=``, ``dtype=``),
else from the environment (``WARPFVM_DEVICE``, ``WARPFVM_PRECISION``), else from Warp:

    device     Warp's preferred device (the first CUDA device when one exists, else the CPU)
    precision  float64, like FiPy; float32 halves memory traffic at ~1e-6 relative accuracy
"""

from __future__ import annotations

import os

import numpy as np
import warp as wp

_PRECISIONS = {"float64": wp.float64, "double": wp.float64, "float32": wp.float32, "single": wp.float32}


def resolve_device(device=None):
    wp.init()
    if device is None:
        device = os.environ.get("WARPFVM_DEVICE") or None
    if device is None or device == "auto":
        return wp.get_preferred_device()
    return wp.get_device(device)


def resolve_scalar(dtype=None):
    if dtype is None:
        dtype = os.environ.get("WARPFVM_PRECISION", "float64")
    if isinstance(dtype, str):
        try:
            return _PRECISIONS[dtype.lower()]
        except KeyError:
            raise ValueError(f"unknown precision {dtype!r}: use 'float64' or 'float32'") from None
    if dtype in (wp.float64, np.float64, float):
        return wp.float64
    if dtype in (wp.float32, np.float32):
        return wp.float32
    raise ValueError(f"unsupported dtype {dtype!r}: use float64 or float32")


def numpy_dtype(scalar):
    return np.float64 if scalar == wp.float64 else np.float32
