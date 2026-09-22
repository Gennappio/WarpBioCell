"""Device selection.

Warp has no CUDA backend on macOS, so the development machine runs CPU-only. Nothing in the
package hard-codes "cuda"; callers resolve a device here and pass it down.
"""

from __future__ import annotations

import warp as wp


def resolve_device(name: str | None = "auto") -> wp.context.Device:
    """Return the Warp device named by ``name``.

    ``None`` or ``"auto"`` selects the first CUDA device when one exists, otherwise the CPU.
    Any other value (``"cpu"``, ``"cuda:0"``, ...) is passed to Warp unchanged.
    """
    wp.init()
    if name is None or name == "auto":
        return wp.get_device("cuda:0") if wp.is_cuda_available() else wp.get_device("cpu")
    return wp.get_device(name)


def cuda_available() -> bool:
    wp.init()
    return wp.is_cuda_available()


def synchronize(device: wp.context.Device) -> None:
    """Block until all work queued on ``device`` has finished (no-op on CPU)."""
    if device.is_cuda:
        wp.synchronize_device(device)
