"""Checkpoints: every persistent cell array plus the field, as a compressed ``.npz``.

The RNG states are included, so a checkpoint carries everything needed to continue a run
bit-for-bit on the same device (restart logic itself is not implemented yet).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from warpbiocell.cells.state import CellPopulation
from warpbiocell.fields.oxygen import OxygenField

CELL_ARRAYS = ("position", "radius", "cell_state", "cell_type", "age", "oxygen_local", "glucose_local", "rng_state")


def save_checkpoint(path: str | Path, population: CellPopulation, oxygen: OxygenField | None, time_h: float, region=None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = population.count
    arrays = {name: getattr(population, name).numpy()[:n].copy() for name in CELL_ARRAYS}
    arrays["count"] = np.array(n)
    arrays["capacity"] = np.array(population.capacity)
    arrays["seed"] = np.array(population.seed)
    arrays["time_h"] = np.array(time_h)
    for name, values in population.extra_local.items():
        arrays[f"{name}_local"] = values.numpy()[:n].copy()
    if oxygen is not None:
        fields = getattr(oxygen, "species_fields", None) or {"oxygen": oxygen}
        for name, species in fields.items():
            arrays["field" if name == "oxygen" else f"field_{name}"] = species.numpy()
        arrays["density"] = fields["oxygen"].density.numpy()
        arrays["grid_origin"] = np.array(oxygen.geometry.origin)
        arrays["grid_dx"] = np.array(oxygen.geometry.dx)
    if region is not None:
        arrays["tissue_sdf"] = region.sdf_numpy()
        arrays["grid_origin"] = np.array(region.geometry.origin)
        arrays["grid_dx"] = np.array(region.geometry.dx)
    np.savez_compressed(path, **arrays)
    return path


def load_checkpoint(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path)) as data:
        return {key: data[key] for key in data.files}
