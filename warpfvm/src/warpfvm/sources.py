"""Point sources to per-cell volumetric sources.

MicroC computes a reaction rate per biological cell (mol/s) and needs a volumetric source per
grid cell (mol/(m^3 s) = mM/s). ``bin_rates`` does it for all points at once with a
deterministic ``numpy.bincount``: each point belongs to the grid cell that contains it
(``floor((position - origin) / spacing)``), points outside the grid are dropped.
"""

from __future__ import annotations

import numpy as np

from warpfvm.mesh import UniformGrid


def cell_ids(mesh: UniformGrid, positions) -> np.ndarray:
    """Grid-cell id of each point (FiPy's numbering), -1 for points outside the grid.

    ``positions`` has shape ``(n, dim)`` in the mesh's length unit and coordinate frame.
    """
    points = np.asarray(positions, dtype=np.float64)
    if points.ndim == 1 and mesh.dim == 1:
        points = points[:, None]
    if points.ndim != 2 or points.shape[1] != mesh.dim:
        raise ValueError(f"positions must have shape (n, {mesh.dim}), got {points.shape}")
    spacing = np.asarray(mesh._spacing[: mesh.dim])
    index = np.floor((points - mesh.origin) / spacing).astype(np.int64)
    inside = np.all((index >= 0) & (index < np.asarray(mesh.shape)), axis=1)
    ids = index[:, 0].copy()
    if mesh.dim >= 2:
        ids += mesh.nx * index[:, 1]
    if mesh.dim == 3:
        ids += mesh.nx * mesh.ny * index[:, 2]
    return np.where(inside, ids, -1)


def bin_rates(mesh: UniformGrid, positions, rates) -> np.ndarray:
    """Sum of the rates of the points in each grid cell divided by the cell volume."""
    ids = cell_ids(mesh, positions)
    rates = np.broadcast_to(np.asarray(rates, dtype=np.float64), ids.shape)
    inside = ids >= 0
    total = np.bincount(ids[inside], weights=rates[inside], minlength=mesh.numberOfCells)
    return total / mesh._cell_volume
