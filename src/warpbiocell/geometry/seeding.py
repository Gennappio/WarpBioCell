"""Seed cells inside a tissue region."""

from __future__ import annotations

import numpy as np

from warpbiocell.geometry.region import TissueRegion


class CellBudgetError(ValueError):
    """The region holds more cells than the run is allowed to allocate."""


def fill_region(
    region: TissueRegion,
    radius: float,
    spacing_factor: float = 1.0,
    jitter: float = 0.1,
    seed: int = 0,
    margin: float = 0.0,
    max_cells: int | None = None,
    within=None,
) -> np.ndarray:
    """Positions [um] of cells on a jittered cubic lattice that lie inside the region.

    A lattice point is kept when its centre is at least ``radius + margin`` inside the
    surface (so the whole sphere is inside), and, if ``within`` is given (a shape with an
    ``sdf`` method), also inside that sub-volume. ``max_cells`` guards against filling a
    region the cell budget cannot hold: the count is reported in the error so the caller can
    choose a sub-volume, a coarser representation or a larger budget.
    """
    if radius <= 0.0:
        raise ValueError("radius must be positive")
    rng = np.random.default_rng(seed)
    spacing = spacing_factor * 2.0 * radius
    lo, hi = np.asarray(region.geometry.origin), np.asarray(region.geometry.upper)
    axes = [np.arange(lo[a] + 0.5 * spacing, hi[a], spacing) for a in range(3)]
    gx, gy, gz = np.meshgrid(*axes, indexing="ij")
    points = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    points += rng.uniform(-jitter * radius, jitter * radius, size=points.shape)

    keep = region.contains(points, margin=radius + margin)
    if within is not None:
        keep &= within.sdf(points[:, 0], points[:, 1], points[:, 2]) < -(radius + margin)
    points = points[keep]
    if max_cells is not None and points.shape[0] > max_cells:
        raise CellBudgetError(
            f"the region holds {points.shape[0]} cells at this packing, above the budget of {max_cells}; "
            "seed a sub-volume (within=...), raise cells.max_cells or coarsen the representation"
        )
    return points.astype(np.float32)
