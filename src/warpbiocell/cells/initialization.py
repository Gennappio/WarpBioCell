"""Initial cell configurations. Pure numpy; the result is uploaded with CellPopulation.from_numpy."""

from __future__ import annotations

import numpy as np


def spherical_cluster(
    n_cells: int,
    radius: float,
    spacing_factor: float = 0.9,
    jitter: float = 0.1,
    seed: int = 0,
) -> np.ndarray:
    """Return ``(n_cells, 3)`` positions [um] forming an approximately spherical cluster.

    Cells sit on a cubic lattice with spacing ``spacing_factor * 2 * radius`` (values below 1
    create controlled initial overlap that the mechanics must relax), perturbed by uniform
    jitter of amplitude ``jitter * radius``, and the ``n_cells`` lattice points closest to the
    origin are kept. The lattice guarantees that no two centres coincide, which the contact
    kernel relies on (coincident centres have no defined separation direction and are skipped).
    """
    if n_cells <= 0:
        raise ValueError("n_cells must be positive")
    rng = np.random.default_rng(seed)

    spacing = spacing_factor * 2.0 * radius
    # Side of the smallest cube of lattice points that surely contains n_cells points inside its
    # inscribed sphere (sphere/cube volume ratio is pi/6 ~ 0.52).
    half = int(np.ceil(((n_cells / (np.pi / 6.0)) ** (1.0 / 3.0)) / 2.0)) + 1
    axis = np.arange(-half, half + 1, dtype=np.float64) * spacing
    gx, gy, gz = np.meshgrid(axis, axis, axis, indexing="ij")
    lattice = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)

    order = np.argsort(np.einsum("ij,ij->i", lattice, lattice), kind="stable")
    points = lattice[order[:n_cells]]
    points += rng.uniform(-jitter * radius, jitter * radius, size=points.shape)
    return points.astype(np.float32)
