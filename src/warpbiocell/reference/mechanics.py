"""O(N^2) numpy reference of one contact substep. Slow by design; used only in tests."""

from __future__ import annotations

import numpy as np

from warpbiocell.kernels.mechanics_kernels import COINCIDENT_EPS


def contact_velocities_reference(
    positions: np.ndarray, radii: np.ndarray, rate: float, query_radius: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(velocities, neighbor_counts)`` over all pairs, same law as the Warp kernel.

    ``neighbor_counts[i]`` is the number of other cells whose centre lies within
    ``query_radius`` of cell ``i``; velocities only receive contributions from overlapping pairs.
    """
    x = np.asarray(positions, dtype=np.float64)
    r = np.asarray(radii, dtype=np.float64)

    d_vec = x[:, None, :] - x[None, :, :]  # (n, n, 3): x_i - x_j
    d = np.linalg.norm(d_vec, axis=2)
    np.fill_diagonal(d, np.inf)

    neighbors = d < query_radius
    overlap = r[:, None] + r[None, :] - d
    active = neighbors & (overlap > 0.0) & (d > float(COINCIDENT_EPS))

    with np.errstate(divide="ignore", invalid="ignore"):
        coeff = np.where(active, rate * overlap / d, 0.0)
    velocities = np.einsum("ij,ijk->ik", coeff, d_vec)
    counts = neighbors.sum(axis=1).astype(np.int32)
    return velocities, counts


def contact_substep_reference(
    positions: np.ndarray, radii: np.ndarray, rate: float, query_radius: float, dt: float
) -> np.ndarray:
    v, _ = contact_velocities_reference(positions, radii, rate, query_radius)
    return np.asarray(positions, dtype=np.float64) + v * dt
