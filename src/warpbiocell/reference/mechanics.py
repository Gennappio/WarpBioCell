"""O(N^2) numpy reference of one contact substep. Slow by design; used only in tests."""

from __future__ import annotations

import numpy as np

from warpbiocell.kernels.mechanics_kernels import COINCIDENT_EPS


def contact_velocities_reference(
    positions: np.ndarray, radii: np.ndarray, rate: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(velocities, contact_counts)`` for all pairs, same law as the Warp kernel."""
    x = np.asarray(positions, dtype=np.float64)
    r = np.asarray(radii, dtype=np.float64)
    n = x.shape[0]

    d_vec = x[:, None, :] - x[None, :, :]  # (n, n, 3): x_i - x_j
    d = np.linalg.norm(d_vec, axis=2)
    overlap = r[:, None] + r[None, :] - d
    np.fill_diagonal(overlap, -np.inf)
    active = (overlap > 0.0) & (d > float(COINCIDENT_EPS))

    with np.errstate(divide="ignore", invalid="ignore"):
        coeff = np.where(active, rate * overlap / d, 0.0)
    velocities = np.einsum("ij,ijk->ik", coeff, d_vec)
    counts = active.sum(axis=1).astype(np.int32)
    return velocities, counts


def contact_substep_reference(positions: np.ndarray, radii: np.ndarray, rate: float, dt: float) -> np.ndarray:
    v, _ = contact_velocities_reference(positions, radii, rate)
    return np.asarray(positions, dtype=np.float64) + v * dt
