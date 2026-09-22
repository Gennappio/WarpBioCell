"""Slow numpy references for the field kernels. Used only in tests.

Same discretisation as ``kernels/field_kernels.py``: 7-point Laplacian, Dirichlet faces
fixed, Neumann faces mirrored. ``steady_state_reference`` solves the linearised system
exactly (dense) inside a Picard loop, so it is the discrete truth on tiny grids.
"""

from __future__ import annotations

import numpy as np

from warpbiocell.fields.scalar_field import Boundary


def trilinear_terms(positions, origin, dx, shape):
    """Yield ``(cell_index, (i, j, k), weight)`` for the 8 nodes around each position inside the grid."""
    u = (np.asarray(positions, dtype=np.float64) - np.asarray(origin)) / dx
    base = np.floor(u).astype(int)
    frac = u - base
    for c in range(u.shape[0]):
        i0, j0, k0 = base[c]
        if i0 < 0 or j0 < 0 or k0 < 0 or i0 + 1 >= shape[0] or j0 + 1 >= shape[1] or k0 + 1 >= shape[2]:
            continue
        fx, fy, fz = frac[c]
        for di, wx in ((0, 1 - fx), (1, fx)):
            for dj, wy in ((0, 1 - fy), (1, fy)):
                for dk, wz in ((0, 1 - fz), (1, fz)):
                    yield c, (i0 + di, j0 + dj, k0 + dk), wx * wy * wz


def deposit_reference(positions, consuming, origin, dx, shape):
    """Number density [cells / dx^3] from unit weights of the cells flagged ``consuming``."""
    counts = np.zeros(shape, dtype=np.float64)
    for c, node, w in trilinear_terms(positions, origin, dx, shape):
        if consuming[c]:
            counts[node] += w
    return counts / dx**3


def sample_reference(values, positions, origin, dx):
    values = np.asarray(values, dtype=np.float64)
    shape = values.shape
    u = (np.asarray(positions, dtype=np.float64) - np.asarray(origin)) / dx
    u = np.clip(u, 0.0, np.array(shape) - 1.0)
    base = np.minimum(np.floor(u).astype(int), np.array(shape) - 2)
    frac = u - base
    out = np.zeros(u.shape[0])
    for c in range(u.shape[0]):
        i0, j0, k0 = base[c]
        fx, fy, fz = frac[c]
        for di, wx in ((0, 1 - fx), (1, fx)):
            for dj, wy in ((0, 1 - fy), (1, fy)):
                for dk, wz in ((0, 1 - fz), (1, fz)):
                    out[c] += values[i0 + di, j0 + dj, k0 + dk] * wx * wy * wz
    return out


def _mirror(idx, n):
    if idx < 0:
        return 1
    if idx > n - 1:
        return n - 2
    return idx


def _fixed_mask(shape, boundary, extra=None):
    mask = np.zeros(shape, dtype=bool) if extra is None else np.asarray(extra, dtype=bool).copy()
    for axis, b in enumerate(boundary):
        if Boundary(b) == Boundary.DIRICHLET:
            idx = [slice(None)] * 3
            idx[axis] = 0
            mask[tuple(idx)] = True
            idx[axis] = -1
            mask[tuple(idx)] = True
    return mask


def neighbor_sum_reference(values):
    """Sum of the 6 neighbours with mirroring at every edge (only meaningful on updated nodes)."""
    v = np.asarray(values, dtype=np.float64)
    nx, ny, nz = v.shape
    s = np.zeros_like(v)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                s[i, j, k] = (
                    v[_mirror(i - 1, nx), j, k]
                    + v[_mirror(i + 1, nx), j, k]
                    + v[i, _mirror(j - 1, ny), k]
                    + v[i, _mirror(j + 1, ny), k]
                    + v[i, j, _mirror(k - 1, nz)]
                    + v[i, j, _mirror(k + 1, nz)]
                )
    return s


def ftcs_step_reference(values, density, diffusion, uptake_max, michaelis_k, dx, dt, boundary, fixed_extra=None):
    v = np.asarray(values, dtype=np.float64)
    fixed = _fixed_mask(v.shape, boundary, fixed_extra)
    lap = diffusion / dx**2 * (neighbor_sum_reference(v) - 6.0 * v)
    uptake = np.asarray(density, dtype=np.float64) * uptake_max * v / (michaelis_k + v)
    out = v + dt * (lap - uptake)
    out[fixed] = v[fixed]
    return out


def steady_state_reference(values0, density, diffusion, uptake_max, michaelis_k, dx, boundary, picard_iterations=30, picard_tol=1e-12, fixed_extra=None):
    """Exact solution of the discrete steady-state problem (dense solve + Picard on the uptake)."""
    v = np.asarray(values0, dtype=np.float64).copy()
    density = np.asarray(density, dtype=np.float64)
    shape = v.shape
    fixed = _fixed_mask(shape, boundary, fixed_extra)
    unknown = np.argwhere(~fixed)
    index = -np.ones(shape, dtype=int)
    for row, (i, j, k) in enumerate(unknown):
        index[i, j, k] = row
    m = len(unknown)
    nx, ny, nz = shape

    def neighbors(i, j, k):
        return (
            (_mirror(i - 1, nx), j, k),
            (_mirror(i + 1, nx), j, k),
            (i, _mirror(j - 1, ny), k),
            (i, _mirror(j + 1, ny), k),
            (i, j, _mirror(k - 1, nz)),
            (i, j, _mirror(k + 1, nz)),
        )

    for _ in range(picard_iterations):
        a = np.zeros((m, m))
        rhs = np.zeros(m)
        for row, (i, j, k) in enumerate(unknown):
            c = density[i, j, k] * uptake_max / (michaelis_k + v[i, j, k])
            a[row, row] = -(6.0 + dx**2 * c / diffusion)
            for node in neighbors(i, j, k):
                col = index[node]
                if col >= 0:
                    a[row, col] += 1.0
                else:
                    rhs[row] -= v[node]
        sol = np.linalg.solve(a, rhs)
        new = v.copy()
        new[~fixed] = sol
        change = np.abs(new - v).max()
        v = new
        if change < picard_tol:
            break
    return v
