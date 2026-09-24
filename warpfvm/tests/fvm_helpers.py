"""Shared problem builders: the same Python code runs against FiPy and warpfvm."""

from __future__ import annotations

import numpy as np

import warpfvm


def grid(lib, dims, spacing, **kwargs):
    """``lib.Grid{1,2,3}D`` with the given cell counts and spacings (warpfvm kwargs: device, dtype)."""
    names = [("dx", "nx"), ("dy", "ny"), ("dz", "nz")]
    args = {}
    for (d_name, n_name), n, d in zip(names, dims, spacing):
        args[d_name], args[n_name] = d, n
    factory = {1: lib.Grid1D, 2: lib.Grid2D, 3: lib.Grid3D}[len(dims)]
    if lib is warpfvm:
        args.update(kwargs)
    return factory(**args)


def all_faces(mesh):
    masks = [mesh.facesLeft, mesh.facesRight]
    if len(mesh.shape) >= 2:
        masks += [mesh.facesBottom, mesh.facesTop]
    if len(mesh.shape) == 3:
        masks += [mesh.facesFront, mesh.facesBack]
    out = masks[0]
    for m in masks[1:]:
        out = out | m
    return out


def fipy_system(eq, var, dt=None):
    """FiPy's assembled (L dense, b) for ``eq`` in ``var``, without solving."""
    from fipy.solvers.scipy import LinearLUSolver

    _, L, b = eq._buildAndAddMatrices(
        var=var,
        SparseMatrix=LinearLUSolver()._matrixClass,
        dt=dt,
        transientGeomCoeff=eq._getTransientGeomCoeff(var),
        diffusionGeomCoeff=eq._getDiffusionGeomCoeff(var),
    )
    return L.matrix.toarray(), np.asarray(b, dtype=np.float64)


def warp_system(eq, var, dt=None, underRelaxation=None):
    L, b = eq.assemble(var=var, dt=dt, underRelaxation=underRelaxation).to_scipy()
    return L.toarray(), b
