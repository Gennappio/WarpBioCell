"""Assembly: warpfvm's (L, b) against FiPy's, and against an independent numpy reference.

Computational validation (AGENTS.md): the Warp kernels must build exactly the matrix and
right-hand side that FiPy builds for the same equation, entry by entry, before any solver runs.
"""

import numpy as np
import pytest

import warpfvm as wf
from fvm_helpers import all_faces, fipy_system, grid, warp_system

GEOMETRIES = [
    ((6,), (2e-5,)),
    ((5, 4), (2e-5, 3e-5)),
    ((4, 3, 5), (2e-5, 2.5e-5, 3e-5)),
    ((3, 1, 2), (1e-5, 1e-5, 1e-5)),
]


def _equation(lib, mesh, case, rng):
    """One of several MicroC-shaped equations, built identically for FiPy and warpfvm."""
    n = mesh.numberOfCells
    D = 1.0e-9
    source = lib.CellVariable(mesh=mesh, value=np.where(rng.random(n) < 0.4, -rng.random(n) * 1e-3, 0.0))
    sink = lib.CellVariable(mesh=mesh, value=rng.random(n) * 0.05)
    mixed = lib.CellVariable(mesh=mesh, value=rng.normal(size=n) * 0.05)  # sinks and "wrong-sign" sources
    if case == "steady_source":
        return lib.DiffusionTerm(coeff=D) == -source, None
    if case == "steady_decay":
        return lib.DiffusionTerm(coeff=D) - lib.ImplicitSourceTerm(coeff=0.02) == -source, None
    if case == "steady_sink_field":
        return lib.DiffusionTerm(coeff=D) - lib.ImplicitSourceTerm(coeff=sink) == -source, None
    if case == "steady_mixed_signs":
        return lib.DiffusionTerm(coeff=D) - lib.ImplicitSourceTerm(coeff=mixed) + source == 0, None
    if case == "transient":
        return lib.TransientTerm() == lib.DiffusionTerm(coeff=D) - lib.ImplicitSourceTerm(coeff=sink) + source, 30.0
    if case == "transient_mixed_signs":
        return lib.TransientTerm(coeff=2.0) == lib.DiffusionTerm(coeff=D) + lib.ImplicitSourceTerm(coeff=mixed), 5.0
    if case == "scaled_terms":
        return 2.0 * lib.DiffusionTerm(coeff=D) - lib.ImplicitSourceTerm(coeff=0.5 * 0.02) * 3 == -source, None
    raise ValueError(case)


CASES = ["steady_source", "steady_decay", "steady_sink_field", "steady_mixed_signs", "transient", "transient_mixed_signs", "scaled_terms"]


def _constrain(lib, mesh, var, boundary):
    if boundary == "all":
        var.constrain(0.07, all_faces(mesh))
    elif boundary == "some":
        var.constrain(0.07, mesh.facesLeft)
        if len(mesh.shape) >= 2:
            var.constrain(0.02, mesh.facesTop)
    elif boundary == "per_face":
        fc = np.asarray(mesh.faceCenters)
        var.constrain(1.0 + fc[0] / fc[0].max(), all_faces(mesh))


@pytest.mark.parametrize("dims, spacing", GEOMETRIES)
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("boundary", ["all", "some", "per_face"])
def test_assembly_matches_fipy(dims, spacing, case, boundary):
    fp = pytest.importorskip("fipy")
    systems = []
    for lib in (fp, wf):
        rng = np.random.default_rng(7)
        mesh = grid(lib, dims, spacing, device="cpu")
        var = lib.CellVariable(mesh=mesh, value=0.05 + 0.01 * rng.random(mesh.numberOfCells))
        _constrain(lib, mesh, var, boundary)
        eq, dt = _equation(lib, mesh, case, rng)
        systems.append(fipy_system(eq, var, dt) if lib is fp else warp_system(eq, var, dt))
    (L_fp, b_fp), (L_wf, b_wf) = systems
    assert np.abs(L_wf - L_fp).max() <= 1e-12 * np.abs(L_fp).max()
    assert np.abs(b_wf - b_fp).max() <= 1e-12 * max(np.abs(b_fp).max(), 1e-300)


def test_under_relaxation_matches_fipy():
    fp = pytest.importorskip("fipy")
    out = []
    for lib in (fp, wf):
        rng = np.random.default_rng(3)
        mesh = grid(lib, (4, 3, 3), (1e-5, 1e-5, 1e-5), device="cpu")
        var = lib.CellVariable(mesh=mesh, value=rng.random(mesh.numberOfCells))
        var.constrain(0.1, mesh.facesRight)
        eq = lib.DiffusionTerm(coeff=2e-9) - lib.ImplicitSourceTerm(coeff=0.1) == -1e-4
        if lib is fp:
            from fipy.solvers.scipy import LinearLUSolver

            solver = LinearLUSolver()
            solver = eq._prepareLinearSystem(var, solver, (), None)
            solver._applyUnderRelaxation(0.7)
            out.append((solver.matrix.matrix.toarray(), np.asarray(solver.RHSvector)))
        else:
            out.append(warp_system(eq, var, underRelaxation=0.7))
    (L_fp, b_fp), (L_wf, b_wf) = out
    assert np.abs(L_wf - L_fp).max() <= 1e-12 * np.abs(L_fp).max()
    assert np.abs(b_wf - b_fp).max() <= 1e-12 * np.abs(b_fp).max()


# --- an independent reference: FiPy's discretization written out with loops --------------------


def _reference(dims, spacing, D, sink, source, fixed_value, fixed_mask_by_side):
    """Steady ``D lap(phi) - sink phi + source = 0`` with Dirichlet sides, in FiPy's (L, b)."""
    nx, ny, nz = tuple(dims) + (1,) * (3 - len(dims))
    d = tuple(spacing) + (1.0,) * (3 - len(spacing))
    V = d[0] * d[1] * d[2]
    area = (d[1] * d[2], d[0] * d[2], d[0] * d[1])
    n = nx * ny * nz
    L = np.zeros((n, n))
    b = np.zeros(n)
    counts = (nx, ny, nz)
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                p = i + nx * (j + ny * k)
                idx = [i, j, k]
                for axis in range(len(dims)):
                    T = D * area[axis] / d[axis]
                    for direction, side in ((-1, 2 * axis), (1, 2 * axis + 1)):
                        q = list(idx)
                        q[axis] += direction
                        if 0 <= q[axis] < counts[axis]:
                            nb = q[0] + nx * (q[1] + ny * q[2])
                            L[p, p] -= T
                            L[p, nb] += T
                        elif fixed_mask_by_side[side]:
                            L[p, p] -= 2 * T
                            b[p] -= 2 * T * fixed_value
                L[p, p] -= sink[p] * V  # -ImplicitSourceTerm(sink): implicit, on the diagonal
                b[p] -= source[p] * V  # + source: explicit
    return L, b


@pytest.mark.parametrize("dims, spacing", GEOMETRIES)
def test_assembly_matches_loop_reference(dims, spacing, rng):
    mesh = grid(wf, dims, spacing, device="cpu")
    n = mesh.numberOfCells
    sink, source = rng.random(n) * 0.05, rng.random(n) * 1e-3
    var = wf.CellVariable(mesh=mesh, value=0.0)
    var.constrain(0.07, mesh.facesLeft | (mesh.facesTop if len(dims) >= 2 else mesh.facesLeft))
    eq = wf.DiffusionTerm(coeff=1e-9) - wf.ImplicitSourceTerm(coeff=sink) + source == 0
    L_wf, b_wf = warp_system(eq, var)
    sides = [True, False, False, len(dims) >= 2, False, False]
    L_ref, b_ref = _reference(dims, spacing, 1e-9, sink, source, 0.07, sides)
    assert np.abs(L_wf - L_ref).max() <= 1e-12 * np.abs(L_ref).max()
    assert np.abs(b_wf - b_ref).max() <= 1e-12 * np.abs(b_ref).max()


def test_scaled_system_is_symmetric_with_positive_diagonal(rng):
    mesh = wf.Grid3D(dx=2e-5, dy=2e-5, dz=2e-5, nx=5, ny=4, nz=3, device="cpu")
    var = wf.CellVariable(mesh=mesh, value=0.0)
    var.constrain(0.07, mesh.exteriorFaces)
    for eq, dt in [
        (wf.DiffusionTerm(coeff=1e-9) - wf.ImplicitSourceTerm(coeff=rng.random(60)) == -1e-3, None),
        (wf.TransientTerm() == wf.DiffusionTerm(coeff=1e-9) - wf.ImplicitSourceTerm(coeff=0.01), 10.0),
    ]:
        system = eq.assemble(var=var, dt=dt)
        A = (system.to_scipy()[0] / system.scale).toarray()
        assert np.allclose(A, A.T, rtol=0.0, atol=1e-14)
        assert np.all(np.diag(A) > 0.0)
        assert np.all(np.linalg.eigvalsh(A) > 0.0)  # SPD: conjugate gradient applies
        assert 0.5 < np.median(np.diag(A)) < 2.0  # the diagonal is of order one after scaling


def test_transient_needs_dt_and_rejects_unsupported_terms():
    mesh = wf.Grid1D(dx=1.0, nx=4, device="cpu")
    var = wf.CellVariable(mesh=mesh)
    with pytest.raises(TypeError):
        (wf.TransientTerm() == wf.DiffusionTerm()).solve(var=var)
    with pytest.raises(NotImplementedError):
        wf.DiffusionTerm(coeff=var)
    with pytest.raises(NotImplementedError):
        wf.DiffusionTerm(coeff=(1.0, 2.0))
    with pytest.raises(ValueError):
        (wf.DiffusionTerm() == 0).solve()
    other = wf.CellVariable(mesh=mesh)
    with pytest.raises(NotImplementedError):
        (wf.DiffusionTerm(var=other) == 0).solve(var=var)
