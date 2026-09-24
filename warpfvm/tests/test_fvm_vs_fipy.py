"""Solutions against FiPy, on problems shaped like MicroC's.

The functions below are written once and run twice, with ``lib = fipy`` and ``lib = warpfvm``:
the same code path MicroC would take after changing its imports.
"""

import numpy as np
import pytest

import warpfvm as wf
from fvm_helpers import all_faces, grid

fp = pytest.importorskip("fipy")
from fipy.solvers.scipy import LinearGMRESSolver as FipyGMRES  # noqa: E402
from fipy.solvers.scipy import LinearLUSolver as FipyLU  # noqa: E402

LIBS = {"fipy": (fp, FipyLU, FipyGMRES), "warpfvm": (wf, wf.LinearLUSolver, wf.LinearGMRESSolver)}

# MicroC's substances (MicroC adapter workflows): D [m^2/s], boundary [mM]
SUBSTANCES = {
    "Oxygen": dict(D=1.0e-9, boundary=0.07, initial=0.07, decay=0.0),
    "Glucose": dict(D=6.7e-10, boundary=5.0, initial=5.0, decay=0.0),
    "Lactate": dict(D=6.7e-10, boundary=1.0, initial=1.0, decay=0.0),
    "TGFA": dict(D=5.18e-11, boundary=0.0, initial=0.0, decay=1.0e-4),
}


def microc_setup(lib, dims, size_m, boundary_type="fixed"):
    """MeshManager + MultiSubstanceSimulator._create_fipy_variables_for_substances."""
    spacing = [size_m / n for n in dims]
    mesh = grid(lib, dims, spacing, device="cpu")
    variables = {}
    for name, cfg in SUBSTANCES.items():
        var = lib.CellVariable(name=name, mesh=mesh, value=cfg["initial"])
        if boundary_type == "fixed":
            var.constrain(cfg["boundary"], all_faces(mesh))
        else:  # MicroC's "linear_gradient": 0 left, 1 right, linear in x on top and bottom
            var.constrain(0.0, mesh.facesLeft)
            var.constrain(1.0, mesh.facesRight)
            fc = np.asarray(mesh.faceCenters)
            ramp = np.clip(fc[0] / size_m, 0.0, 1.0)
            var.constrain(ramp, mesh.facesTop | mesh.facesBottom)
        variables[name] = var
    return mesh, variables


def microc_update(lib, lu, gmres, mesh, variables, sources, sinks=None, boundary_type="fixed", transient_dt=None):
    """MultiSubstanceSimulator.update, line for line (solver choice included)."""
    for name, var in variables.items():
        cfg = SUBSTANCES[name]
        source_var = lib.CellVariable(mesh=mesh, value=sources[name])
        decay_rate = cfg["decay"]
        sink_field = None if sinks is None else sinks.get(name)
        if sink_field is not None:
            sink_var = lib.CellVariable(mesh=mesh, value=sink_field + decay_rate)
            steady_terms = lib.DiffusionTerm(coeff=cfg["D"]) - lib.ImplicitSourceTerm(coeff=sink_var)
        elif decay_rate > 0.0:
            steady_terms = lib.DiffusionTerm(coeff=cfg["D"]) - lib.ImplicitSourceTerm(coeff=decay_rate)
        else:
            steady_terms = lib.DiffusionTerm(coeff=cfg["D"])
        if transient_dt is not None:
            equation = lib.TransientTerm() == steady_terms + source_var
        else:
            equation = steady_terms == -source_var
        if transient_dt is not None or boundary_type == "fixed" or decay_rate > 0.0 or sink_field is not None:
            solver = lu(iterations=10, tolerance=1e-6)
        else:
            solver = gmres(iterations=1000, tolerance=1e-6)
        if transient_dt is not None:
            equation.solve(var=var, solver=solver, dt=float(transient_dt))
        else:
            equation.solve(var=var, solver=solver)


def spheroid_sources(mesh_dims, size_m, rng):
    """Per-voxel volumetric rates [mM/s] of a spheroid of cells at the centre (MicroC's units)."""
    n = int(np.prod(mesh_dims))
    spacing = [size_m / d for d in mesh_dims]
    volume = np.prod(spacing) if len(mesh_dims) == 3 else np.prod(spacing)
    idx = np.indices(mesh_dims[::-1]).reshape(len(mesh_dims), -1)[::-1]  # (i, j[, k]) per cell, x fastest
    centre = [(d - 1) / 2 for d in mesh_dims]
    r2 = sum(((idx[a] - centre[a]) * spacing[a]) ** 2 for a in range(len(mesh_dims)))
    cells = np.where(r2 < (0.3 * size_m) ** 2, rng.integers(1, 9, size=n), 0)  # cells per voxel
    per_cell = {"Oxygen": -3.0e-17, "Glucose": -5.0e-17, "Lactate": 1.0e-16, "TGFA": 2.0e-20}  # mol/s/cell
    return {name: cells * rate / volume for name, rate in per_cell.items()}, {"TGFA": cells * 1e-3}


def _run(which, dims, size, boundary_type="fixed", transient_dt=None, steps=1, with_sinks=True):
    lib, lu, gmres = LIBS[which]
    rng = np.random.default_rng(11)
    mesh, variables = microc_setup(lib, dims, size, boundary_type)
    sources, sinks = spheroid_sources(dims, size, rng)
    for _ in range(steps):
        microc_update(lib, lu, gmres, mesh, variables, sources, sinks if with_sinks else None, boundary_type, transient_dt)
    return {name: np.asarray(var.value, dtype=np.float64) for name, var in variables.items()}


def _assert_close(ours, theirs, rel):
    """Differences within ``rel`` of the field's variation, or at round-off of its magnitude
    (the CG answer is converged to a 1e-12 relative residual, FiPy's LU to machine precision)."""
    for name in theirs:
        spread = np.ptp(theirs[name])
        magnitude = np.abs(theirs[name]).max()
        err = np.abs(ours[name] - theirs[name]).max()
        assert err <= rel * spread + 1e-11 * magnitude, f"{name}: max |diff| {err:.3g}, variation {spread:.3g}, magnitude {magnitude:.3g}"


GEOMETRIES = [((15, 15, 15), 750e-6), ((12, 10, 8), 600e-6), ((30, 30), 1500e-6), ((50, 50), 1500e-6)]


@pytest.mark.parametrize("dims, size", GEOMETRIES)
def test_steady_microc_update_matches_fipy(dims, size):
    _assert_close(_run("warpfvm", dims, size), _run("fipy", dims, size), rel=1e-9)


@pytest.mark.parametrize("dims, size", GEOMETRIES[:3])
def test_transient_microc_update_matches_fipy(dims, size):
    kwargs = dict(transient_dt=60.0, steps=5)
    _assert_close(_run("warpfvm", dims, size, **kwargs), _run("fipy", dims, size, **kwargs), rel=1e-9)


def test_gradient_boundaries_with_fipy_gmres():
    """MicroC routes non-"fixed" boundaries to GMRES with tolerance 1e-6: both answers agree to
    the accuracy that tolerance allows, and both converge to the same tight solution."""
    dims, size = (20, 16), 1000e-6
    ours = _run("warpfvm", dims, size, boundary_type="linear_gradient", with_sinks=False)
    theirs = _run("fipy", dims, size, boundary_type="linear_gradient", with_sinks=False)
    _assert_close(ours, theirs, rel=1e-4)


def _picard(which, dims, size, alpha=0.7, tolerance=1e-10, max_iterations=200):
    """MicroC's coupled loop (run_diffusion_solver_coupled): Michaelis-Menten oxygen uptake that
    depends on the local oxygen, under-relaxed reaction terms, stop when two consecutive solves
    agree. Same code for both libraries."""
    lib, lu, _ = LIBS[which]
    spacing = [size / n for n in dims]
    mesh = grid(lib, dims, spacing, device="cpu")
    oxygen = lib.CellVariable(mesh=mesh, value=0.07)
    oxygen.constrain(0.07, all_faces(mesh))
    rng = np.random.default_rng(2)
    cells = np.where(rng.random(mesh.numberOfCells) < 0.5, rng.integers(1, 30, mesh.numberOfCells), 0)
    q_max, K = 3.0e-17 / np.prod(spacing), 0.005  # mol/s/cell -> mM/s ; mM
    rate = None
    previous = np.array(oxygen.value, dtype=np.float64)  # a copy: FiPy's value is its live array
    for iteration in range(1, max_iterations + 1):
        c = np.clip(np.array(oxygen.value, dtype=np.float64), 0.0, None)
        new_rate = -cells * q_max * c / (K + c)
        rate = new_rate if rate is None else alpha * new_rate + (1 - alpha) * rate
        source = lib.CellVariable(mesh=mesh, value=rate)
        (lib.DiffusionTerm(coeff=1e-9) == -source).solve(var=oxygen, solver=lu(iterations=10, tolerance=1e-6))
        current = np.array(oxygen.value, dtype=np.float64)
        if np.abs(current - previous).max() < tolerance:
            return current, iteration
        previous = current
    return current, max_iterations


def test_picard_coupled_oxygen_matches_fipy():
    ours, n_ours = _picard("warpfvm", (12, 12, 12), 600e-6)
    theirs, n_theirs = _picard("fipy", (12, 12, 12), 600e-6)
    assert abs(n_ours - n_theirs) <= 1
    assert np.abs(ours - theirs).max() <= 1e-9 * np.ptp(theirs)
    assert theirs.min() < 0.07 * 0.9  # the uptake is strong enough to matter
