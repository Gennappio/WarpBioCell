"""Solvers: FiPy's stopping criteria, warnings, singular systems, precision, warm starts."""

import warnings

import numpy as np
import pytest

import warpfvm as wf
from warpfvm.solvers import SingularSystemWarning


def _oxygen_problem(n=12, dtype="float64", value=0.07):
    """MicroC-shaped: D = 1e-9 m^2/s, 50 um voxels, fixed 0.07 mM boundary, a consuming blob."""
    dx = 50e-6
    mesh = wf.Grid3D(dx=dx, dy=dx, dz=dx, nx=n, ny=n, nz=n, device="cpu", dtype=dtype)
    x, y, z = mesh.cellCenters
    c = n * dx / 2
    blob = ((x - c) ** 2 + (y - c) ** 2 + (z - c) ** 2) < (0.3 * n * dx) ** 2
    consumption = np.where(blob, 3.0e-17 * 8 / dx**3, 0.0)  # 8 cells per voxel, mol/s -> mM/s
    phi = wf.CellVariable(mesh=mesh, value=value)
    phi.constrain(0.07, mesh.exteriorFaces)
    source = wf.CellVariable(mesh=mesh, value=-consumption)
    return mesh, phi, wf.DiffusionTerm(coeff=1.0e-9) == -source


@pytest.mark.parametrize("tolerance", [1e-4, 1e-8])
def test_rhs_criterion(tolerance):
    _, phi, eq = _oxygen_problem()
    solver = wf.LinearPCGSolver(tolerance=tolerance)
    eq.solve(var=phi, solver=solver)
    c = solver.convergence
    assert c.converged and c.residual <= tolerance * c.rhs_norm
    residual = np.linalg.norm(eq.justResidualVector(var=phi))
    assert residual == pytest.approx(c.residual, rel=1e-6)


def test_unscaled_and_initial_criteria():
    _, phi, eq = _oxygen_problem()
    start = np.linalg.norm(eq.justResidualVector(var=phi))
    solver = wf.LinearPCGSolver(tolerance=1e-6, criterion="initial")
    eq.solve(var=phi, solver=solver)
    assert solver.convergence.residual <= 1e-6 * start
    _, phi, eq = _oxygen_problem()
    target = 1e-3 * start
    solver = wf.LinearPCGSolver(tolerance=target, criterion="unscaled")
    eq.solve(var=phi, solver=solver)
    assert solver.convergence.residual <= target
    with pytest.raises(NotImplementedError):
        wf.LinearPCGSolver(criterion="solution")


def test_iteration_limit_warns():
    _, phi, eq = _oxygen_problem(n=16)
    solver = wf.LinearPCGSolver(tolerance=1e-12, iterations=3)
    with pytest.warns(wf.ConvergenceWarning):
        eq.solve(var=phi, solver=solver)
    assert not solver.convergence.converged and solver.convergence.iterations == 3


def test_lu_name_solves_to_machine_precision():
    _, phi, eq = _oxygen_problem()
    solver = wf.LinearLUSolver(iterations=10, tolerance=1e-6)  # MicroC's arguments
    eq.solve(var=phi, solver=solver)
    assert solver.convergence.converged
    assert solver.convergence.relative_residual <= 1e-12


def test_every_name_gives_the_same_solution():
    solutions = []
    for cls in (wf.LinearPCGSolver, wf.LinearBicgstabSolver, wf.LinearGMRESSolver, wf.LinearLUSolver):
        _, phi, eq = _oxygen_problem()
        eq.solve(var=phi, solver=cls(tolerance=1e-12))
        solutions.append(phi.value)
    for other in solutions[1:]:
        assert np.allclose(other, solutions[0], rtol=0.0, atol=1e-13)


def test_no_preconditioner_and_context_manager():
    _, phi, eq = _oxygen_problem()
    with wf.LinearPCGSolver(tolerance=1e-10, precon=None) as solver:
        eq.solve(var=phi, solver=solver)
    reference = phi.value
    _, phi, eq = _oxygen_problem()
    eq.solve(var=phi, solver=wf.LinearPCGSolver(tolerance=1e-10))
    assert np.allclose(phi.value, reference, atol=1e-12)


def test_warm_start_from_the_solution_needs_no_iteration():
    _, phi, eq = _oxygen_problem()
    eq.solve(var=phi, solver=wf.LinearPCGSolver(tolerance=1e-10))
    solver = wf.LinearPCGSolver(tolerance=1e-6)
    eq.solve(var=phi, solver=solver)
    assert solver.convergence.iterations == 0


def test_float32_agrees_with_float64():
    _, phi64, eq64 = _oxygen_problem(dtype="float64")
    eq64.solve(var=phi64, solver=wf.LinearLUSolver())
    _, phi32, eq32 = _oxygen_problem(dtype="float32")
    solver = wf.LinearLUSolver()
    eq32.solve(var=phi32, solver=solver)
    assert solver.convergence.converged
    dip = 0.07 - phi64.value.min()  # the consumption signal the solve must resolve
    assert np.abs(phi32.value - phi64.value).max() < 1e-4 * dip


def test_singular_consistent_problem_keeps_the_mean(rng):
    mesh = wf.Grid2D(dx=1.0, dy=1.0, nx=10, ny=8, device="cpu")
    source = rng.normal(size=mesh.numberOfCells)
    source -= source.mean()  # balanced sources: a steady state exists
    start = rng.random(mesh.numberOfCells)
    phi = wf.CellVariable(mesh=mesh, value=start)
    solver = wf.LinearGMRESSolver(tolerance=1e-10)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no warning for a consistent problem
        (wf.DiffusionTerm(coeff=2.0) == -source).solve(var=phi, solver=solver)
    assert solver.convergence.singular and solver.convergence.converged
    assert phi.value.mean() == pytest.approx(start.mean(), abs=1e-12)
    residual = (wf.DiffusionTerm(coeff=2.0) == -source).justResidualVector(var=phi)
    assert np.abs(residual).max() < 1e-9 * np.abs(source).max()


def test_singular_unbalanced_problem_warns():
    mesh = wf.Grid1D(dx=1.0, nx=20, device="cpu")
    phi = wf.CellVariable(mesh=mesh, value=0.0)
    with pytest.warns(SingularSystemWarning):
        (wf.DiffusionTerm(coeff=1.0) == -1.0).solve(var=phi, solver=wf.LinearGMRESSolver(tolerance=1e-10))


def test_sink_or_boundary_or_transient_is_not_singular():
    mesh = wf.Grid1D(dx=1.0, nx=5, device="cpu")
    phi = wf.CellVariable(mesh=mesh, value=0.0)
    assert (wf.DiffusionTerm() == 0).assemble(var=phi).singular
    assert not (wf.DiffusionTerm() - wf.ImplicitSourceTerm(coeff=0.1) == 0).assemble(var=phi).singular
    assert not (wf.DiffusionTerm() - wf.ImplicitSourceTerm(coeff=np.r_[0, 0, 1.0, 0, 0]) == 0).assemble(var=phi).singular
    assert (wf.DiffusionTerm() + wf.ImplicitSourceTerm(coeff=0.1) == 0).assemble(var=phi).singular  # lagged, not on the diagonal
    assert not (wf.TransientTerm() == wf.DiffusionTerm()).assemble(var=phi, dt=1.0).singular
    phi.constrain(1.0, mesh.facesRight)
    assert not (wf.DiffusionTerm() == 0).assemble(var=phi).singular


# --- parity of sweep and of the residual vector with FiPy -----------------------------------------


def test_sweep_residual_and_residual_vector_match_fipy():
    fp = pytest.importorskip("fipy")
    from fipy.solvers.scipy import LinearLUSolver

    results = []
    for lib, solver in ((fp, LinearLUSolver()), (wf, wf.LinearLUSolver())):
        rng = np.random.default_rng(5)
        mesh = lib.Grid2D(dx=1e-5, dy=2e-5, nx=6, ny=5, **({"device": "cpu"} if lib is wf else {}))
        phi = lib.CellVariable(mesh=mesh, value=rng.random(30))
        phi.constrain(0.2, mesh.facesLeft | mesh.facesBottom)
        sink = lib.CellVariable(mesh=mesh, value=rng.random(30))
        eq = lib.DiffusionTerm(coeff=1e-9) - lib.ImplicitSourceTerm(coeff=sink) == -1e-3
        vector = np.asarray(eq.justResidualVector(var=phi))
        first = eq.sweep(var=phi, solver=solver, underRelaxation=0.8)
        second = eq.sweep(var=phi, solver=solver)
        results.append((vector, first, second, np.asarray(phi.value)))
    (v_fp, a_fp, b_fp, x_fp), (v_wf, a_wf, b_wf, x_wf) = results
    assert np.allclose(v_wf, v_fp, rtol=1e-12, atol=1e-12 * np.abs(v_fp).max())
    assert a_wf == pytest.approx(a_fp, rel=1e-10)
    assert b_wf == pytest.approx(b_fp, rel=1e-6, abs=1e-12 * a_fp)
    assert np.allclose(x_wf, x_fp, rtol=0.0, atol=1e-12)
