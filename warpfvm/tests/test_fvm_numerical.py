"""Numerical validation: does the scheme solve the equations? (analytic solutions, orders, balances)"""

import numpy as np
import pytest

import warpfvm as wf

TIGHT = dict(tolerance=1e-13)


def _solve(eq, var, dt=None):
    solver = wf.LinearPCGSolver(iterations=100000, **TIGHT)
    eq.solve(var=var, solver=solver, dt=dt)
    assert solver.convergence.converged
    return var.value


def test_linear_profile_is_exact():
    mesh = wf.Grid1D(dx=0.1, nx=10, device="cpu")
    phi = wf.CellVariable(mesh=mesh, value=0.3)
    phi.constrain(0.0, mesh.facesLeft)
    phi.constrain(1.0, mesh.facesRight)
    values = _solve(wf.DiffusionTerm(coeff=2.0) == 0, phi)
    assert np.allclose(values, mesh.x / 1.0, atol=1e-12)


def test_uniform_boundary_gives_uniform_field():
    mesh = wf.Grid3D(dx=1.0, dy=2.0, dz=0.5, nx=5, ny=4, nz=6, device="cpu")
    phi = wf.CellVariable(mesh=mesh, value=0.0)
    phi.constrain(0.07, mesh.exteriorFaces)
    assert np.allclose(_solve(wf.DiffusionTerm(coeff=1e-9) == 0, phi), 0.07, atol=1e-14)


def _decay_error(n):
    L, D, k = 1.0, 1.0, 16.0
    mesh = wf.Grid1D(dx=L / n, nx=n, device="cpu")
    phi = wf.CellVariable(mesh=mesh, value=0.0)
    phi.constrain(1.0, mesh.exteriorFaces)
    values = _solve(wf.DiffusionTerm(coeff=D) - wf.ImplicitSourceTerm(coeff=k) == 0, phi)
    m = np.sqrt(k / D)
    exact = np.cosh(m * (mesh.x - L / 2)) / np.cosh(m * L / 2)
    return np.abs(values - exact).max()


def test_diffusion_decay_converges_at_second_order():
    errors = [_decay_error(n) for n in (20, 40, 80, 160)]
    orders = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
    assert np.all(orders > 1.8), orders
    assert errors[-1] < 1e-3


def _manufactured_error(n):
    L, D = 1.0, 0.5
    mesh = wf.Grid3D(dx=L / n, dy=L / n, dz=L / n, nx=n, ny=n, nz=n, device="cpu")
    x, y, z = mesh.cellCenters
    exact = np.sin(np.pi * x / L) * np.sin(np.pi * y / L) * np.sin(np.pi * z / L)
    source = 3.0 * np.pi**2 * D / L**2 * exact
    phi = wf.CellVariable(mesh=mesh, value=0.0)
    phi.constrain(0.0, mesh.exteriorFaces)
    values = _solve(wf.DiffusionTerm(coeff=D) + source == 0, phi)
    return np.abs(values - exact).max()


def test_manufactured_solution_3d_converges_at_second_order():
    errors = [_manufactured_error(n) for n in (8, 16, 32)]
    orders = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
    assert np.all(orders > 1.8), orders


def test_mirror_symmetry():
    mesh = wf.Grid3D(dx=1.0, dy=1.0, dz=1.0, nx=7, ny=6, nz=5, device="cpu")
    x, y, z = mesh.cellCenters
    source = np.exp(-((x - 3.5) ** 2 + (y - 3.0) ** 2 + (z - 2.5) ** 2))  # symmetric about the centre
    phi = wf.CellVariable(mesh=mesh, value=0.0)
    phi.constrain(1.0, mesh.exteriorFaces)
    field = _solve(wf.DiffusionTerm(coeff=1.0) - wf.ImplicitSourceTerm(coeff=0.1) == -source, phi).reshape(5, 6, 7)
    for axis in range(3):
        assert np.allclose(field, np.flip(field, axis=axis), atol=1e-12)


def test_transient_conserves_mass_with_zero_flux_boundaries(rng):
    mesh = wf.Grid2D(dx=2.0, dy=1.0, nx=9, ny=7, device="cpu")
    phi = wf.CellVariable(mesh=mesh, value=rng.random(mesh.numberOfCells))
    source = rng.normal(size=mesh.numberOfCells)
    eq = wf.TransientTerm() == wf.DiffusionTerm(coeff=3.0) + source
    volume = mesh.cellVolumes
    total = np.sum(phi.value * volume)
    for _ in range(5):
        dt = 0.4
        _solve(eq, phi, dt=dt)
        total += dt * np.sum(source * volume)
        assert np.sum(phi.value * volume) == pytest.approx(total, rel=1e-11)


def test_steady_flux_balance(rng):
    """Everything produced inside leaves through the Dirichlet faces (discrete conservation)."""
    mesh = wf.Grid3D(dx=1.0, dy=1.0, dz=1.0, nx=6, ny=5, nz=4, device="cpu")
    D, boundary = 2.0, 0.5
    source = rng.random(mesh.numberOfCells)
    phi = wf.CellVariable(mesh=mesh, value=0.0)
    phi.constrain(boundary, mesh.exteriorFaces)
    values = _solve(wf.DiffusionTerm(coeff=D) + source == 0, phi).reshape(4, 5, 6)
    outflow = 0.0
    for axis in range(3):
        for index in (0, -1):
            cells = np.take(values, index, axis=axis)
            outflow += np.sum(D * 1.0 * (cells - boundary) / 0.5)  # T = D A / (d/2), A = 1
    assert outflow == pytest.approx(np.sum(source * mesh.cellVolumes), rel=1e-10)


def test_transient_heat_equation_against_the_analytic_mode():
    L, D, n, t_end = 1.0, 1.0, 200, 0.05
    mesh = wf.Grid1D(dx=L / n, nx=n, device="cpu")
    phi = wf.CellVariable(mesh=mesh, value=np.sin(np.pi * mesh.x / L))
    phi.constrain(0.0, mesh.exteriorFaces)
    eq = wf.TransientTerm() == wf.DiffusionTerm(coeff=D)
    steps = 400
    for _ in range(steps):
        _solve(eq, phi, dt=t_end / steps)
    exact = np.exp(-D * np.pi**2 * t_end / L**2) * np.sin(np.pi * mesh.x / L)
    assert np.abs(phi.value - exact).max() < 2e-3 * exact.max()


def test_long_transient_reaches_the_steady_state(rng):
    mesh = wf.Grid2D(dx=1.0, dy=1.0, nx=8, ny=8, device="cpu")
    source = rng.random(mesh.numberOfCells)
    steady = wf.CellVariable(mesh=mesh, value=0.0)
    steady.constrain(0.2, mesh.exteriorFaces)
    _solve(wf.DiffusionTerm(coeff=1.0) - wf.ImplicitSourceTerm(coeff=0.3) + source == 0, steady)
    phi = wf.CellVariable(mesh=mesh, value=0.0)
    phi.constrain(0.2, mesh.exteriorFaces)
    eq = wf.TransientTerm() == wf.DiffusionTerm(coeff=1.0) - wf.ImplicitSourceTerm(coeff=0.3) + source
    for _ in range(60):
        _solve(eq, phi, dt=5.0)
    assert np.allclose(phi.value, steady.value, atol=1e-10)
