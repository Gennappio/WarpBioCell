"""Numerical validation of the field solvers: do they solve the equations correctly?"""

import numpy as np
import pytest

from warpbiocell.fields.diffusion import SolverSettings, max_stable_dt
from warpbiocell.fields.oxygen import OxygenField, OxygenParams
from warpbiocell.fields.scalar_field import Boundary, GridGeometry

DIRICHLET = (Boundary.DIRICHLET,) * 3


def _field(geom, params, device, omega=1.5, tol=1e-6, boundary=DIRICHLET, max_sweeps=20000):
    return OxygenField(geom, params, SolverSettings(omega=omega, tolerance=tol, max_sweeps=max_sweeps, check_every=10), boundary=boundary, device=device)


def test_uniform_field_stays_uniform_without_consumption(cpu):
    geom = GridGeometry.centered_cube(200.0, 20.0)
    ox = _field(geom, OxygenParams(boundary_value=40.0), cpu)
    ox.density.zero_()

    ox.solve_steady_state()
    np.testing.assert_array_equal(ox.numpy(), 40.0)

    for _ in range(20):
        ox.explicit_step(0.5 * max_stable_dt(ox.field, ox.params.kinetics))
    np.testing.assert_array_equal(ox.numpy(), 40.0)


def test_zero_consumption_converges_to_boundary_value_from_zero(cpu):
    geom = GridGeometry.centered_cube(300.0, 20.0)
    ox = _field(geom, OxygenParams(boundary_value=38.0), cpu, omega=1.8)
    ox.density.zero_()
    ox.field.values.fill_(0.0)
    ox.field.apply_dirichlet()

    report = ox.solve_steady_state()

    assert report.converged
    assert np.abs(ox.numpy() - 38.0).max() < 1e-4


@pytest.mark.parametrize("omega", [1.0, 1.5, 1.9])
def test_one_dimensional_linear_uptake_matches_cosh_profile(cpu, omega):
    """Dirichlet in x, zero-flux in y and z, linear uptake: O(x) = O_b cosh(kx) / cosh(kL)."""
    L, nx = 200.0, 81
    geom = GridGeometry(origin=(-L, 0.0, 0.0), dx=2 * L / (nx - 1), shape=(nx, 3, 3))
    D, kappa, K = 1.0e4, 3.0 / L, 1.0e9
    params = OxygenParams(diffusion_coefficient=D, uptake_max=kappa**2 * D * K, michaelis_k=K, boundary_value=1.0)
    ox = _field(geom, params, cpu, omega=omega, tol=1e-5, boundary=(Boundary.DIRICHLET, Boundary.NEUMANN, Boundary.NEUMANN))
    ox.density.fill_(1.0)

    report = ox.solve_steady_state()

    x = geom.node_coordinates()[0]
    exact = np.cosh(kappa * x) / np.cosh(kappa * L)
    values = ox.numpy()
    assert report.converged
    # Discretisation error ~ (kappa dx)^2 / 12 ~ 5e-4 plus the solver error at residual 1e-5;
    # a wrong stencil or boundary treatment would be off by O(0.1).
    np.testing.assert_allclose(values[:, 1, 1], exact, atol=2e-3)
    np.testing.assert_allclose(values[:, 0, 0], values[:, 1, 1], atol=1e-6)  # no y/z variation


def test_gaussian_pulse_diffuses_analytically(cpu):
    """Explicit reference scheme against the free-space Gaussian solution (boundary far away)."""
    n, dx, D = 41, 1.0, 1.0
    geom = GridGeometry(origin=(-20.0, -20.0, -20.0), dx=dx, shape=(n, n, n))
    params = OxygenParams(diffusion_coefficient=D, uptake_max=0.0, michaelis_k=1.0, boundary_value=0.0)
    ox = _field(geom, params, cpu)
    ox.density.zero_()
    x, y, z = np.meshgrid(*geom.node_coordinates(), indexing="ij")
    r2 = x**2 + y**2 + z**2
    sigma0 = 3.0
    ox.field.assign(np.exp(-r2 / (2 * sigma0**2)))

    dt = 0.5 * max_stable_dt(ox.field, params.kinetics)
    steps = 48
    for _ in range(steps):
        ox.explicit_step(dt)

    t = steps * dt
    var = sigma0**2 + 2 * D * t
    exact = (sigma0**2 / var) ** 1.5 * np.exp(-r2 / (2 * var))
    values = ox.numpy()
    assert np.abs(values - exact).max() < 0.01 * exact.max()
    # The discrete second moment grows exactly by 2 D dt per step (up to boundary leakage).
    mass = values.sum()
    measured_var = (values * x**2).sum() / mass
    assert measured_var == pytest.approx(var, rel=2e-3)


def test_symmetric_consumption_gives_symmetric_field(cpu):
    geom = GridGeometry.centered_cube(400.0, 20.0)
    ox = _field(geom, OxygenParams(), cpu, omega=1.8, tol=1e-5)
    x, y, z = np.meshgrid(*geom.node_coordinates(), indexing="ij")
    density = np.where(x**2 + y**2 + z**2 < 120.0**2, 2.4e-4, 0.0).astype(np.float32)
    ox.density.assign(density)

    report = ox.solve_steady_state()

    v = ox.numpy()
    assert report.converged
    for axis in range(3):
        np.testing.assert_allclose(v, np.flip(v, axis=axis), atol=1e-4)
    np.testing.assert_allclose(v, np.transpose(v, (1, 2, 0)), atol=1e-4)
    centre = v[geom.shape[0] // 2, geom.shape[1] // 2, geom.shape[2] // 2]
    assert centre < 0.75 * ox.params.boundary_value  # consumption actually depletes the core


def test_strong_uptake_keeps_field_non_negative(cpu):
    geom = GridGeometry.centered_cube(300.0, 20.0)
    ox = _field(geom, OxygenParams(uptake_max=1.0e12), cpu, omega=1.9, tol=1e-5)
    ox.density.fill_(1.0e-3)

    report = ox.solve_steady_state()

    v = ox.numpy()
    assert report.converged
    assert v.min() >= 0.0
    assert v[ox.field.interior_mask()].max() < 1.0  # essentially fully depleted inside


def test_warm_start_needs_few_sweeps(cpu):
    geom = GridGeometry.centered_cube(400.0, 20.0)
    ox = _field(geom, OxygenParams(), cpu, omega=1.8, tol=1e-5)
    x, y, z = np.meshgrid(*geom.node_coordinates(), indexing="ij")
    ox.density.assign(np.where(x**2 + y**2 + z**2 < 150.0**2, 2.4e-4, 0.0).astype(np.float32))

    first = ox.solve_steady_state()
    second = ox.solve_steady_state()

    assert first.converged and second.converged
    assert first.sweeps > 20
    assert second.sweeps == ox.settings.check_every


def test_explicit_scheme_rejects_unstable_dt(cpu):
    geom = GridGeometry.centered_cube(100.0, 20.0)
    ox = _field(geom, OxygenParams(), cpu)
    with pytest.raises(ValueError):
        ox.explicit_step(1.5 * max_stable_dt(ox.field, ox.params.kinetics))


def test_geometry_helpers():
    geom = GridGeometry.centered_cube(400.0, 20.0)
    assert geom.shape == (21, 21, 21)
    assert geom.origin == (-200.0, -200.0, -200.0)
    assert geom.upper == (200.0, 200.0, 200.0)
    inside = geom.contains(np.array([[0.0, 0.0, 0.0], [199.0, -199.0, 0.0], [201.0, 0.0, 0.0]]))
    np.testing.assert_array_equal(inside, [True, True, False])
    with pytest.raises(ValueError):
        GridGeometry(origin=(0, 0, 0), dx=0.0, shape=(4, 4, 4))
    with pytest.raises(ValueError):
        SolverSettings(omega=2.0)
