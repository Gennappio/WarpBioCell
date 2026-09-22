"""Computational validation: Warp field kernels against numpy references on tiny problems."""

import numpy as np

from warpbiocell.cells.initialization import spherical_cluster
from warpbiocell.cells.model import CellState
from warpbiocell.cells.state import CellPopulation
from warpbiocell.fields.diffusion import SolverSettings, max_stable_dt
from warpbiocell.fields.oxygen import OxygenField, OxygenParams
from warpbiocell.fields.scalar_field import Boundary, GridGeometry
from warpbiocell.reference.fields import (
    deposit_reference,
    ftcs_step_reference,
    sample_reference,
    steady_state_reference,
)

MIXED = (Boundary.DIRICHLET, Boundary.NEUMANN, Boundary.DIRICHLET)


def test_deposit_matches_reference_and_conserves_cells(cpu):
    n = 300
    rng = np.random.default_rng(4)
    pos = spherical_cluster(n, 8.0, spacing_factor=1.0, jitter=0.3, seed=4)
    states = np.full(n, int(CellState.PROLIFERATIVE), dtype=np.int32)
    states[rng.choice(n, 40, replace=False)] = int(CellState.DEAD)
    pop = CellPopulation.from_numpy(pos, np.full(n, 8.0), device=cpu, states=states)
    geom = GridGeometry.centered_cube(300.0, 20.0)
    ox = OxygenField(geom, OxygenParams(), device=cpu)

    ox.deposit_uptake(pop)

    expected = deposit_reference(pos, states != int(CellState.DEAD), geom.origin, geom.dx, geom.shape)
    density = ox.density.numpy()
    np.testing.assert_allclose(density, expected, atol=1e-8)  # fixed-point quantisation is 2^-24
    assert abs(density.sum() * geom.voxel_volume - (n - 40)) < 1e-3


def test_deposit_ignores_cells_outside_the_grid(cpu):
    pos = np.array([[0.0, 0.0, 0.0], [500.0, 0.0, 0.0]], dtype=np.float32)
    pop = CellPopulation.from_numpy(pos, np.full(2, 8.0), device=cpu)
    geom = GridGeometry.centered_cube(200.0, 20.0)
    ox = OxygenField(geom, OxygenParams(), device=cpu)

    ox.deposit_uptake(pop)

    assert abs(ox.density.numpy().sum() * geom.voxel_volume - 1.0) < 1e-6


def test_sampling_is_exact_for_linear_fields(cpu):
    geom = GridGeometry.centered_cube(200.0, 25.0)
    # Neumann on every face so that assign() keeps the linear values on the face nodes too.
    ox = OxygenField(geom, OxygenParams(), boundary=(Boundary.NEUMANN,) * 3, device=cpu)
    x, y, z = np.meshgrid(*geom.node_coordinates(), indexing="ij")
    ox.field.assign(3.0 + 0.1 * x - 0.05 * y + 0.02 * z)
    rng = np.random.default_rng(1)
    pos = rng.uniform(-90.0, 90.0, size=(200, 3)).astype(np.float32)
    pop = CellPopulation.from_numpy(pos, np.full(200, 8.0), device=cpu)

    ox.sample_at_cells(pop)

    expected = 3.0 + 0.1 * pos[:, 0] - 0.05 * pos[:, 1] + 0.02 * pos[:, 2]
    np.testing.assert_allclose(pop.oxygen_numpy(), expected, atol=1e-4)
    np.testing.assert_allclose(pop.oxygen_numpy(), sample_reference(ox.numpy(), pos, geom.origin, geom.dx), atol=1e-4)


def test_sampling_clamps_positions_outside_the_grid(cpu):
    geom = GridGeometry.centered_cube(100.0, 20.0)
    ox = OxygenField(geom, OxygenParams(boundary_value=30.0), device=cpu)
    pos = np.array([[1000.0, 0.0, 0.0], [0.0, -1000.0, 0.0]], dtype=np.float32)
    pop = CellPopulation.from_numpy(pos, np.full(2, 8.0), device=cpu)

    ox.sample_at_cells(pop)

    np.testing.assert_allclose(pop.oxygen_numpy(), 30.0)


def test_explicit_step_matches_reference(cpu):
    geom = GridGeometry(origin=(0.0, 0.0, 0.0), dx=10.0, shape=(7, 6, 5))
    params = OxygenParams(diffusion_coefficient=2.0e3, uptake_max=5.0e4, michaelis_k=3.0, boundary_value=20.0)
    ox = OxygenField(geom, params, boundary=MIXED, device=cpu)
    rng = np.random.default_rng(2)
    values = rng.uniform(5.0, 20.0, size=geom.shape)
    density = rng.uniform(0.0, 1.0e-3, size=geom.shape)
    ox.field.assign(values)
    ox.density.assign(density.astype(np.float32))
    dt = 0.9 * max_stable_dt(ox.field, params.kinetics)

    ref = ox.numpy().astype(np.float64)
    for _ in range(5):
        ox.explicit_step(dt)
        ref = ftcs_step_reference(ref, ox.density.numpy(), params.diffusion_coefficient, params.uptake_max, params.michaelis_k, geom.dx, dt, MIXED)
        np.testing.assert_allclose(ox.numpy(), ref, rtol=1e-5, atol=1e-4)


def test_steady_state_matches_exact_discrete_solution(cpu):
    geom = GridGeometry(origin=(0.0, 0.0, 0.0), dx=10.0, shape=(8, 7, 6))
    params = OxygenParams(diffusion_coefficient=2.0e3, uptake_max=5.0e4, michaelis_k=3.0, boundary_value=20.0)
    # Residual tolerance 1e-5 sits above the float32 floor (~2e-6 at omega = 1.5, see diffusion.py).
    ox = OxygenField(geom, params, SolverSettings(omega=1.5, tolerance=1e-5, max_sweeps=20000, check_every=10), boundary=MIXED, device=cpu)
    rng = np.random.default_rng(3)
    density = rng.uniform(0.0, 2.0e-3, size=geom.shape).astype(np.float32)
    ox.density.assign(density)

    report = ox.solve_steady_state()

    exact = steady_state_reference(np.full(geom.shape, 20.0), density, params.diffusion_coefficient, params.uptake_max, params.michaelis_k, geom.dx, MIXED)
    assert report.converged
    np.testing.assert_allclose(ox.numpy(), exact, atol=2e-3)
    assert exact.min() > 0.0 and exact[ox.field.interior_mask()].max() < 20.0  # the problem is non-trivial


def test_explicit_time_stepping_relaxes_to_the_steady_state(cpu):
    geom = GridGeometry(origin=(0.0, 0.0, 0.0), dx=10.0, shape=(7, 7, 7))
    params = OxygenParams(diffusion_coefficient=2.0e3, uptake_max=5.0e4, michaelis_k=3.0, boundary_value=20.0)
    ox = OxygenField(geom, params, SolverSettings(omega=1.5, tolerance=1e-5, max_sweeps=20000, check_every=10), device=cpu)
    density = np.zeros(geom.shape, dtype=np.float32)
    density[2:5, 2:5, 2:5] = 1.5e-3
    ox.density.assign(density)
    ox.solve_steady_state()
    steady = ox.numpy()

    ox.field.values.fill_(params.boundary_value)
    dt = 0.9 * max_stable_dt(ox.field, params.kinetics)
    for _ in range(3000):
        ox.explicit_step(dt)

    np.testing.assert_allclose(ox.numpy(), steady, atol=1e-3)
