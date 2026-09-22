"""Biological validation (qualitative): does the coupled model behave like a spheroid?

These tests check emergence, not numbers: an oxygen gradient from rim to core, hypoxia and
death that appear only above a critical size, and a viable rim that survives. Parameters are
the illustrative defaults; quantitative comparison with data is Milestone 5 work.
"""

import numpy as np

from warpbiocell.cells.initialization import spherical_cluster
from warpbiocell.cells.lifecycle import LifecycleParams
from warpbiocell.cells.mechanics import ContactParams, make_neighbor_grid
from warpbiocell.cells.model import CellState
from warpbiocell.cells.state import CellPopulation
from warpbiocell.fields.diffusion import SolverSettings
from warpbiocell.fields.oxygen import OxygenField, OxygenParams
from warpbiocell.fields.scalar_field import GridGeometry
from warpbiocell.metrics.population import state_counts
from warpbiocell.simulation.simulator import TimeStepping, cell_step

R = 8.0


def _static_spheroid(n, device):
    pos = spherical_cluster(n, R, spacing_factor=1.0, jitter=0.1, seed=0)
    pop = CellPopulation.from_numpy(pos, np.full(n, R), device=device)
    geom = GridGeometry.centered_cube(800.0, 20.0)
    ox = OxygenField(geom, OxygenParams(), SolverSettings(omega=1.8, tolerance=1e-5, check_every=10), device=device)
    ox.update(pop)
    return pop, ox


def test_large_spheroid_has_radial_gradient_and_hypoxic_core(cpu):
    pop, ox = _static_spheroid(12_000, cpu)
    x = pop.positions_numpy()
    r = np.linalg.norm(x, axis=1)
    o = pop.oxygen_numpy()

    core, rim = o[r < 40.0].mean(), o[r > r.max() - 30.0].mean()
    assert rim > 5.0 * core
    assert np.mean(o < 8.0) > 0.3  # substantial hypoxic fraction
    # Monotone radial profile: binned means never increase inward.
    bins = np.linspace(0.0, r.max(), 10)
    profile = np.array([o[(r >= lo) & (r < hi)].mean() for lo, hi in zip(bins[:-1], bins[1:])])
    assert np.all(np.diff(profile) >= -1e-3)


def test_small_spheroid_is_fully_oxygenated(cpu):
    pop, ox = _static_spheroid(800, cpu)
    o = pop.oxygen_numpy()
    assert o.min() > 8.0
    assert o.min() < o.max()  # a gradient still exists


def test_growing_spheroid_develops_hypoxia_and_keeps_a_viable_rim(cpu):
    n0 = 3000
    pos = spherical_cluster(n0, R, spacing_factor=1.0, jitter=0.1, seed=1)
    pop = CellPopulation.from_numpy(pos, np.full(n0, R), capacity=40_000, device=cpu, seed=1)
    contact = ContactParams(stiffness=10.0, damping=1.0, max_radius=R, margin=2.0)
    grid = make_neighbor_grid(contact, cpu)
    lifecycle = LifecycleParams.oxygen_dependent(division_rate=0.08, inhibition_threshold=8)
    stepping = TimeStepping(dt_cells=0.5, dt_mechanics=0.05)
    ox = OxygenField(GridGeometry.centered_cube(900.0, 20.0), OxygenParams(), SolverSettings(omega=1.8, tolerance=1e-5, check_every=10), device=cpu)

    for _ in range(40):  # 20 h
        report = cell_step(pop, grid, lifecycle, contact, stepping, oxygen=ox)
        assert report.field_converged

    counts = state_counts(pop)
    x = pop.positions_numpy()
    r = np.linalg.norm(x - x.mean(axis=0), axis=1)
    states = pop.states_numpy()
    outer = r > np.percentile(r, 90)
    inner = r < np.percentile(r, 20)
    assert pop.count > n0
    assert counts[CellState.HYPOXIC] > 0 and counts[CellState.DEAD] > 0
    assert np.mean(states[outer] == int(CellState.DEAD)) < 0.05  # viable rim
    assert np.mean(states[inner] != int(CellState.PROLIFERATIVE)) > 0.9  # non-proliferating core
    assert np.mean(states[outer] == int(CellState.PROLIFERATIVE)) > np.mean(states[inner] == int(CellState.PROLIFERATIVE))
