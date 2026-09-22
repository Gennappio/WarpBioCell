"""Full cell steps (lifecycle + mechanics): reproducibility, capacity independence, metrics."""

import numpy as np
import pytest

from warpbiocell.cells.initialization import spherical_cluster
from warpbiocell.cells.lifecycle import LifecycleParams
from warpbiocell.cells.mechanics import ContactParams, make_neighbor_grid
from warpbiocell.cells.model import CellState
from warpbiocell.cells.state import CellPopulation
from warpbiocell.metrics.population import population_summary, state_counts
from warpbiocell.simulation.simulator import TimeStepping, cell_step

R = 8.0


def _run(device, seed, capacity, steps=12, n0=400):
    pos = spherical_cluster(n0, R, spacing_factor=1.0, jitter=0.1, seed=seed)
    pop = CellPopulation.from_numpy(pos, np.full(n0, R), capacity=capacity, device=device, seed=seed)
    contact = ContactParams(stiffness=10.0, damping=1.0, max_radius=R, margin=2.0)
    grid = make_neighbor_grid(contact, device)
    lifecycle = LifecycleParams(division_rate=0.2, death_rate=0.01, inhibition_threshold=8)
    stepping = TimeStepping(dt_cells=0.25, dt_mechanics=0.025)
    for _ in range(steps):
        cell_step(pop, grid, lifecycle, contact, stepping)
    return pop


def _snapshot(pop):
    return pop.count, pop.positions_numpy(), pop.states_numpy(), pop.ages_numpy()


def test_full_step_is_bitwise_reproducible(cpu):
    a = _snapshot(_run(cpu, seed=5, capacity=4000))
    b = _snapshot(_run(cpu, seed=5, capacity=4000))
    assert a[0] == b[0] and a[0] > 400
    for x, y in zip(a[1:], b[1:]):
        np.testing.assert_array_equal(x, y)


def test_result_does_not_depend_on_capacity(cpu):
    """RNG streams are keyed on the slot, so a larger allocation changes nothing."""
    a = _snapshot(_run(cpu, seed=5, capacity=4000))
    b = _snapshot(_run(cpu, seed=5, capacity=9000))
    assert a[0] == b[0]
    for x, y in zip(a[1:], b[1:]):
        np.testing.assert_array_equal(x, y)


def test_different_seed_gives_different_run(cpu):
    a = _run(cpu, seed=5, capacity=4000)
    b = _run(cpu, seed=6, capacity=4000)
    assert a.count != b.count or not np.array_equal(a.positions_numpy()[:400], b.positions_numpy()[:400])


def test_daughters_get_separated_by_mechanics(cpu):
    """After a step, no pair should still overlap by anything close to the placement offset."""
    pop = _run(cpu, seed=1, capacity=4000, steps=4)
    x = pop.positions_numpy()
    d = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    assert (2.0 * R - d).max() < 0.35 * R


def test_metrics_are_consistent(cpu):
    pop = _run(cpu, seed=2, capacity=4000)
    counts = state_counts(pop)
    summary = population_summary(pop)
    assert sum(counts.values()) == pop.count == summary["cells"]
    assert summary["living_cells"] + summary["dead_cells"] == pop.count
    assert counts[CellState.HYPOXIC] == 0  # no oxygen rule yet
    assert summary["dead_cells"] > 0 and summary["proliferative_cells"] > 0
    assert summary["spheroid_radius"] > 50.0


def test_time_stepping_validation():
    assert TimeStepping(dt_cells=0.25, dt_mechanics=0.025).mechanics_substeps == 10
    with pytest.raises(ValueError):
        TimeStepping(dt_cells=0.1, dt_mechanics=0.2)
    with pytest.raises(ValueError):
        TimeStepping(dt_cells=0.0)
