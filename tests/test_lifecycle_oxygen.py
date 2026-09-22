"""Oxygen-dependent lifecycle rules with prescribed per-cell oxygen values."""

import numpy as np

from warpbiocell.cells.initialization import spherical_cluster
from warpbiocell.cells.lifecycle import LifecycleParams, lifecycle_step
from warpbiocell.cells.model import CellState
from warpbiocell.cells.state import CellPopulation
from warpbiocell.metrics.population import state_counts

R = 8.0
CERTAIN = 1.0e3
NO_INHIBITION = 10**6


def _population(n, device, oxygen, capacity=None):
    pos = spherical_cluster(n, R, spacing_factor=1.0, jitter=0.1, seed=0)
    pop = CellPopulation.from_numpy(pos, np.full(n, R), capacity=capacity or 3 * n, device=device)
    buf = pop.oxygen_local.numpy()
    buf[:n] = oxygen
    pop.oxygen_local.assign(buf)
    return pop


def test_states_follow_oxygen_thresholds(cpu):
    n = 300
    oxygen = np.linspace(0.0, 40.0, n).astype(np.float32)
    pop = _population(n, cpu, oxygen)
    params = LifecycleParams(division_rate=0.0, hypoxia_threshold=8.0, death_threshold=2.0, inhibition_threshold=NO_INHIBITION)

    lifecycle_step(pop, params, dt=1.0)

    states = pop.states_numpy()
    np.testing.assert_array_equal(states[oxygen < 8.0], int(CellState.HYPOXIC))
    np.testing.assert_array_equal(states[oxygen >= 8.0], int(CellState.PROLIFERATIVE))


def test_hypoxia_takes_precedence_over_crowding(cpu):
    n = 100
    pop = _population(n, cpu, np.full(n, 1.0, dtype=np.float32))
    buf = pop.neighbor_count.numpy()
    buf[:n] = 50
    pop.neighbor_count.assign(buf)
    params = LifecycleParams(division_rate=CERTAIN, hypoxia_threshold=8.0, death_threshold=0.5, inhibition_threshold=8)

    created = lifecycle_step(pop, params, dt=1.0)

    assert created == 0  # crowded cells never divide
    assert state_counts(pop)[CellState.HYPOXIC] == n


def test_division_stops_below_death_threshold_and_ramps_in_between(cpu):
    n = 30_000
    oxygen = np.concatenate([np.full(n // 3, 1.0), np.full(n // 3, 5.0), np.full(n // 3, 20.0)]).astype(np.float32)
    pop = _population(n, cpu, oxygen, capacity=2 * n + 10)
    rate = 0.2
    params = LifecycleParams(division_rate=rate, hypoxia_threshold=8.0, death_threshold=2.0, inhibition_threshold=NO_INHIBITION)

    lifecycle_step(pop, params, dt=1.0)

    flags = pop.divide_flag.numpy()[:n]
    thirds = np.split(flags, 3)
    assert thirds[0].sum() == 0  # below death threshold: no division
    p_full = 1.0 - np.exp(-rate)
    p_half = 1.0 - np.exp(-rate * (5.0 - 2.0) / (8.0 - 2.0))
    for observed, p in ((thirds[1].mean(), p_half), (thirds[2].mean(), p_full)):
        assert abs(observed - p) < 4.0 * np.sqrt(p * (1 - p) / (n // 3))


def test_anoxic_death_rate_applies_only_below_threshold(cpu):
    n = 40_000
    oxygen = np.concatenate([np.full(n // 2, 1.0), np.full(n // 2, 30.0)]).astype(np.float32)
    pop = _population(n, cpu, oxygen)
    params = LifecycleParams(division_rate=0.0, death_rate=0.0, anoxic_death_rate=0.5, hypoxia_threshold=8.0, death_threshold=2.0, inhibition_threshold=NO_INHIBITION)

    lifecycle_step(pop, params, dt=1.0)

    states = pop.states_numpy()
    p = 1.0 - np.exp(-0.5)
    dead_low = np.mean(states[: n // 2] == int(CellState.DEAD))
    assert abs(dead_low - p) < 4.0 * np.sqrt(p * (1 - p) / (n // 2))
    assert np.all(states[n // 2 :] != int(CellState.DEAD))


def test_daughter_inherits_oxygen_and_state(cpu):
    n = 50
    pop = _population(n, cpu, np.full(n, 5.0, dtype=np.float32))
    params = LifecycleParams(division_rate=CERTAIN, hypoxia_threshold=8.0, death_threshold=2.0, inhibition_threshold=NO_INHIBITION)

    created = lifecycle_step(pop, params, dt=1.0)

    assert created == n
    assert np.all(pop.states_numpy() == int(CellState.HYPOXIC))
    np.testing.assert_array_equal(pop.oxygen_numpy(), 5.0)


def test_oxygen_independent_defaults_are_unchanged(cpu):
    """Thresholds at 0 with no field attached reproduce the Milestone 3 rules exactly."""
    n = 200
    pop = _population(n, cpu, np.zeros(n, dtype=np.float32))
    params = LifecycleParams(division_rate=CERTAIN, inhibition_threshold=NO_INHIBITION)

    created = lifecycle_step(pop, params, dt=1.0)

    assert created == n
    assert np.all(pop.states_numpy() == int(CellState.PROLIFERATIVE))
