"""Lifecycle rules: division, death, contact inhibition, capacity, ages, exponential growth."""

import numpy as np
import pytest

from warpbiocell.cells.initialization import spherical_cluster
from warpbiocell.cells.lifecycle import CapacityError, LifecycleParams, lifecycle_step
from warpbiocell.cells.mechanics import ContactParams, contact_substep, make_neighbor_grid
from warpbiocell.cells.model import CellState, CellType
from warpbiocell.cells.state import CellPopulation
from warpbiocell.metrics.population import state_counts

R = 8.0
CERTAIN = 1.0e3  # rate * dt = 1000 -> P = 1 - exp(-1000) rounds to 1.0 in float32
NEVER = 0.0
NO_INHIBITION = 10**6


def _population(n, device, capacity=None, seed=0, states=None, spacing=1.0):
    pos = spherical_cluster(n, R, spacing_factor=spacing, jitter=0.1, seed=seed)
    return CellPopulation.from_numpy(pos, np.full(n, R), capacity=capacity, device=device, seed=seed, states=states)


def test_every_proliferative_cell_divides_exactly_once(cpu):
    n = 300
    pop = _population(n, cpu, capacity=3 * n)
    before = pop.positions_numpy()
    params = LifecycleParams(division_rate=CERTAIN, death_rate=NEVER, inhibition_threshold=NO_INHIBITION, placement_factor=1.0)

    created = lifecycle_step(pop, params, dt=1.0)

    assert created == n
    assert pop.count == 2 * n
    x = pop.positions_numpy()
    parents, daughters = x[:n], x[n:]
    # Slot order: the k-th dividing cell fills slot n + k, so daughter n + i belongs to parent i.
    np.testing.assert_allclose(np.linalg.norm(daughters - parents, axis=1), params.placement_factor * R, rtol=1e-5)
    np.testing.assert_allclose(0.5 * (daughters + parents), before, atol=1e-5)  # symmetric split
    states = pop.states_numpy()
    assert np.all(states == int(CellState.PROLIFERATIVE))
    np.testing.assert_array_equal(pop.radii_numpy(), R)
    np.testing.assert_array_equal(pop.cell_type.numpy()[: pop.count], int(CellType.TUMOR))
    ages = pop.ages_numpy()
    assert np.all(ages[:n] == 1.0) and np.all(ages[n:] == 0.0)
    # Free slots stay untouched.
    assert np.all(pop.cell_state.numpy()[2 * n :] == int(CellState.DEAD))
    assert np.all(pop.position.numpy()[2 * n :] == 0.0)


def test_dead_cells_neither_divide_nor_move_nor_age(cpu):
    n = 200
    states = np.full(n, int(CellState.PROLIFERATIVE), dtype=np.int32)
    dead = np.arange(0, n, 3)
    states[dead] = int(CellState.DEAD)
    pop = _population(n, cpu, capacity=3 * n, states=states)
    before = pop.positions_numpy()
    params = LifecycleParams(division_rate=CERTAIN, death_rate=NEVER, inhibition_threshold=NO_INHIBITION)

    created = lifecycle_step(pop, params, dt=1.0)

    assert created == n - dead.size
    assert pop.count == 2 * n - dead.size
    after = pop.positions_numpy()
    np.testing.assert_array_equal(after[dead], before[dead])
    assert np.all(pop.states_numpy()[dead] == int(CellState.DEAD))
    assert np.all(pop.ages_numpy()[dead] == 0.0)
    assert np.all(pop.states_numpy()[n:] == int(CellState.PROLIFERATIVE))


def test_certain_death_kills_everyone_and_blocks_division(cpu):
    n = 100
    pop = _population(n, cpu, capacity=3 * n)
    params = LifecycleParams(division_rate=CERTAIN, death_rate=CERTAIN, inhibition_threshold=NO_INHIBITION)

    created = lifecycle_step(pop, params, dt=1.0)

    assert created == 0
    assert pop.count == n
    assert state_counts(pop)[CellState.DEAD] == n


def test_capacity_overflow_raises_and_places_nothing(cpu):
    n = 50
    pop = _population(n, cpu, capacity=n + 10)
    before = pop.positions_numpy()
    params = LifecycleParams(division_rate=CERTAIN, death_rate=NEVER, inhibition_threshold=NO_INHIBITION)

    with pytest.raises(CapacityError):
        lifecycle_step(pop, params, dt=1.0)

    assert pop.count == n
    np.testing.assert_array_equal(pop.positions_numpy(), before)
    assert np.all(pop.position.numpy()[n:] == 0.0)
    assert np.all(pop.cell_state.numpy()[n:] == int(CellState.DEAD))


def test_contact_inhibition_makes_crowded_cells_quiescent(cpu):
    # A compact cluster plus a few isolated cells far away.
    cluster = spherical_cluster(400, R, spacing_factor=1.0, jitter=0.05, seed=3)
    isolated = np.array([[500.0, 0.0, 0.0], [0.0, 500.0, 0.0], [0.0, 0.0, 500.0]], dtype=np.float32)
    pos = np.vstack([cluster, isolated])
    n = pos.shape[0]
    pop = CellPopulation.from_numpy(pos, np.full(n, R), capacity=3 * n, device=cpu)
    contact = ContactParams(stiffness=10.0, damping=1.0, max_radius=R, margin=2.0)
    grid = make_neighbor_grid(contact, cpu)
    contact_substep(pop, grid, contact, dt=0.0)  # neighbour counts of the initial configuration
    neighbors = pop.neighbor_counts_numpy()
    threshold = 6  # interior cubic-lattice cells have exactly 6 neighbours within 18 um
    crowded = neighbors >= threshold
    assert crowded.sum() > 100 and (~crowded).sum() >= 3

    params = LifecycleParams(division_rate=CERTAIN, death_rate=NEVER, inhibition_threshold=threshold)
    created = lifecycle_step(pop, params, dt=1.0)

    states = pop.states_numpy()[:n]
    assert np.all(states[crowded] == int(CellState.QUIESCENT))
    assert np.all(states[~crowded] == int(CellState.PROLIFERATIVE))
    assert created == int((~crowded).sum())


def test_growth_without_inhibition_follows_the_discrete_branching_law(cpu):
    """Each cell divides at most once per step with p = 1 - exp(-lambda dt), so E[N_k] = N0 (1+p)^k.

    The relative spread of N_k / E[N_k] is ~ 1/sqrt(N0) ~ 2% at N0 = 2000; tolerances are ~3 sigma.
    (1+p)^k differs from exp(lambda dt k) at second order in lambda dt: 17% here at lambda dt = 0.1,
    0.5% at the default dt_cells = 0.25 h and lambda = 0.0289/h.
    """
    n0 = 2000
    rate, dt, steps = 0.1, 1.0, 20
    pop = _population(n0, cpu, capacity=25_000)
    params = LifecycleParams(division_rate=rate, death_rate=NEVER, inhibition_threshold=NO_INHIBITION)

    counts = [pop.count]
    for _ in range(steps):
        lifecycle_step(pop, params, dt=dt)
        counts.append(pop.count)

    p = 1.0 - np.exp(-rate * dt)
    expected = n0 * (1.0 + p) ** np.arange(steps + 1)
    ratio = np.array(counts) / expected
    assert np.all(np.abs(ratio - 1.0) < 0.1), ratio
    assert abs(ratio[-1] - 1.0) < 0.07
    assert counts[-1] < n0 * np.exp(rate * dt * steps) * 0.95  # and it is not the continuous law


def test_death_rate_is_statistically_right(cpu):
    n = 20_000
    pop = _population(n, cpu)
    rate, dt = 0.05, 1.0
    params = LifecycleParams(division_rate=NEVER, death_rate=rate, inhibition_threshold=NO_INHIBITION)

    lifecycle_step(pop, params, dt=dt)

    p = 1.0 - np.exp(-rate * dt)
    dead = state_counts(pop)[CellState.DEAD]
    assert abs(dead / n - p) < 4.0 * np.sqrt(p * (1 - p) / n)


def test_lifecycle_params_validation():
    with pytest.raises(ValueError):
        LifecycleParams(division_rate=-1.0)
    with pytest.raises(ValueError):
        LifecycleParams(inhibition_threshold=0)
    with pytest.raises(ValueError):
        LifecycleParams(placement_factor=0.0)
