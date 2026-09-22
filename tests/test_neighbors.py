"""The hash-grid contact pass must find exactly the neighbours a brute-force search finds."""

import numpy as np
import warp as wp

from warpbiocell.cells.initialization import spherical_cluster
from warpbiocell.cells.mechanics import ContactParams, contact_substep, make_neighbor_grid
from warpbiocell.cells.state import CellPopulation
from warpbiocell.reference.mechanics import contact_velocities_reference


def _run_one_pass(positions, radii, params, device):
    pop = CellPopulation.from_numpy(positions, radii, device=device)
    grid = make_neighbor_grid(params, device)
    # dt = 0 leaves positions untouched, so contact_count reflects the input configuration.
    contact_substep(pop, grid, params, dt=0.0)
    return pop


def test_contact_counts_match_brute_force(cpu):
    r = 8.0
    pos = spherical_cluster(600, radius=r, spacing_factor=0.85, jitter=0.3, seed=11)
    radii = np.full(600, r, dtype=np.float32)
    params = ContactParams(stiffness=1.0, damping=1.0, max_radius=r, margin=0.5)

    pop = _run_one_pass(pos, radii, params, cpu)
    _, expected = contact_velocities_reference(pos, radii, params.rate)

    np.testing.assert_array_equal(pop.contact_counts_numpy(), expected)
    assert expected.max() >= 6  # the configuration is actually crowded


def test_contact_counts_with_mixed_radii(cpu):
    rng = np.random.default_rng(5)
    pos = spherical_cluster(400, radius=8.0, spacing_factor=1.0, jitter=0.2, seed=5)
    radii = rng.uniform(5.0, 10.0, size=400).astype(np.float32)
    params = ContactParams(stiffness=1.0, damping=1.0, max_radius=10.0, margin=0.0)

    pop = _run_one_pass(pos, radii, params, cpu)
    _, expected = contact_velocities_reference(pos, radii, params.rate)

    np.testing.assert_array_equal(pop.contact_counts_numpy(), expected)


def test_isolated_cells_have_no_contacts(cpu):
    pos = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [0.0, 100.0, 0.0]], dtype=np.float32)
    radii = np.full(3, 8.0, dtype=np.float32)
    params = ContactParams(stiffness=1.0, damping=1.0, max_radius=8.0)

    pop = _run_one_pass(pos, radii, params, cpu)

    assert np.all(pop.contact_counts_numpy() == 0)
    assert np.all(pop.velocity.numpy()[:3] == 0.0)


def test_grid_only_indexes_active_prefix(cpu):
    """Free slots (zeros at the origin) must not be reported as neighbours of real cells."""
    pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)  # both at the origin region
    radii = np.full(2, 8.0, dtype=np.float32)
    params = ContactParams(stiffness=1.0, damping=1.0, max_radius=8.0)

    pop = CellPopulation.from_numpy(pos, radii, capacity=50, device=cpu)
    grid = make_neighbor_grid(params, cpu)
    contact_substep(pop, grid, params, dt=0.0)

    # Each real cell sees exactly the other real cell, not the 48 zero-filled free slots.
    np.testing.assert_array_equal(pop.contact_counts_numpy(), [1, 1])
    assert wp.types.types_equal(pop.position.dtype, wp.vec3)
