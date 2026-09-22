"""Numerical and computational validation of the overdamped contact mechanics."""

import numpy as np
import pytest

from warpbiocell.cells.initialization import spherical_cluster
from warpbiocell.cells.mechanics import ContactParams, contact_substep, make_neighbor_grid, relax_contacts
from warpbiocell.cells.state import CellPopulation
from warpbiocell.reference.mechanics import contact_substep_reference


def _make(positions, radii, device, capacity=None):
    return CellPopulation.from_numpy(positions, radii, capacity=capacity, device=device)


def test_single_cell_stays_put(cpu):
    pos = np.array([[3.0, -2.0, 5.0]], dtype=np.float32)
    pop = _make(pos, np.array([8.0]), cpu)
    params = ContactParams(stiffness=10.0, damping=1.0, max_radius=8.0)
    grid = make_neighbor_grid(params, cpu)

    relax_contacts(pop, grid, params, dt=0.01, substeps=50)

    np.testing.assert_array_equal(pop.positions_numpy(), pos)
    assert pop.count == 1


def test_pair_overlap_decays_at_analytical_rate(cpu):
    """Explicit Euler on one pair contracts the overlap by exactly (1 - 2*rate*dt) per substep."""
    r = 8.0
    overlap0 = 3.0
    pos = np.array([[0.0, 0.0, 0.0], [2.0 * r - overlap0, 0.0, 0.0]], dtype=np.float32)
    pop = _make(pos, np.full(2, r), cpu)
    params = ContactParams(stiffness=10.0, damping=1.0, max_radius=r)
    grid = make_neighbor_grid(params, cpu)
    dt = 0.01
    n = 15  # overlap decays to ~0.1 um: still far above float32 resolution on 16 um positions

    relax_contacts(pop, grid, params, dt=dt, substeps=n)

    x = pop.positions_numpy()
    overlap = 2.0 * r - np.linalg.norm(x[1] - x[0])
    expected = overlap0 * (1.0 - 2.0 * params.rate * dt) ** n
    assert overlap == pytest.approx(expected, rel=1e-3)
    # Symmetric push: the pair separates along its own axis and its midpoint does not move.
    np.testing.assert_allclose(x[0] + x[1], [2.0 * r - overlap0, 0.0, 0.0], atol=1e-4)
    assert np.all(x[:, 1:] == 0.0)


def test_non_overlapping_pair_does_not_interact(cpu):
    r = 8.0
    pos = np.array([[0.0, 0.0, 0.0], [2.0 * r + 0.5, 0.0, 0.0]], dtype=np.float32)
    pop = _make(pos, np.full(2, r), cpu)
    params = ContactParams(stiffness=10.0, damping=1.0, max_radius=r, margin=2.0)
    grid = make_neighbor_grid(params, cpu)

    relax_contacts(pop, grid, params, dt=0.01, substeps=10)

    np.testing.assert_array_equal(pop.positions_numpy(), pos)


def test_centre_of_mass_is_conserved(cpu):
    """Pair forces are exactly antisymmetric, so the net displacement of the cluster is zero."""
    r = 8.0
    pos = spherical_cluster(1500, radius=r, spacing_factor=0.85, jitter=0.2, seed=2)
    pop = _make(pos, np.full(1500, r), cpu)
    params = ContactParams(stiffness=10.0, damping=1.0, max_radius=r)
    grid = make_neighbor_grid(params, cpu)

    com_before = pop.positions_numpy().astype(np.float64).mean(axis=0)
    relax_contacts(pop, grid, params, dt=0.01, substeps=100)
    com_after = pop.positions_numpy().astype(np.float64).mean(axis=0)

    np.testing.assert_allclose(com_after, com_before, atol=1e-3)


def test_warp_matches_numpy_reference(cpu):
    """Computational validation: the hash-grid kernel equals the O(N^2) reference step by step."""
    r = 8.0
    n = 400
    rng = np.random.default_rng(9)
    pos = spherical_cluster(n, radius=r, spacing_factor=0.85, jitter=0.3, seed=9)
    radii = rng.uniform(6.0, 9.0, size=n).astype(np.float32)
    pop = _make(pos, radii, cpu)
    params = ContactParams(stiffness=10.0, damping=1.0, max_radius=9.0, margin=0.5)
    grid = make_neighbor_grid(params, cpu)
    dt = 0.01

    ref = pos.astype(np.float64)
    for _ in range(20):
        contact_substep(pop, grid, params, dt)
        ref = contact_substep_reference(ref, radii, params.rate, dt)
        np.testing.assert_allclose(pop.positions_numpy(), ref, atol=2e-3)

    assert np.linalg.norm(ref - pos, axis=1).max() > 0.5  # the cluster actually moved


def test_compressed_cluster_relaxes(cpu):
    """A cluster with 10% initial compression expands until overlaps are small; no cell is lost."""
    r = 8.0
    n = 1000
    pos = spherical_cluster(n, radius=r, spacing_factor=0.9, jitter=0.1, seed=0)
    pop = _make(pos, np.full(n, r), cpu, capacity=2 * n)
    params = ContactParams(stiffness=10.0, damping=1.0, max_radius=r, margin=0.5)
    grid = make_neighbor_grid(params, cpu)

    def max_overlap():
        x = pop.positions_numpy()
        d = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        return (2.0 * r - d).max()

    before = max_overlap()
    relax_contacts(pop, grid, params, dt=0.01, substeps=600)
    after = max_overlap()

    assert before > 1.0
    assert after < 0.05 * r
    assert pop.count == n
    assert np.all(np.isfinite(pop.positions_numpy()))


def test_substep_stability_rule():
    params = ContactParams(stiffness=10.0, damping=1.0, max_radius=8.0)
    params.check_substep(0.04)
    with pytest.raises(ValueError):
        params.check_substep(0.1)
    with pytest.raises(ValueError):
        params.check_substep(0.0)
