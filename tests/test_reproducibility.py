import numpy as np

from warpbiocell.cells.initialization import spherical_cluster
from warpbiocell.cells.mechanics import ContactParams, make_neighbor_grid, relax_contacts
from warpbiocell.cells.state import CellPopulation


def _simulate(seed, device, substeps=50):
    r = 8.0
    pos = spherical_cluster(800, radius=r, spacing_factor=0.85, jitter=0.2, seed=seed)
    pop = CellPopulation.from_numpy(pos, np.full(800, r), device=device)
    params = ContactParams(stiffness=10.0, damping=1.0, max_radius=r, margin=0.5)
    grid = make_neighbor_grid(params, device)
    relax_contacts(pop, grid, params, dt=0.01, substeps=substeps)
    return pop.positions_numpy()


def test_same_seed_same_device_is_bitwise_reproducible(cpu):
    a = _simulate(seed=7, device=cpu)
    b = _simulate(seed=7, device=cpu)
    np.testing.assert_array_equal(a, b)


def test_different_seed_differs(cpu):
    a = _simulate(seed=7, device=cpu)
    b = _simulate(seed=8, device=cpu)
    assert not np.array_equal(a, b)
