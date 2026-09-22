import numpy as np
import pytest

from warpbiocell.cells.initialization import spherical_cluster
from warpbiocell.cells.state import CellPopulation


def test_population_prefix_layout(cpu):
    pos = np.arange(15, dtype=np.float32).reshape(5, 3)
    pop = CellPopulation.from_numpy(pos, np.full(5, 8.0), capacity=12, device=cpu)

    assert pop.count == 5
    assert pop.capacity == 12
    assert pop.position.shape == (12,)
    assert pop.active_position.shape == (5,)
    np.testing.assert_array_equal(pop.positions_numpy(), pos)
    # Free slots stay zeroed and are never handed to callers.
    assert np.all(pop.position.numpy()[5:] == 0.0)


def test_population_rejects_bad_shapes(cpu):
    with pytest.raises(ValueError):
        CellPopulation.from_numpy(np.zeros((4, 2)), np.ones(4), device=cpu)
    with pytest.raises(ValueError):
        CellPopulation.from_numpy(np.zeros((4, 3)), np.ones(3), device=cpu)
    with pytest.raises(ValueError):
        CellPopulation.from_numpy(np.zeros((4, 3)), np.ones(4), capacity=2, device=cpu)


def test_spherical_cluster_shape_and_determinism():
    a = spherical_cluster(500, radius=8.0, seed=3)
    b = spherical_cluster(500, radius=8.0, seed=3)
    c = spherical_cluster(500, radius=8.0, seed=4)

    assert a.shape == (500, 3)
    assert a.dtype == np.float32
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_spherical_cluster_is_compact_and_non_coincident():
    r = 8.0
    pts = spherical_cluster(2000, radius=r, spacing_factor=0.9, jitter=0.1, seed=0)

    # Cells stay within a sphere not much larger than the ideal packed radius.
    ideal = r * (2000 / 0.52) ** (1.0 / 3.0)  # cubic packing ~0.52
    assert np.linalg.norm(pts, axis=1).max() < 1.3 * ideal

    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    assert d.min() > 1.0  # never coincident; lattice spacing minus jitter is 1.6 r = 12.8 um
