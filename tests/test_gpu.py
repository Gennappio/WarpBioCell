"""CUDA-only checks. Skipped automatically when no CUDA device exists (see conftest.py)."""

import numpy as np
import pytest

from warpbiocell.cells.initialization import spherical_cluster
from warpbiocell.cells.mechanics import ContactParams, make_neighbor_grid, relax_contacts
from warpbiocell.cells.state import CellPopulation

pytestmark = pytest.mark.gpu


def _simulate(device, substeps=100):
    r = 8.0
    n = 3000
    pos = spherical_cluster(n, radius=r, spacing_factor=0.85, jitter=0.2, seed=21)
    pop = CellPopulation.from_numpy(pos, np.full(n, r), device=device)
    params = ContactParams(stiffness=10.0, damping=1.0, max_radius=r, margin=0.5)
    grid = make_neighbor_grid(params, device)
    relax_contacts(pop, grid, params, dt=0.01, substeps=substeps)
    return pop.positions_numpy()


def test_cuda_matches_cpu_within_float_tolerance(cpu, cuda):
    """Same law, different summation order: expect agreement to float32 round-off, not bitwise."""
    a = _simulate(cpu)
    b = _simulate(cuda)
    np.testing.assert_allclose(a, b, atol=1e-2)


def test_cuda_run_is_reproducible(cuda):
    a = _simulate(cuda)
    b = _simulate(cuda)
    np.testing.assert_array_equal(a, b)
