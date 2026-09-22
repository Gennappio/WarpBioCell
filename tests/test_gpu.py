"""CUDA-only checks. Skipped automatically when no CUDA device exists (see conftest.py).

Run them on the CUDA machine with ``pytest -m gpu``. They compare CUDA against the CPU
implementation on the same inputs: bitwise where the algorithm is order-independent
(integer deposit), float round-off where it is not (mechanics, field), statistical for the
biology (threshold crossings can flip on round-off).
"""

from pathlib import Path

import numpy as np
import pytest

from warpbiocell.cells.initialization import spherical_cluster
from warpbiocell.cells.lifecycle import LifecycleParams, lifecycle_step
from warpbiocell.cells.mechanics import ContactParams, contact_substep, make_neighbor_grid, relax_contacts
from warpbiocell.cells.state import CellPopulation
from warpbiocell.fields.diffusion import SolverSettings
from warpbiocell.fields.oxygen import OxygenField, OxygenParams
from warpbiocell.fields.scalar_field import GridGeometry
from warpbiocell.simulation.config import config_from_dict
from warpbiocell.simulation.experiment import Experiment

pytestmark = pytest.mark.gpu

R = 8.0


def _cluster(n, seed=21):
    return spherical_cluster(n, R, spacing_factor=0.85, jitter=0.2, seed=seed)


def _relax(device, substeps=100):
    n = 3000
    pop = CellPopulation.from_numpy(_cluster(n), np.full(n, R), device=device)
    params = ContactParams(stiffness=10.0, damping=1.0, max_radius=R, margin=0.5)
    grid = make_neighbor_grid(params, device)
    relax_contacts(pop, grid, params, dt=0.01, substeps=substeps)
    return pop


def test_mechanics_cuda_matches_cpu_within_float_tolerance(cpu, cuda):
    a = _relax(cpu).positions_numpy()
    b = _relax(cuda).positions_numpy()
    np.testing.assert_allclose(a, b, atol=1e-2)


def test_mechanics_cuda_run_is_reproducible(cuda):
    a = _relax(cuda).positions_numpy()
    b = _relax(cuda).positions_numpy()
    np.testing.assert_array_equal(a, b)


def test_neighbor_counts_identical_on_cuda(cpu, cuda):
    n = 2000
    params = ContactParams(stiffness=1.0, damping=1.0, max_radius=R, margin=2.0)
    counts = []
    for device in (cpu, cuda):
        pop = CellPopulation.from_numpy(_cluster(n), np.full(n, R), device=device)
        grid = make_neighbor_grid(params, device)
        contact_substep(pop, grid, params, dt=0.0)
        counts.append(pop.neighbor_counts_numpy())
    np.testing.assert_array_equal(counts[0], counts[1])


def test_field_deposit_is_bitwise_identical_and_order_independent(cpu, cuda):
    n = 5000
    geom = GridGeometry.centered_cube(600.0, 20.0)
    densities = []
    for device in (cpu, cuda, cuda):
        pop = CellPopulation.from_numpy(_cluster(n), np.full(n, R), device=device)
        ox = OxygenField(geom, OxygenParams(), device=device)
        ox.deposit_uptake(pop)
        densities.append(ox.density.numpy())
    np.testing.assert_array_equal(densities[1], densities[2])  # int64 atomics: no order dependence
    np.testing.assert_array_equal(densities[0], densities[1])


def test_field_steady_state_matches_cpu(cpu, cuda):
    n = 8000
    geom = GridGeometry.centered_cube(800.0, 20.0)
    fields, oxygen_at_cells = [], []
    for device in (cpu, cuda):
        pop = CellPopulation.from_numpy(_cluster(n), np.full(n, R), device=device)
        ox = OxygenField(geom, OxygenParams(), SolverSettings(omega=1.8, tolerance=1e-6, check_every=10), device=device)
        report = ox.update(pop)
        assert report.converged
        fields.append(ox.numpy())
        oxygen_at_cells.append(pop.oxygen_numpy())
    np.testing.assert_allclose(fields[0], fields[1], atol=5e-3)
    np.testing.assert_allclose(oxygen_at_cells[0], oxygen_at_cells[1], atol=5e-3)


def test_lifecycle_statistics_match_cpu(cpu, cuda):
    n = 20000
    params = LifecycleParams(division_rate=0.1, death_rate=0.02, inhibition_threshold=10**6)
    created, dead = [], []
    for device in (cpu, cuda):
        pop = CellPopulation.from_numpy(_cluster(n), np.full(n, R), capacity=3 * n, device=device, seed=5)
        created.append(lifecycle_step(pop, params, dt=1.0))
        dead.append(int((pop.states_numpy() == 3).sum()))
    # Same per-cell RNG streams on both devices: decisions should coincide exactly here because
    # no threshold depends on floating-point inputs in this configuration.
    assert created[0] == created[1]
    assert dead[0] == dead[1]


def test_experiment_runs_on_cuda(cuda, tmp_path):
    config = config_from_dict(
        {
            "simulation": {"duration_h": 2.0, "dt_cells_h": 0.5, "dt_mechanics_h": 0.05, "device": "cuda:0"},
            "cells": {"initial_count": 500, "max_cells": 5000},
            "oxygen": {"grid": {"box_um": 400.0}},
            "output": {"metrics_every_h": 1.0, "figures": False},
        }
    )
    result = Experiment(config).run(tmp_path / "run")
    assert result.status == "completed"
    assert result.summary["cells"] >= 500
    assert (tmp_path / "run" / "metrics.csv").exists()


def test_network_updates_match_cpu(cpu, cuda):
    """Synchronous sweeps are deterministic (bitwise equal); asynchronous and MaBoSS updates
    draw from the same per-cell streams, so they are bitwise equal too."""
    from warpbiocell.network.model import load_network
    from warpbiocell.network.runtime import InputClamp, NetworkParams, NetworkRuntime

    networks = Path(__file__).resolve().parents[1] / "configs" / "networks"
    net = load_network(networks / "microc_jaya.bnd", networks / "microc_jaya.cfg")
    clamps = (InputClamp("Oxygen_supply", "oxygen", 15.7), InputClamp("Glucose_supply", "glucose", 4.0), InputClamp("EGFR_stimulus", "constant", 1.0))
    n = 4000
    for mode, kw in (("synchronous", {"updates_per_step": 5}), ("asynchronous", {"updates_per_step": 300}), ("maboss", {"time_units_per_h": 8.0})):
        results = []
        for device in (cpu, cuda):
            pop = CellPopulation.from_numpy(_cluster(n), np.full(n, R), device=device, seed=13)
            o = np.zeros(pop.capacity, dtype=np.float32)
            o[: n // 2] = 30.0
            pop.oxygen_local.assign(o)
            pop.glucose_local.assign(np.full(pop.capacity, 5.0, dtype=np.float32))
            rt = NetworkRuntime(NetworkParams(net, update=mode, inputs=clamps, **kw), pop.capacity, device=device, species_names=("oxygen", "glucose"))
            rt.initialize(pop)
            for _ in range(3):
                rt.step(pop, dt_h=0.25)
            results.append((rt.states_numpy(pop), pop.fate_flags_numpy()))
        assert np.array_equal(results[0][0], results[1][0]), mode
        assert np.array_equal(results[0][1], results[1][1]), mode


def test_network_experiment_runs_on_cuda(cuda, tmp_path):
    from warpbiocell.simulation.config import load_config

    config = load_config(
        Path(__file__).resolve().parents[1] / "configs" / "tumor_spheroid_network.yaml",
        ["simulation.duration_h=2.0", "simulation.device=cuda:0", "cells.initial_count=500", "cells.max_cells=5000", "oxygen.grid.box_um=400.0", "output.figures=false"],
    )
    result = Experiment(config).run(tmp_path / "run")
    assert result.status == "completed"
    assert result.summary["network_proliferation_cells"] >= 0
