"""Metabolic species: per-state densities, production from a source species, lifecycle rules, experiment."""

import csv
import json

import numpy as np
import pytest

from warpbiocell.cells.initialization import spherical_cluster
from warpbiocell.cells.lifecycle import LifecycleParams, lifecycle_step
from warpbiocell.cells.model import CellState
from warpbiocell.cells.state import CellPopulation
from warpbiocell.fields.densities import CellDensities
from warpbiocell.fields.diffusion import SolverSettings
from warpbiocell.fields.metabolism import MetabolicFields
from warpbiocell.fields.oxygen import OxygenField, OxygenParams
from warpbiocell.fields.scalar_field import Boundary, GridGeometry
from warpbiocell.fields.species import SpeciesField, SpeciesParams, by_state
from warpbiocell.io.checkpoints import load_checkpoint
from warpbiocell.reference.fields import deposit_reference, steady_state_reference
from warpbiocell.simulation.config import ConfigError, config_from_dict
from warpbiocell.simulation.experiment import Experiment

R = 8.0
DIRICHLET = (Boundary.DIRICHLET,) * 3


def _mixed_population(n, device, hypoxic_fraction=0.4, seed=1):
    pos = spherical_cluster(n, R, spacing_factor=1.0, jitter=0.1, seed=seed)
    states = np.full(n, int(CellState.PROLIFERATIVE), dtype=np.int32)
    r = np.linalg.norm(pos, axis=1)
    states[r < np.quantile(r, hypoxic_fraction)] = int(CellState.HYPOXIC)  # hypoxic core
    states[::17] = int(CellState.DEAD)
    return CellPopulation.from_numpy(pos, np.full(n, R), device=device, states=states), pos, states


def test_by_state_parsing():
    assert by_state(None) == (0.0, 0.0, 0.0, 0.0)
    assert by_state(2.0) == (2.0, 2.0, 2.0, 0.0)
    assert by_state({"hypoxic": 3.0, "PROLIFERATIVE": 1.0}) == (1.0, 0.0, 3.0, 0.0)
    assert by_state({"dead": 1.0}, default=1.0) == (1.0, 1.0, 1.0, 1.0)
    with pytest.raises(KeyError):
        by_state({"zombie": 1.0})


def test_per_state_densities_match_reference_and_combine(cpu):
    pop, pos, states = _mixed_population(500, cpu)
    geom = GridGeometry.centered_cube(400.0, 20.0)
    densities = CellDensities(geom, device=cpu)
    densities.deposit(pop)
    dens = densities.numpy()
    for state in CellState:
        expected = deposit_reference(pos, states == int(state), geom.origin, geom.dx, geom.shape)
        np.testing.assert_allclose(dens[int(state)], expected, atol=1e-8)
    total = densities.total_numpy()
    assert abs(total.sum() * geom.voxel_volume - 500) < 1e-3
    out = OxygenField(geom, OxygenParams(), device=cpu, densities=densities, uptake_state_factors={"hypoxic": 0.5})
    out.combine()
    expected = dens[0] + dens[1] + 0.5 * dens[2]
    np.testing.assert_allclose(out.density.numpy(), expected, rtol=1e-6, atol=1e-9)


def test_species_chain_matches_exact_discrete_solution(cpu):
    """Glucose with per-state uptake, then lactate produced by hypoxic cells from the glucose uptake."""
    geom = GridGeometry(origin=(-60.0, -60.0, -60.0), dx=15.0, shape=(9, 9, 9))
    pop, pos, states = _mixed_population(150, cpu, hypoxic_fraction=0.5, seed=4)
    settings = SolverSettings(omega=1.5, tolerance=1e-6, max_sweeps=50000, check_every=10)
    glucose_p = SpeciesParams("glucose", "mM", diffusion=2.4e5, michaelis_k=0.04, boundary_value=5.0, uptake_max=(3.0e4, 3.0e4, 4.5e5, 0.0))
    lactate_p = SpeciesParams("lactate", "mM", diffusion=2.4e5, michaelis_k=1.0, boundary_value=1.0, production=(0.0, 0.0, 9.0e5, 0.0), production_source="glucose")
    fields = MetabolicFields(geom, OxygenParams(), [glucose_p, lactate_p], settings, device=cpu)

    report = fields.update(pop)

    assert report.converged and set(report.reports) == {"oxygen", "glucose", "lactate"}
    dens = fields.densities.numpy()
    glucose = fields["glucose"]
    eff = sum(w * dens[k] for k, w in enumerate(glucose_p.uptake_weights))
    exact_g = steady_state_reference(np.full(geom.shape, 5.0), eff, 2.4e5, glucose_p.uptake_scale, 0.04, geom.dx, DIRICHLET)
    np.testing.assert_allclose(glucose.numpy(), exact_g, atol=2e-3)
    assert exact_g.min() < 4.9  # the core really consumes (small domain: a few percent)

    production = 9.0e5 * dens[int(CellState.HYPOXIC)]
    exact_l = steady_state_reference(np.full(geom.shape, 1.0), np.zeros(geom.shape), 2.4e5, 0.0, 1.0, geom.dx, DIRICHLET, production=production, source=exact_g, source_k=0.04)
    np.testing.assert_allclose(fields["lactate"].numpy(), exact_l, atol=2e-3)
    assert exact_l.max() > 1.05  # produced above its boundary value
    # Sampled values reach the cells.
    assert pop.glucose_numpy().min() < 5.0 and pop.local_numpy("lactate").max() > 1.0


def test_no_glycolytic_cells_means_no_lactate_and_no_glucose_means_no_production(cpu):
    geom = GridGeometry.centered_cube(300.0, 20.0)
    lactate_p = SpeciesParams("lactate", "mM", diffusion=2.4e5, michaelis_k=1.0, boundary_value=1.0, production=(0.0, 0.0, 9.0e5, 0.0), production_source="glucose")
    glucose_p = SpeciesParams("glucose", "mM", diffusion=2.4e5, michaelis_k=0.04, boundary_value=5.0, uptake_max=(3.0e4, 3.0e4, 4.5e5, 0.0))

    pos = spherical_cluster(300, R, seed=2)
    pop = CellPopulation.from_numpy(pos, np.full(300, R), device=cpu)  # all PROLIFERATIVE
    fields = MetabolicFields(geom, OxygenParams(), [glucose_p, lactate_p], device=cpu)
    fields.update(pop)
    np.testing.assert_allclose(fields["lactate"].numpy(), 1.0, atol=1e-6)

    pop_h, _, _ = _mixed_population(300, cpu, hypoxic_fraction=1.0, seed=2)  # all living cells hypoxic
    starved = SpeciesParams("glucose", "mM", diffusion=2.4e5, michaelis_k=0.04, boundary_value=0.0, uptake_max=(3.0e4, 3.0e4, 4.5e5, 0.0))
    fields = MetabolicFields(geom, OxygenParams(), [starved, lactate_p], device=cpu)
    fields.update(pop_h)
    np.testing.assert_allclose(fields["lactate"].numpy(), 1.0, atol=1e-6)  # no glucose, no lactate


def test_metabolic_fields_validate_species_order_and_names(cpu):
    geom = GridGeometry.centered_cube(200.0, 20.0)
    lactate_p = SpeciesParams("lactate", "mM", diffusion=1.0e5, michaelis_k=1.0, boundary_value=1.0, production=(0.0, 0.0, 1.0, 0.0), production_source="glucose")
    with pytest.raises(ValueError, match="earlier species"):
        MetabolicFields(geom, OxygenParams(), [lactate_p], device=cpu)
    with pytest.raises(ValueError, match="duplicate"):
        MetabolicFields(geom, OxygenParams(), [SpeciesParams("oxygen", "mmHg", 1.0e5, 1.0, 1.0)], device=cpu)


def test_lifecycle_glucose_rules(cpu):
    n = 400
    pos = spherical_cluster(n, R, seed=0)
    pop = CellPopulation.from_numpy(pos, np.full(n, R), capacity=3 * n, device=cpu)
    # Anoxic everywhere; glucose available for the first half only.
    o = pop.oxygen_local.numpy(); o[:n] = 1.0; pop.oxygen_local.assign(o)
    g = pop.glucose_local.numpy(); g[: n // 2] = 5.0; g[n // 2 : n] = 0.1; pop.glucose_local.assign(g)
    params = LifecycleParams(division_rate=0.0, anoxic_death_rate=1.0e3, hypoxia_threshold=8.0, death_threshold=2.0, glucose_threshold=4.0, glucose_death_threshold=0.5, necrosis_requires_glucose=True, inhibition_threshold=10**6)

    lifecycle_step(pop, params, dt=1.0)

    states = pop.states_numpy()
    assert np.all(states[: n // 2] == int(CellState.HYPOXIC))  # anoxic but fed: alive
    assert np.all(states[n // 2 :] == int(CellState.DEAD))  # anoxic and starved: dead

    # Glucose ramps the division rate: 5 mM full, 2.25 mM half, 0.5 mM none (oxygen ample).
    pop = CellPopulation.from_numpy(pos, np.full(n, R), capacity=3 * n, device=cpu)
    o = pop.oxygen_local.numpy(); o[:n] = 30.0; pop.oxygen_local.assign(o)
    g = pop.glucose_local.numpy(); g[:n] = 0.5; pop.glucose_local.assign(g)
    params = LifecycleParams(division_rate=1.0e3, hypoxia_threshold=8.0, death_threshold=2.0, glucose_threshold=4.0, glucose_death_threshold=0.5, inhibition_threshold=10**6)
    assert lifecycle_step(pop, params, dt=1.0) == 0
    g = pop.glucose_local.numpy(); g[:n] = 5.0; pop.glucose_local.assign(g)
    assert lifecycle_step(pop, params, dt=1.0) == n
    with pytest.raises(ValueError):
        LifecycleParams(glucose_threshold=1.0, glucose_death_threshold=2.0)


def test_experiment_with_species(tmp_path):
    config = config_from_dict(
        {
            "name": "metabolic",
            "simulation": {"duration_h": 2.0, "dt_cells_h": 0.5, "dt_mechanics_h": 0.05, "device": "cpu"},
            "cells": {"initial_count": 300, "max_cells": 3000},
            "lifecycle": {"division_rate_per_h": 0.2, "glucose_threshold_mM": 4.0, "glucose_death_threshold_mM": 0.5, "necrosis_requires_glucose": True},
            "oxygen": {"grid": {"box_um": 400.0}, "uptake_state_factors": {"hypoxic": 0.5}},
            "species": [
                {"name": "glucose", "uptake_max_by_state": {"proliferative": 3.0e4, "quiescent": 3.0e4, "hypoxic": 4.5e5}},
                {"name": "lactate", "boundary_value": 1.0, "michaelis_k": 1.0, "production_by_state": {"hypoxic": 9.0e5}, "production_source": "glucose"},
            ],
            "output": {"metrics_every_h": 1.0, "checkpoint_every_h": 2.0, "figures": True},
        }
    )
    assert config.validate() == []
    assert config.has_glucose and config.lifecycle_params().necrosis_requires_glucose

    result = Experiment(config).run(tmp_path / "run")

    assert result.status == "completed"
    with (tmp_path / "run" / "metrics.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert 0.0 < float(rows[-1]["glucose_cells_min_mM"]) < 5.0
    assert float(rows[-1]["lactate_cells_max_mM"]) >= 1.0
    with (tmp_path / "run" / "profiles.csv").open() as f:
        profile = list(csv.DictReader(f))
    assert profile[0]["glucose_mean_mM"] != "" and profile[0]["lactate_mean_mM"] != ""
    state = load_checkpoint(tmp_path / "run" / "checkpoint" / "state_0002.00h.npz")
    assert {"field", "field_glucose", "field_lactate", "glucose_local", "lactate_local"} <= set(state)
    assert (tmp_path / "run" / "figures" / "species_slices.png").exists()
    meta = json.loads((tmp_path / "run" / "metadata.json").read_text())
    assert [s["name"] for s in meta["config"]["species"]] == ["glucose", "lactate"]


def test_species_config_validation():
    with pytest.raises(ConfigError, match="unique"):
        config_from_dict({"species": [{"name": "oxygen"}]}).validate()
    with pytest.raises(ConfigError, match="unique"):
        config_from_dict({"species": [{"name": "glucose"}, {"name": "glucose"}]}).validate()
    with pytest.raises(ConfigError, match="earlier species"):
        config_from_dict({"species": [{"name": "lactate", "production_source": "glucose"}]}).validate()
    with pytest.raises(ConfigError, match="species\\[0\\]"):
        config_from_dict({"species": [{"name": "glucose", "diffusion_um2_per_h": "fast"}]})
    warnings = config_from_dict({"lifecycle": {"necrosis_requires_glucose": True}}).validate()
    assert any("glucose" in w for w in warnings)
    # Without a glucose species the lifecycle ignores the glucose thresholds.
    params = config_from_dict({"lifecycle": {"glucose_threshold_mM": 4.0, "necrosis_requires_glucose": True}}).lifecycle_params()
    assert params.glucose_threshold == 0.0 and params.necrosis_requires_glucose is False


def test_oxygen_field_is_a_species_field(cpu):
    geom = GridGeometry.centered_cube(200.0, 20.0)
    ox = OxygenField(geom, OxygenParams(), device=cpu)
    assert isinstance(ox, SpeciesField) and ox.name == "oxygen" and ox.species.unit == "mmHg"
    assert ox.params.boundary_value == 38.0 and ox.species.uptake_weights == (1.0, 1.0, 1.0, 0.0)
