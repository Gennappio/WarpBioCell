"""Tissue regions: shapes, sampling, seeding, confinement, tissue-surface oxygen boundary."""

import numpy as np
import pytest

from warpbiocell.cells.initialization import spherical_cluster
from warpbiocell.cells.mechanics import ContactParams, make_neighbor_grid, relax_contacts
from warpbiocell.cells.state import CellPopulation
from warpbiocell.fields.diffusion import SolverSettings
from warpbiocell.fields.oxygen import OxygenField, OxygenParams
from warpbiocell.fields.scalar_field import Boundary, GridGeometry
from warpbiocell.geometry.region import TissueRegion
from warpbiocell.geometry.seeding import CellBudgetError, fill_region
from warpbiocell.geometry.shapes import Ellipsoid, Sphere, Union, evaluate_on_grid
from warpbiocell.reference.fields import steady_state_reference

R = 8.0


def test_sphere_sdf_is_exact_on_nodes_and_volume_matches():
    geom = GridGeometry.centered_cube(400.0, 20.0)
    sphere = Sphere(radius=150.0)
    sdf = evaluate_on_grid(sphere, geom)
    x, y, z = np.meshgrid(*geom.node_coordinates(), indexing="ij")
    np.testing.assert_allclose(sdf, np.sqrt(x**2 + y**2 + z**2) - 150.0, atol=1e-4)
    region = TissueRegion(geom, sdf, device="cpu")
    assert abs(region.volume() / sphere.volume - 1.0) < 0.05


def test_ellipsoid_and_union_have_the_right_sign_and_bounds():
    ell = Ellipsoid(center=(10.0, 0.0, 0.0), radii=(200.0, 100.0, 50.0))
    x = np.array([10.0, 209.0, 211.0, 10.0, 10.0])
    y = np.array([0.0, 0.0, 0.0, 99.0, 101.0])
    z = np.zeros(5)
    d = ell.sdf(x, y, z)
    assert d[0] < 0 and d[1] < 0 and d[2] > 0 and d[3] < 0 and d[4] > 0
    assert abs(d[1] + 1.0) < 0.2 and abs(d[2] - 1.0) < 0.2  # nearly exact on the axes
    lo, hi = ell.bounds
    np.testing.assert_allclose(lo, [-190.0, -100.0, -50.0])
    np.testing.assert_allclose(hi, [210.0, 100.0, 50.0])

    union = Union((Sphere(center=(-100.0, 0.0, 0.0), radius=80.0), Sphere(center=(100.0, 0.0, 0.0), radius=80.0)))
    d = union.sdf(np.array([-100.0, 100.0, 0.0]), np.zeros(3), np.zeros(3))
    assert d[0] == -80.0 and d[1] == -80.0 and d[2] == 20.0
    with pytest.raises(ValueError):
        Union(())


def test_region_sampling_host_and_device_agree_with_the_analytical_distance(cpu):
    geom = GridGeometry.centered_cube(400.0, 20.0)
    region = TissueRegion.from_shape(Sphere(radius=150.0), geom, device=cpu)
    rng = np.random.default_rng(3)
    pts = rng.uniform(-180.0, 180.0, size=(300, 3)).astype(np.float32)
    exact = np.linalg.norm(pts, axis=1) - 150.0
    host = region.sample(pts)
    pop = CellPopulation.from_numpy(pts, np.full(300, R), device=cpu)
    out = pop.age  # any float32 scratch of the right size
    region.sample_at(pop.position, 300, out)
    device = out.numpy()[:300]
    np.testing.assert_allclose(host, device, atol=1e-4)
    assert np.abs(host - exact).max() < 0.15 * geom.dx  # trilinear error on a curved surface
    np.testing.assert_array_equal(region.contains(pts), host < 0.0)


def test_fill_region_seeds_inside_at_the_expected_density_and_respects_the_budget(cpu):
    geom = GridGeometry.centered_cube(600.0, 20.0)
    sphere = Sphere(radius=200.0)
    region = TissueRegion.from_shape(sphere, geom, device=cpu)

    a = fill_region(region, R, spacing_factor=1.0, jitter=0.1, seed=1)
    b = fill_region(region, R, spacing_factor=1.0, jitter=0.1, seed=1)
    np.testing.assert_array_equal(a, b)
    assert np.all(region.contains(a, margin=R))  # every whole sphere inside
    expected = sphere.volume / (2 * R) ** 3  # cubic lattice at one diameter spacing
    assert abs(a.shape[0] / expected - 1.0) < 0.15

    inner = fill_region(region, R, seed=1, within=Sphere(radius=100.0))
    assert 0 < inner.shape[0] < a.shape[0]
    assert np.all(np.linalg.norm(inner, axis=1) < 100.0)
    with pytest.raises(CellBudgetError, match="holds"):
        fill_region(region, R, seed=1, max_cells=100)


def test_confinement_pushes_cells_back_inside(cpu):
    geom = GridGeometry.centered_cube(400.0, 20.0)
    region = TissueRegion.from_shape(Sphere(radius=100.0), geom, device=cpu)
    pos = np.array([[130.0, 0.0, 0.0], [0.0, -120.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    pop = CellPopulation.from_numpy(pos, np.full(3, R), device=cpu)
    params = ContactParams(stiffness=10.0, damping=1.0, max_radius=R, wall_rate=10.0)
    grid = make_neighbor_grid(params, cpu)

    relax_contacts(pop, grid, params, dt=0.02, substeps=100, region=region)

    x = pop.positions_numpy()
    r = np.linalg.norm(x, axis=1)
    assert r[0] < 100.0 - R + 0.5 and r[1] < 100.0 - R + 0.5  # back inside, whole sphere within the tissue
    np.testing.assert_allclose(x[0] / r[0], [1.0, 0.0, 0.0], atol=1e-4)  # pushed radially
    np.testing.assert_array_equal(x[2], [0.0, 0.0, 0.0])  # untouched


def test_growing_population_stays_confined(cpu):
    geom = GridGeometry.centered_cube(400.0, 20.0)
    # A compressed 600-cell cluster relaxes freely to R ~ 84 um; the tissue surface at 85 um holds it.
    region = TissueRegion.from_shape(Sphere(radius=85.0), geom, device=cpu)
    pos = spherical_cluster(600, R, spacing_factor=0.85, jitter=0.2, seed=2)
    pop = CellPopulation.from_numpy(pos, np.full(600, R), device=cpu)
    params = ContactParams(stiffness=10.0, damping=1.0, max_radius=R, wall_rate=10.0)
    grid = make_neighbor_grid(params, cpu)

    relax_contacts(pop, grid, params, dt=0.02, substeps=300, region=region)

    r = np.linalg.norm(pop.positions_numpy(), axis=1)
    assert r.max() < 85.0  # every centre inside the surface (the soft wall allows some penetration of the sphere)
    free = CellPopulation.from_numpy(pos, np.full(600, R), device=cpu)
    relax_contacts(free, make_neighbor_grid(params, cpu), params, dt=0.02, substeps=300)
    assert np.linalg.norm(free.positions_numpy(), axis=1).max() > r.max() + 3.0  # the wall did something


def test_mechanics_without_region_is_unchanged(cpu):
    pos = spherical_cluster(400, R, spacing_factor=0.9, jitter=0.1, seed=5)
    params = ContactParams(stiffness=10.0, damping=1.0, max_radius=R)
    results = []
    for region in (None, None):
        pop = CellPopulation.from_numpy(pos, np.full(400, R), device=cpu)
        relax_contacts(pop, make_neighbor_grid(params, cpu), params, dt=0.02, substeps=50, region=region)
        results.append(pop.positions_numpy())
    np.testing.assert_array_equal(results[0], results[1])


def test_wall_rate_stability_rule():
    params = ContactParams(stiffness=1.0, damping=1.0, max_radius=R, wall_rate=40.0)
    with pytest.raises(ValueError, match="wall_rate"):
        params.check_substep(0.02)


def test_tissue_surface_oxygen_boundary_matches_exact_discrete_solution(cpu):
    geom = GridGeometry(origin=(0.0, 0.0, 0.0), dx=10.0, shape=(9, 8, 8))
    region = TissueRegion.from_shape(Sphere(center=(40.0, 35.0, 35.0), radius=28.0), geom, device=cpu)
    params = OxygenParams(diffusion_coefficient=2.0e3, uptake_max=5.0e4, michaelis_k=3.0, boundary_value=20.0)
    ox = OxygenField(geom, params, SolverSettings(omega=1.5, tolerance=1e-5, max_sweeps=20000, check_every=10), device=cpu, fixed_outside=region)
    density = np.where(region.inside_mask(), 1.5e-3, 0.0).astype(np.float32)
    ox.density.assign(density)

    report = ox.solve_steady_state()

    outside = ~region.inside_mask()
    values = ox.numpy()
    assert report.converged
    np.testing.assert_array_equal(values[outside], 20.0)  # pinned to the boundary value
    assert values[region.inside_mask()].min() < 20.0
    exact = steady_state_reference(np.full(geom.shape, 20.0), density, params.diffusion_coefficient, params.uptake_max, params.michaelis_k, geom.dx, (Boundary.DIRICHLET,) * 3, fixed_extra=outside)
    np.testing.assert_allclose(values, exact, atol=2e-3)
    assert ox.field.interior_mask().sum() == region.inside_mask().sum()


def test_experiment_with_ellipsoid_tissue(tmp_path):
    from warpbiocell.simulation.config import config_from_dict
    from warpbiocell.simulation.experiment import Experiment

    config = config_from_dict(
        {
            "name": "tissue",
            "simulation": {"duration_h": 2.0, "dt_cells_h": 0.5, "dt_mechanics_h": 0.05, "device": "cpu"},
            "cells": {"max_cells": 5000},
            "lifecycle": {"division_rate_per_h": 0.2},
            "oxygen": {"grid": {"box_um": 400.0}},
            "geometry": {"enabled": True, "shape": "ellipsoid", "radii_um": [120.0, 90.0, 60.0], "oxygen_source": "tissue_surface"},
            "output": {"metrics_every_h": 1.0, "figures": True},
        }
    )
    assert config.validate() == []
    experiment = Experiment(config)
    seeded = experiment.population.count
    assert 100 < seeded < 5000

    result = experiment.run(tmp_path / "run")

    assert result.status == "completed"
    assert result.summary["cells"] > seeded
    assert result.summary["cells_outside_tissue"] < 0.05 * result.summary["cells"]
    assert result.summary["oxygen_cells_min_mmHg"] < 38.0
    import json

    meta = json.loads((tmp_path / "run" / "metadata.json").read_text())
    assert meta["tissue"]["seeded_cells"] == seeded and meta["tissue"]["confined"] is True
    assert (tmp_path / "run" / "figures" / "cells_3d.png").exists()


def test_geometry_config_validation():
    from warpbiocell.simulation.config import ConfigError, config_from_dict

    with pytest.raises(ConfigError, match="does not fit"):
        config_from_dict({"geometry": {"enabled": True, "shape": "sphere", "radii_um": [500.0]}}).validate()
    with pytest.raises(ConfigError, match="shape"):
        config_from_dict({"geometry": {"enabled": True, "shape": "cube"}}).validate()
    with pytest.raises(ConfigError, match="mask_file"):
        config_from_dict({"geometry": {"enabled": True, "shape": "mask"}}).validate()
    with pytest.raises(ConfigError, match="oxygen_source"):
        config_from_dict({"geometry": {"enabled": True, "oxygen_source": "vessel"}}).validate()
