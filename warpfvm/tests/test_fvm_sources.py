"""Point sources to volumetric sources: bin_rates against MicroC's per-cell loop."""

import numpy as np
import pytest

import warpfvm as wf


def microc_source_field(nx, ny, nz, size_um, positions_um, rates):
    """MultiSubstanceSimulator._create_source_field_from_reactions, 3-D, positions in micrometres."""
    field = np.zeros(nx * ny * nz)
    dx_m = size_um[0] * 1e-6 / nx
    dy_m = size_um[1] * 1e-6 / ny
    dz_m = size_um[2] * 1e-6 / nz
    volume = dx_m * dy_m * dz_m
    for (x_pos, y_pos, z_pos), rate in zip(positions_um, rates):
        x = int(x_pos / (size_um[0] / nx))
        y = int(y_pos / (size_um[1] / ny))
        z = int(z_pos / (size_um[2] / nz))
        if 0 <= x < nx and 0 <= y < ny and 0 <= z < nz:
            field[z * (nx * ny) + y * nx + x] += rate / volume
    return field


def test_bin_rates_matches_microc(rng):
    nx, ny, nz, size_um = 15, 12, 10, (750.0, 600.0, 500.0)
    mesh = wf.Grid3D(dx=size_um[0] * 1e-6 / nx, dy=size_um[1] * 1e-6 / ny, dz=size_um[2] * 1e-6 / nz, nx=nx, ny=ny, nz=nz, device="cpu")
    positions_um = rng.random((4000, 3)) * np.array(size_um)
    rates = -rng.random(4000) * 3e-17
    ours = wf.bin_rates(mesh, positions_um * 1e-6, rates)
    theirs = microc_source_field(nx, ny, nz, size_um, positions_um, rates)
    assert np.allclose(ours, theirs, rtol=1e-12, atol=0.0)


def test_points_outside_are_dropped_and_ids_follow_fipy_order():
    mesh = wf.Grid2D(dx=1.0, dy=2.0, nx=3, ny=2, device="cpu") + (10.0, 0.0)
    positions = np.array([[10.5, 0.5], [12.9, 3.9], [9.9, 1.0], [13.0, 1.0], [11.2, 2.0]])
    assert wf.cell_ids(mesh, positions).tolist() == [0, 5, -1, -1, 4]
    field = wf.bin_rates(mesh, positions, [1.0, 2.0, 3.0, 4.0, 5.0])
    assert field.tolist() == [0.5, 0.0, 0.0, 0.0, 2.5, 1.0]  # rate / cell area (2)


def test_bin_rates_is_deterministic(rng):
    mesh = wf.Grid3D(dx=1.0, dy=1.0, dz=1.0, nx=8, ny=8, nz=8, device="cpu")
    positions = rng.random((10000, 3)) * 8.0
    rates = rng.normal(size=10000)
    assert np.array_equal(wf.bin_rates(mesh, positions, rates), wf.bin_rates(mesh, positions, rates))
    with pytest.raises(ValueError):
        wf.bin_rates(mesh, positions[:, :2], rates)
