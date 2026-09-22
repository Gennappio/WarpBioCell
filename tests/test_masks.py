"""Tissue regions from voxel masks (optional scipy / nibabel)."""

import numpy as np
import pytest

from warpbiocell.fields.scalar_field import GridGeometry
from warpbiocell.geometry.shapes import Sphere

scipy = pytest.importorskip("scipy")
from warpbiocell.geometry.masks import load_nifti_mask, region_from_mask, sdf_from_mask  # noqa: E402


def _voxel_sphere(n=40, voxel=10.0, radius=150.0):
    coords = (np.arange(n) - (n - 1) / 2.0) * voxel
    x, y, z = np.meshgrid(coords, coords, coords, indexing="ij")
    return np.sqrt(x**2 + y**2 + z**2) < radius, coords[0]


def test_sdf_from_mask_matches_the_analytical_distance_to_within_a_voxel():
    mask, origin = _voxel_sphere()
    sdf = sdf_from_mask(mask, 10.0)
    coords = origin + 10.0 * np.arange(mask.shape[0])
    x, y, z = np.meshgrid(coords, coords, coords, indexing="ij")
    exact = np.sqrt(x**2 + y**2 + z**2) - 150.0
    assert np.all((sdf < 0) == mask)
    assert np.abs(sdf - exact).max() < 1.5 * 10.0  # voxelisation error only
    assert np.abs(sdf - exact)[np.abs(exact) < 50].mean() < 5.0
    with pytest.raises(ValueError):
        sdf_from_mask(np.zeros((4, 4, 4), dtype=bool), 10.0)


def test_region_from_mask_resamples_onto_the_simulation_grid(cpu):
    mask, origin = _voxel_sphere()
    geom = GridGeometry.centered_cube(400.0, 20.0)
    region = region_from_mask(mask, 10.0, geom, mask_origin=(origin, origin, origin), device=cpu)

    assert abs(region.volume() / Sphere(radius=150.0).volume - 1.0) < 0.1
    pts = np.array([[0.0, 0.0, 0.0], [140.0, 0.0, 0.0], [160.0, 0.0, 0.0], [0.0, 195.0, 0.0]])
    inside = region.contains(pts)
    np.testing.assert_array_equal(inside, [True, True, False, False])


def test_nifti_roundtrip(tmp_path, cpu):
    nib = pytest.importorskip("nibabel")
    mask, origin = _voxel_sphere(n=30, voxel=20.0, radius=120.0)
    labels = np.where(mask, 2, 0).astype(np.int16)
    affine = np.diag([0.02, 0.02, 0.02, 1.0])  # 20 um voxels expressed in mm
    path = tmp_path / "tumour.nii.gz"
    nib.save(nib.Nifti1Image(labels, affine), str(path))

    loaded, voxel = load_nifti_mask(path, label=2)

    np.testing.assert_array_equal(loaded, mask)
    np.testing.assert_allclose(voxel, (20.0, 20.0, 20.0), rtol=1e-5)
    geom = GridGeometry.centered_cube(400.0, 20.0)
    region = region_from_mask(loaded, voxel, geom, mask_origin=(origin,) * 3, device=cpu)
    assert abs(region.volume() / Sphere(radius=120.0).volume - 1.0) < 0.15
