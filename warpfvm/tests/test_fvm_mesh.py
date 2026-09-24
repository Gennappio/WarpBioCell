"""Mesh geometry and numbering: FiPy's cell and face order, masks, translation (unit + FiPy parity)."""

import numpy as np
import pytest

import warpfvm as wf
from fvm_helpers import grid

CASES = [
    ((5,), (0.5,)),
    ((4, 3), (1.0, 2.0)),
    ((3, 1), (1.0, 1.0)),
    ((2, 3, 4), (1.0, 2.0, 3.0)),
    ((1, 1, 1), (2.0, 2.0, 2.0)),
    ((4, 1, 3), (0.1, 0.2, 0.3)),
]


@pytest.mark.parametrize("dims, spacing", CASES)
def test_counts_and_volumes(dims, spacing):
    mesh = grid(wf, dims, spacing, device="cpu")
    n = int(np.prod(dims))
    padded = tuple(dims) + (1,) * (3 - len(dims))
    nx, ny, nz = padded
    faces = {1: nx + 1, 2: nx * (ny + 1) + (nx + 1) * ny, 3: nx * ny * (nz + 1) + nx * (ny + 1) * nz + (nx + 1) * ny * nz}
    assert mesh.shape == tuple(dims)
    assert mesh.numberOfCells == n
    assert mesh.numberOfFaces == faces[len(dims)]
    assert np.allclose(mesh.cellVolumes, np.prod(spacing))
    assert mesh.cellCenters.shape == (len(dims), n)
    assert mesh.faceCenters.shape == (len(dims), mesh.numberOfFaces)
    assert mesh.exteriorFaces.sum() == mesh._n_exterior
    assert np.array_equal(mesh.exteriorFaces, ~mesh.interiorFaces)


def test_cells_are_x_fastest():
    mesh = wf.Grid3D(dx=1.0, dy=2.0, dz=3.0, nx=2, ny=3, nz=4, device="cpu")
    centers = mesh.cellCenters
    ids = np.arange(mesh.numberOfCells)
    i, j, k = ids % 2, (ids // 2) % 3, ids // 6
    assert np.allclose(centers[0], (i + 0.5) * 1.0)
    assert np.allclose(centers[1], (j + 0.5) * 2.0)
    assert np.allclose(centers[2], (k + 0.5) * 3.0)


def test_side_masks_select_the_boundary_planes():
    mesh = wf.Grid3D(dx=1.0, dy=1.0, dz=1.0, nx=3, ny=4, nz=5, device="cpu")
    fc = mesh.faceCenters
    assert np.array_equal(mesh.facesLeft, mesh.exteriorFaces & np.isclose(fc[0], 0.0))
    assert np.array_equal(mesh.facesRight, mesh.exteriorFaces & np.isclose(fc[0], 3.0))
    assert np.array_equal(mesh.facesBottom, mesh.exteriorFaces & np.isclose(fc[1], 0.0))
    assert np.array_equal(mesh.facesTop, mesh.exteriorFaces & np.isclose(fc[1], 4.0))
    assert np.array_equal(mesh.facesFront, mesh.exteriorFaces & np.isclose(fc[2], 0.0))
    assert np.array_equal(mesh.facesBack, mesh.exteriorFaces & np.isclose(fc[2], 5.0))
    assert np.array_equal(mesh.facesDown, mesh.facesBottom) and np.array_equal(mesh.facesUp, mesh.facesTop)
    sides = [mesh.facesLeft, mesh.facesRight, mesh.facesBottom, mesh.facesTop, mesh.facesFront, mesh.facesBack]
    assert np.array_equal(np.sum(sides, axis=0), mesh.exteriorFaces.astype(int))  # disjoint cover


def test_exterior_index_is_a_bijection():
    for dims in [(4,), (3, 5), (2, 3, 4)]:
        mesh = grid(wf, dims, (1.0,) * len(dims), device="cpu")
        index = mesh._exterior_index
        assert np.all(index[mesh.interiorFaces] == -1)
        assert np.array_equal(np.sort(index[mesh.exteriorFaces]), np.arange(mesh._n_exterior))


def test_length_arguments_follow_fipy():
    assert wf.Grid1D(dx=0.3, Lx=1.0, device="cpu").shape == (3,)
    assert wf.Grid1D(dx=0.3, Lx=1.0, device="cpu").dx == pytest.approx(1.0 / 3.0)
    mesh = wf.Grid3D(Lx=2.5, Ly=2.0, Lz=1.0, device="cpu")
    assert mesh.shape == (2, 2, 1) and mesh.dx == pytest.approx(1.25)
    assert wf.Grid3D(dx=0.3, Lx=1.0, nx=5, device="cpu").dx == pytest.approx(0.2)


def test_translation_moves_coordinates_only():
    mesh = wf.Grid2D(dx=1.0, dy=2.0, nx=3, ny=2, device="cpu")
    for offset in [(10.0, 20.0), [[10.0], [20.0]]]:
        moved = mesh + offset
        assert type(moved) is type(mesh)
        assert np.allclose(moved.cellCenters, mesh.cellCenters + np.array([[10.0], [20.0]]))
        assert np.allclose(moved.faceCenters, mesh.faceCenters + np.array([[10.0], [20.0]]))
        assert np.array_equal(moved.facesLeft, mesh.facesLeft)
        assert moved._same_cells(mesh)
    assert np.allclose(mesh.origin, 0.0)  # the original is untouched


def test_missing_axes_and_bad_inputs():
    mesh = wf.Grid2D(nx=2, ny=2, device="cpu")
    with pytest.raises(AttributeError):
        mesh.dz
    assert not mesh.facesFront.any() and not mesh.facesBack.any()
    with pytest.raises(NotImplementedError):
        wf.Grid1D(dx=[1.0, 2.0])
    with pytest.raises(ValueError):
        wf.Grid2D(dx=-1.0, nx=2, ny=2)
    with pytest.raises(ValueError):
        mesh + (1.0, 2.0, 3.0)


# --- parity with FiPy -------------------------------------------------------------------------


@pytest.mark.parametrize("dims, spacing", CASES)
def test_geometry_matches_fipy(dims, spacing):
    fp = pytest.importorskip("fipy")
    ours = grid(wf, dims, spacing, device="cpu")
    theirs = grid(fp, dims, spacing)
    assert ours.shape == tuple(theirs.shape)
    assert ours.numberOfCells == theirs.numberOfCells
    assert ours.numberOfFaces == theirs.numberOfFaces
    assert np.allclose(ours.cellCenters, np.asarray(theirs.cellCenters))
    assert np.allclose(ours.faceCenters, np.asarray(theirs.faceCenters))
    assert np.allclose(ours.cellVolumes, np.asarray(theirs.cellVolumes))
    assert np.allclose(ours._faceAreas, np.asarray(theirs._faceAreas))
    assert np.allclose(ours._cellDistances, np.asarray(theirs._cellDistances))
    names = ["exteriorFaces", "interiorFaces", "facesLeft", "facesRight"]
    if len(dims) >= 2:
        names += ["facesBottom", "facesTop"]
    if len(dims) == 3:
        names += ["facesFront", "facesBack"]
    for name in names:
        assert np.array_equal(getattr(ours, name), np.asarray(getattr(theirs, name), dtype=bool)), name


def test_translation_matches_fipy():
    fp = pytest.importorskip("fipy")
    offset = [[-1.5], [2.0], [0.25]]
    ours = wf.Grid3D(dx=1.0, dy=2.0, dz=3.0, nx=2, ny=3, nz=4, device="cpu") + offset
    theirs = fp.Grid3D(dx=1.0, dy=2.0, dz=3.0, nx=2, ny=3, nz=4) + offset
    assert np.allclose(ours.cellCenters, np.asarray(theirs.cellCenters))
    assert np.allclose(ours.faceCenters, np.asarray(theirs.faceCenters))
