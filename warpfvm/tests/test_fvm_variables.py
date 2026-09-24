"""CellVariable: values, Dirichlet constraints, old values, and the errors for what is not supported."""

import numpy as np
import pytest
import warp as wp

import warpfvm as wf


@pytest.fixture
def mesh():
    return wf.Grid2D(dx=1.0, dy=1.0, nx=4, ny=3, device="cpu")


def test_value_roundtrip_and_where(mesh):
    var = wf.CellVariable(mesh=mesh, value=2.0, name="c")
    assert var.value.shape == (12,) and np.all(var.value == 2.0)
    values = np.arange(12.0)
    var.setValue(values)
    assert np.array_equal(var.value, values)
    var.setValue(-1.0, where=values < 3)
    assert np.array_equal(var.value, np.where(values < 3, -1.0, values))
    var.value = 5.0
    assert np.all(np.asarray(var) == 5.0) and len(var) == 12 and var[3] == 5.0
    copy = var.value
    copy[:] = 0.0
    assert np.all(var.value == 5.0)  # value returns a copy


def test_value_from_device_array_and_precision(mesh):
    src = wp.array(np.linspace(0.0, 1.0, 12), dtype=wp.float64, device="cpu")
    var = wf.CellVariable(mesh=mesh, value=src)
    assert np.allclose(var.value, np.linspace(0.0, 1.0, 12))
    single = wf.Grid2D(nx=4, ny=3, device="cpu", dtype="float32")
    assert wf.CellVariable(mesh=single, value=1.0).value.dtype == np.float32
    with pytest.raises(ValueError):
        wf.CellVariable(mesh=mesh, value=np.zeros(5))


def test_later_constraint_wins_and_release(mesh):
    var = wf.CellVariable(mesh=mesh)
    first = var.constrain(1.0, mesh.facesLeft)
    var.constrain(5.0, mesh.facesLeft & (mesh.faceCenters[1] < 1.0))
    fixed, values = var._boundary_arrays()
    ext = mesh._exterior_index[mesh.facesLeft]
    assert np.all(fixed.numpy()[ext] == 1)
    assert sorted(values.numpy()[ext].tolist()) == [1.0, 1.0, 5.0]
    var.release(first)
    fixed, values = var._boundary_arrays()
    assert fixed.numpy()[ext].tolist().count(1) == 1


def test_per_face_constraint_values(mesh):
    var = wf.CellVariable(mesh=mesh)
    face_values = mesh.faceCenters[0] + 10.0 * mesh.faceCenters[1]
    var.constrain(face_values, mesh.exteriorFaces)
    fixed, values = var._boundary_arrays()
    ext = mesh._exterior_index[mesh.exteriorFaces]
    assert np.all(fixed.numpy() == 1)
    assert np.allclose(values.numpy()[ext], face_values[mesh.exteriorFaces])


def test_unsupported_constraints_raise(mesh):
    var = wf.CellVariable(mesh=mesh)
    with pytest.raises(NotImplementedError):
        var.constrain(1.0, mesh.interiorFaces)
    with pytest.raises(NotImplementedError):
        var.constrain(1.0, np.ones(mesh.numberOfCells, dtype=bool))
    with pytest.raises(ValueError):
        var.constrain(1.0)
    with pytest.raises(NotImplementedError):
        var.faceGrad.constrain([0.0, 1.0], mesh.facesLeft)
    with pytest.raises(NotImplementedError):
        wf.CellVariable(mesh=mesh, unit="mM")


def test_old_values(mesh):
    plain = wf.CellVariable(mesh=mesh, value=1.0)
    assert plain.old is plain
    tracked = wf.CellVariable(mesh=mesh, value=1.0, hasOld=True)
    tracked.setValue(3.0)
    assert np.all(tracked.old.value == 1.0)
    tracked.updateOld()
    assert np.all(tracked.old.value == 3.0)


def test_scaled_variables(mesh):
    var = wf.CellVariable(mesh=mesh, value=2.0)
    assert np.all((-var).value == -2.0)
    assert np.all((3 * var).value == 6.0) and np.all((var * 3).value == 6.0)
    assert np.all((-(var / 4)).value == -0.5)
    with pytest.raises(TypeError):
        var + 1.0
