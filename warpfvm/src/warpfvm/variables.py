"""CellVariable: one value per cell, kept in a Warp array on the mesh's device.

What FiPy's CellVariable offers and warpfvm keeps: ``value`` / ``setValue``, ``constrain`` on
boundary faces (Dirichlet; later constraints win on overlapping faces, as in FiPy), ``release``,
``hasOld`` / ``old`` / ``updateOld``, unary minus and scaling by a number (so ``-source`` and
``2 * source`` can enter an equation as explicit sources).

What it does not keep, deliberately: FiPy's lazy variable algebra (``var + 1`` is an error, not
a new variable), constraints on interior faces or on cell values, fixed-gradient constraints
(``var.faceGrad.constrain``; unconstrained boundary faces are zero-flux, as in FiPy), physical
units and vector-valued variables.
"""

from __future__ import annotations

import numbers
from dataclasses import dataclass, field

import numpy as np
import warp as wp

from warpfvm._config import numpy_dtype
from warpfvm.mesh import UniformGrid


@dataclass(eq=False)
class Constraint:
    """A Dirichlet constraint: ``values`` (one per face) on the faces where ``mask`` is true."""

    values: np.ndarray
    mask: np.ndarray = field(repr=False)


class _FaceGradient:
    def __init__(self, var):
        self._var = var

    def constrain(self, value, where=None):
        raise NotImplementedError(
            "fixed-gradient (Neumann) constraints are not supported by warpfvm; "
            "unconstrained boundary faces are already zero-flux, as in FiPy"
        )

    @property
    def value(self):
        raise NotImplementedError("face gradients are not computed by warpfvm")


class CellVariable:
    """A scalar field on the cells of a uniform grid.

    ``value`` returns a host copy (numpy); the solver works in place on the device array, so
    a solve never moves the field between host and device.
    """

    def __init__(self, mesh, value=0.0, name="", hasOld=False, unit=None, rank=None, elementshape=None):
        if not isinstance(mesh, UniformGrid):
            raise TypeError(f"mesh must be a warpfvm grid, got {type(mesh).__name__}")
        if unit is not None:
            raise NotImplementedError("physical units are not supported")
        if rank not in (None, 0) or elementshape not in (None, ()):
            raise NotImplementedError("only scalar (rank-0) variables are supported")
        self.mesh = mesh
        self.name = name
        self._array = wp.empty(mesh.numberOfCells, dtype=mesh.dtype, device=mesh.device)
        self._assign(value)
        self._constraints: list[Constraint] = []
        self._bc_version = 0
        self._bc_cache = None
        self._old = wp.clone(self._array) if hasOld else None

    # --- values -------------------------------------------------------------------------------

    @property
    def value(self) -> np.ndarray:
        values = self._array.numpy()
        return values.copy() if self._array.device.is_cpu else values

    @value.setter
    def value(self, value):
        self._assign(value)

    def setValue(self, value, unit=None, where=None):
        if unit is not None:
            raise NotImplementedError("physical units are not supported")
        self._assign(value, where)

    @property
    def globalValue(self) -> np.ndarray:
        return self.value

    @property
    def numericValue(self) -> np.ndarray:
        return self.value

    @property
    def device_array(self) -> wp.array:
        """The Warp array the solvers read and write (no copy)."""
        return self._array

    def _assign(self, value, where=None):
        n = self.mesh.numberOfCells
        if isinstance(value, wp.array) and where is None:
            if value.shape != (n,):
                raise ValueError(f"value has shape {value.shape}, the mesh has {n} cells")
            if value.device == self._array.device and value.dtype == self._array.dtype:
                wp.copy(self._array, value)
                return
            value = value.numpy()
        if isinstance(value, (CellVariable, _ScaledVariable)):
            value = value.value
        host = np.asarray(value, dtype=np.float64)
        if host.ndim == 0:
            host = np.full(n, float(host))
        else:
            host = host.reshape(-1)
            if host.size != n:
                raise ValueError(f"value has {host.size} entries, the mesh has {n} cells")
        if where is not None:
            mask = np.asarray(where, dtype=bool).reshape(-1)
            if mask.size != n:
                raise ValueError(f"where has {mask.size} entries, the mesh has {n} cells")
            current = np.asarray(self.value, dtype=np.float64)
            current[mask] = host[mask]
            host = current
        self._array.assign(np.ascontiguousarray(host, dtype=numpy_dtype(self.mesh.dtype)))

    # --- boundary constraints -----------------------------------------------------------------

    def constrain(self, value, where=None) -> Constraint:
        """Fix the value on boundary faces (FiPy's ``var.constrain(value, where=faces)``).

        ``value`` is a number or one value per face; ``where`` is a boolean mask over the faces
        (``mesh.facesLeft | mesh.facesTop``, or a mask built from ``mesh.faceCenters``).
        """
        mesh = self.mesh
        n_faces = mesh.numberOfFaces
        if where is None:
            raise ValueError("constrain() needs `where`, a boolean mask over the mesh faces")
        mask = np.asarray(where, dtype=bool).reshape(-1)
        if mask.size != n_faces:
            if mask.size == mesh.numberOfCells:
                raise NotImplementedError("constraints on cell values are not supported; constrain boundary faces")
            raise ValueError(f"where has {mask.size} entries, the mesh has {n_faces} faces")
        interior = mask & mesh.interiorFaces
        if interior.any():
            raise NotImplementedError(f"{int(interior.sum())} interior faces in `where`: only boundary faces can be constrained")
        values = np.asarray(value, dtype=np.float64)
        if values.ndim == 0:
            values = np.full(n_faces, float(values))
        else:
            values = values.reshape(-1)
            if values.size != n_faces:
                raise ValueError(f"a constraint array needs one value per face ({n_faces}), got {values.size}")
            values = values.copy()
        constraint = Constraint(values, mask.copy())
        self._constraints.append(constraint)
        self._bc_version += 1
        return constraint

    def release(self, constraint: Constraint) -> None:
        self._constraints.remove(constraint)
        self._bc_version += 1

    @property
    def constraints(self) -> tuple:
        return tuple(self._constraints)

    @property
    def faceGrad(self) -> _FaceGradient:
        return _FaceGradient(self)

    def _boundary_arrays(self):
        """(fixed flag int32, value) over the compact exterior faces, uploaded once per change."""
        if self._bc_cache is not None and self._bc_cache[0] == self._bc_version:
            return self._bc_cache[1], self._bc_cache[2]
        mesh = self.mesh
        exterior = mesh._exterior_index
        fixed = np.zeros(mesh._n_exterior, dtype=np.int32)
        values = np.zeros(mesh._n_exterior, dtype=np.float64)
        for constraint in self._constraints:  # in order: a later constraint overrides an earlier one
            faces = np.nonzero(constraint.mask)[0]
            fixed[exterior[faces]] = 1
            values[exterior[faces]] = constraint.values[faces]
        fixed_d = wp.array(fixed, dtype=wp.int32, device=mesh.device)
        values_d = wp.array(values.astype(numpy_dtype(mesh.dtype)), dtype=mesh.dtype, device=mesh.device)
        self._bc_cache = (self._bc_version, fixed_d, values_d)
        return fixed_d, values_d

    # --- old values for transient terms -------------------------------------------------------

    @property
    def old(self) -> CellVariable:
        """The values at the start of the step: a separate copy with ``hasOld``, else ``self``."""
        if self._old is None:
            return self
        view = CellVariable.__new__(CellVariable)
        view.mesh, view.name, view._array = self.mesh, f"{self.name}_old", self._old
        view._constraints, view._bc_version, view._bc_cache, view._old = [], 0, None, None
        return view

    def updateOld(self) -> None:
        if self._old is not None:
            wp.copy(self._old, self._array)

    # --- small conveniences -------------------------------------------------------------------

    def copy(self) -> CellVariable:
        other = CellVariable(self.mesh, self._array, name=self.name, hasOld=self._old is not None)
        if self._old is not None:
            wp.copy(other._old, self._old)
        for constraint in self._constraints:
            other.constrain(constraint.values, constraint.mask)
        return other

    @property
    def shape(self) -> tuple:
        return (self.mesh.numberOfCells,)

    @property
    def rank(self) -> int:
        return 0

    def __len__(self) -> int:
        return self.mesh.numberOfCells

    def __getitem__(self, index):
        return self.value[index]

    def __array__(self, dtype=None, copy=None):
        values = self.value
        return values if dtype is None else values.astype(dtype)

    def __neg__(self):
        return _ScaledVariable(self, -1.0)

    def __pos__(self):
        return self

    def __mul__(self, other):
        if isinstance(other, numbers.Real):
            return _ScaledVariable(self, float(other))
        return NotImplemented

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, numbers.Real):
            return _ScaledVariable(self, 1.0 / float(other))
        return NotImplemented

    def __repr__(self) -> str:
        return f"CellVariable(name={self.name!r}, mesh={self.mesh!r})"


class _ScaledVariable:
    """``scale * var`` as an equation coefficient or explicit source (read at assembly time)."""

    def __init__(self, var: CellVariable, scale: float):
        self.var = var
        self.scale = scale

    @property
    def mesh(self):
        return self.var.mesh

    @property
    def value(self) -> np.ndarray:
        return self.scale * self.var.value

    def __neg__(self):
        return _ScaledVariable(self.var, -self.scale)

    def __pos__(self):
        return self

    def __mul__(self, other):
        if isinstance(other, numbers.Real):
            return _ScaledVariable(self.var, self.scale * float(other))
        return NotImplemented

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, numbers.Real):
            return _ScaledVariable(self.var, self.scale / float(other))
        return NotImplemented

    def __array__(self, dtype=None, copy=None):
        values = self.value
        return values if dtype is None else values.astype(dtype)

    def __repr__(self) -> str:
        return f"{self.scale:g} * {self.var!r}"
