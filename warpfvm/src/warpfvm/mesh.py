"""Uniform Cartesian grids with FiPy's cell and face numbering.

Cells are numbered x-fastest, ``id = i + nx*(j + ny*k)``, exactly like FiPy, so a field's flat
array reshapes to ``(nz, ny, nx)`` in C order. Faces follow FiPy's order too, so a mask built
from ``faceCenters`` or combined with ``|`` and ``&`` selects the same faces in both libraries:

    3-D   XY faces (normal z)   i + nx*j + nx*ny*k          k = 0..nz, then
          XZ faces (normal y)   i + nx*j + nx*(ny+1)*k      j = 0..ny, then
          YZ faces (normal x)   i + (nx+1)*j + (nx+1)*ny*k  i = 0..nx
    2-D   faces normal to y     i + nx*j                    j = 0..ny, then
          faces normal to x     i + (nx+1)*j                i = 0..nx
    1-D   face i at x = i*dx

The geometry follows FiPy as well: a 2-D grid is one unit thick (face "areas" are lengths and
cell "volumes" are areas) and a 1-D grid has a unit cross-section. Masks, cell centres and face
centres are numpy arrays on the host; field values live on the mesh's Warp device.
"""

from __future__ import annotations

import copy

import numpy as np

from warpfvm._config import resolve_device, resolve_scalar

# Boundary sides, in the order of the compact exterior-face arrays used by the kernels.
XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX = range(6)


def _dnl(d, n, length):
    """FiPy's rule (meshes/factoryMeshes.py): a given length fixes the spacing to length / n."""
    if np.ndim(d) != 0:
        raise NotImplementedError("non-uniform spacing is not supported: warpfvm meshes are uniform grids")
    if length is None:
        if n is None:
            n = 1
    else:
        if n is None:
            n = length // d or 1
        d = length / int(n)
    return float(d), int(n)


def _as_offset(value, dim):
    offset = np.asarray(value, dtype=np.float64).reshape(-1)
    if offset.size != dim:
        raise ValueError(f"a {dim}-D mesh needs a {dim}-component offset, got {offset.size}")
    return offset


class UniformGrid:
    """A uniform grid of ``nx * ny * nz`` cells (``ny = nz = 1`` below three dimensions)."""

    def __init__(self, dim, shape, spacing, origin=None, device=None, dtype=None):
        if dim not in (1, 2, 3):
            raise ValueError("dim must be 1, 2 or 3")
        shape = tuple(int(n) for n in shape)
        spacing = tuple(float(d) for d in spacing)
        if len(shape) != dim or len(spacing) != dim:
            raise ValueError(f"a {dim}-D grid needs {dim} cell counts and {dim} spacings")
        if min(shape) < 1:
            raise ValueError("every axis needs at least one cell")
        if not all(np.isfinite(d) and d > 0.0 for d in spacing):
            raise ValueError("spacings must be positive and finite")
        self.dim = dim
        self.nx, self.ny, self.nz = shape + (1,) * (3 - dim)
        self._spacing = spacing + (1.0,) * (3 - dim)
        self.origin = np.zeros(dim) if origin is None else _as_offset(origin, dim)
        self.device = resolve_device(device)
        self.dtype = resolve_scalar(dtype)
        self._topology = {}  # origin-independent caches, shared by translated copies

    # --- sizes and spacing -------------------------------------------------------------------

    @property
    def dx(self) -> float:
        return self._spacing[0]

    @property
    def dy(self) -> float:
        if self.dim < 2:
            raise AttributeError("a 1-D grid has no dy")
        return self._spacing[1]

    @property
    def dz(self) -> float:
        if self.dim < 3:
            raise AttributeError(f"a {self.dim}-D grid has no dz")
        return self._spacing[2]

    @property
    def shape(self) -> tuple:
        return (self.nx, self.ny, self.nz)[: self.dim]

    @property
    def numberOfCells(self) -> int:
        return self.nx * self.ny * self.nz

    @property
    def numberOfFaces(self) -> int:
        return sum(int(np.prod(shape)) for _, shape in self._families())

    @property
    def _cell_volume(self) -> float:
        dx, dy, dz = self._spacing
        return dx * dy * dz

    @property
    def _axis_areas(self) -> tuple:
        """Area of a face normal to x, y and z (a length in 2-D, 1 in 1-D)."""
        dx, dy, dz = self._spacing
        return (dy * dz, dx * dz, dx * dy)

    @property
    def cellVolumes(self) -> np.ndarray:
        return np.full(self.numberOfCells, self._cell_volume)

    # --- coordinates --------------------------------------------------------------------------

    @property
    def cellCenters(self) -> np.ndarray:
        """``(dim, numberOfCells)`` cell-centre coordinates."""
        k, j, i = np.indices((self.nz, self.ny, self.nx)).reshape(3, -1)
        dx, dy, dz = self._spacing
        centers = np.stack([(i + 0.5) * dx, (j + 0.5) * dy, (k + 0.5) * dz])
        return centers[: self.dim] + self.origin[:, None]

    @property
    def x(self) -> np.ndarray:
        return self.cellCenters[0]

    @property
    def y(self) -> np.ndarray:
        return self.cellCenters[1]

    @property
    def z(self) -> np.ndarray:
        return self.cellCenters[2]

    @property
    def faceCenters(self) -> np.ndarray:
        """``(dim, numberOfFaces)`` face-centre coordinates, in FiPy's face order."""
        ijk, axis = self._face_ijk
        spacing = np.asarray(self._spacing)[:, None]
        centers = (ijk + 0.5) * spacing
        for a in range(3):
            normal = axis == a
            centers[a, normal] = ijk[a, normal] * self._spacing[a]
        return centers[: self.dim] + self.origin[:, None]

    # --- face masks (numpy booleans over the faces, FiPy's names) -----------------------------

    @property
    def exteriorFaces(self) -> np.ndarray:
        return self._face_side >= 0

    @property
    def interiorFaces(self) -> np.ndarray:
        return self._face_side < 0

    @property
    def facesLeft(self) -> np.ndarray:
        return self._face_side == XMIN

    @property
    def facesRight(self) -> np.ndarray:
        return self._face_side == XMAX

    @property
    def facesBottom(self) -> np.ndarray:
        return self._face_side == YMIN

    @property
    def facesTop(self) -> np.ndarray:
        return self._face_side == YMAX

    @property
    def facesFront(self) -> np.ndarray:
        return self._face_side == ZMIN

    @property
    def facesBack(self) -> np.ndarray:
        return self._face_side == ZMAX

    facesDown = facesBottom
    facesUp = facesTop

    @property
    def _faceAreas(self) -> np.ndarray:
        _, axis = self._face_ijk
        return np.asarray(self._axis_areas)[axis]

    @property
    def _cellDistances(self) -> np.ndarray:
        """Centre-to-centre distance across a face; centre-to-face on the boundary (FiPy's)."""
        _, axis = self._face_ijk
        distance = np.asarray(self._spacing)[axis]
        return np.where(self.exteriorFaces, 0.5 * distance, distance)

    # --- translation --------------------------------------------------------------------------

    def __add__(self, offset):
        """A copy moved by ``offset`` (``(dim,)`` or FiPy's ``(dim, 1)``); same cells and faces."""
        moved = copy.copy(self)
        moved.origin = self.origin + _as_offset(offset, self.dim)
        return moved

    __radd__ = __add__

    def __sub__(self, offset):
        return self + (-_as_offset(offset, self.dim))

    def _same_cells(self, other) -> bool:
        return (
            isinstance(other, UniformGrid)
            and other.dim == self.dim
            and other.shape == self.shape
            and np.allclose(other._spacing, self._spacing, rtol=1e-12, atol=0.0)
            and other.device == self.device
            and other.dtype == self.dtype
        )

    def __repr__(self) -> str:
        sizes = ", ".join(f"n{a}={n}" for a, n in zip("xyz", self.shape))
        spacing = ", ".join(f"d{a}={d:g}" for a, d in zip("xyz", self._spacing[: self.dim]))
        precision = "float64" if self.dtype.__name__ == "float64" else "float32"
        return f"{type(self).__name__}({sizes}, {spacing}, device={self.device.alias!r}, precision={precision!r})"

    # --- topology (cached, origin-independent) -------------------------------------------------

    def _families(self):
        """(normal axis, index-grid shape (k, j, i)) of each face family, in FiPy's order."""
        nx, ny, nz = self.nx, self.ny, self.nz
        if self.dim == 3:
            return [(2, (nz + 1, ny, nx)), (1, (nz, ny + 1, nx)), (0, (nz, ny, nx + 1))]
        if self.dim == 2:
            return [(1, (1, ny + 1, nx)), (0, (1, ny, nx + 1))]
        return [(0, (1, 1, nx + 1))]

    @property
    def _face_ijk(self):
        """Per face: its (i, j, k) index triple ``(3, F)`` and its normal axis ``(F,)``."""
        if "face_ijk" not in self._topology:
            triples, axes = [], []
            for axis, shape in self._families():
                k, j, i = np.indices(shape).reshape(3, -1)
                triples.append(np.stack([i, j, k]))
                axes.append(np.full(i.size, axis, dtype=np.int64))
            self._topology["face_ijk"] = (np.concatenate(triples, axis=1), np.concatenate(axes))
        return self._topology["face_ijk"]

    @property
    def _side_sizes(self) -> tuple:
        nx, ny, nz = self.nx, self.ny, self.nz
        x_side = ny * nz
        y_side = nx * nz if self.dim >= 2 else 0
        z_side = nx * ny if self.dim == 3 else 0
        return (x_side, x_side, y_side, y_side, z_side, z_side)

    @property
    def _side_offsets(self) -> tuple:
        sizes = self._side_sizes
        return tuple(int(sum(sizes[:s])) for s in range(6))

    @property
    def _n_exterior(self) -> int:
        return int(sum(self._side_sizes))

    @property
    def _face_side(self) -> np.ndarray:
        """Boundary side of every face (XMIN..ZMAX), -1 for interior faces."""
        self._build_exterior()
        return self._topology["face_side"]

    @property
    def _exterior_index(self) -> np.ndarray:
        """Position of every face in the compact exterior arrays the kernels read, -1 inside.

        Side s holds its faces at ``offset[s] + local``, with ``local = j + ny*k`` on the x
        sides, ``i + nx*k`` on the y sides and ``i + nx*j`` on the z sides.
        """
        self._build_exterior()
        return self._topology["exterior_index"]

    def _build_exterior(self):
        if "face_side" in self._topology:
            return
        ijk, axis = self._face_ijk
        counts = np.array([self.nx, self.ny, self.nz])
        along = ijk[axis, np.arange(axis.size)]
        side = np.full(axis.size, -1, dtype=np.int64)
        side[along == 0] = 2 * axis[along == 0]
        side[along == counts[axis]] = 2 * axis[along == counts[axis]] + 1
        i, j, k = ijk
        local = np.where(axis == 0, j + self.ny * k, np.where(axis == 1, i + self.nx * k, i + self.nx * j))
        offsets = np.asarray(self._side_offsets)
        self._topology["face_side"] = side
        self._topology["exterior_index"] = np.where(side >= 0, offsets[np.maximum(side, 0)] + local, -1)


class Grid1D(UniformGrid):
    """FiPy's ``Grid1D(dx, nx, Lx)``; ``overlap`` and ``communicator`` are accepted and ignored."""

    def __init__(self, dx=1.0, nx=None, Lx=None, overlap=2, communicator=None, *, device=None, dtype=None):
        dx, nx = _dnl(dx, nx, Lx)
        super().__init__(1, (nx,), (dx,), device=device, dtype=dtype)


class Grid2D(UniformGrid):
    """FiPy's ``Grid2D(dx, dy, nx, ny, Lx, Ly)``."""

    def __init__(self, dx=1.0, dy=1.0, nx=None, ny=None, Lx=None, Ly=None, overlap=2, communicator=None, *, device=None, dtype=None):
        dx, nx = _dnl(dx, nx, Lx)
        dy, ny = _dnl(dy, ny, Ly)
        super().__init__(2, (nx, ny), (dx, dy), device=device, dtype=dtype)


class Grid3D(UniformGrid):
    """FiPy's ``Grid3D(dx, dy, dz, nx, ny, nz, Lx, Ly, Lz)``."""

    def __init__(
        self, dx=1.0, dy=1.0, dz=1.0, nx=None, ny=None, nz=None, Lx=None, Ly=None, Lz=None,
        overlap=2, communicator=None, *, device=None, dtype=None,
    ):
        dx, nx = _dnl(dx, nx, Lx)
        dy, ny = _dnl(dy, ny, Ly)
        dz, nz = _dnl(dz, nz, Lz)
        super().__init__(3, (nx, ny, nz), (dx, dy, dz), device=device, dtype=dtype)
