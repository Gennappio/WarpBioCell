"""Regular 3-D scalar grid with per-axis boundary conditions.

Node ``(i, j, k)`` sits at ``origin + dx * (i, j, k)`` and represents the voxel of volume
``dx**3`` around it. On a DIRICHLET axis the two face planes hold ``boundary_value`` and are
never updated by solvers; on a NEUMANN axis the faces are zero-flux (mirror) planes, so a
symmetric problem can be solved on a fraction of the domain.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import warp as wp


class Boundary(IntEnum):
    DIRICHLET = 0  # fixed value on the face
    NEUMANN = 1  # zero flux across the face


@dataclass(frozen=True)
class GridGeometry:
    origin: tuple[float, float, float]  # um, position of node (0, 0, 0)
    dx: float  # um, node spacing (cubic voxels)
    shape: tuple[int, int, int]  # nodes per axis

    def __post_init__(self):
        if self.dx <= 0.0:
            raise ValueError("dx must be positive")
        if len(self.shape) != 3 or min(self.shape) < 3:
            raise ValueError("shape needs at least 3 nodes per axis")

    @classmethod
    def centered_cube(cls, side: float, dx: float) -> GridGeometry:
        """Cube of edge ``side`` [um] centred on the origin; ``side`` is rounded up to whole voxels."""
        n = int(np.ceil(side / dx)) + 1
        half = 0.5 * (n - 1) * dx
        return cls(origin=(-half, -half, -half), dx=dx, shape=(n, n, n))

    @property
    def node_count(self) -> int:
        return int(np.prod(self.shape))

    @property
    def voxel_volume(self) -> float:
        return self.dx**3

    @property
    def upper(self) -> tuple[float, float, float]:
        return tuple(self.origin[a] + self.dx * (self.shape[a] - 1) for a in range(3))

    def node_coordinates(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Node coordinate arrays along each axis [um]."""
        return tuple(self.origin[a] + self.dx * np.arange(self.shape[a]) for a in range(3))

    def contains(self, positions: np.ndarray) -> np.ndarray:
        """Boolean mask of positions inside the trilinear support of the grid."""
        p = np.asarray(positions)
        lo = np.asarray(self.origin)
        hi = np.asarray(self.upper)
        return np.all((p >= lo) & (p <= hi), axis=1)


@dataclass
class ScalarField:
    geometry: GridGeometry
    boundary: tuple[Boundary, Boundary, Boundary]
    boundary_value: float
    device: wp.context.Device
    values: wp.array  # array3d float32

    @classmethod
    def create(
        cls,
        geometry: GridGeometry,
        boundary_value: float,
        boundary: tuple[Boundary, Boundary, Boundary] = (Boundary.DIRICHLET,) * 3,
        initial_value: float | None = None,
        device: wp.context.Device | str | None = None,
    ) -> ScalarField:
        device = wp.get_device(device)
        fill = boundary_value if initial_value is None else initial_value
        values = wp.full(geometry.shape, float(fill), dtype=wp.float32, device=device)
        field = cls(geometry=geometry, boundary=tuple(Boundary(b) for b in boundary), boundary_value=float(boundary_value), device=device, values=values)
        field.apply_dirichlet()
        return field

    @property
    def boundary_flags(self) -> tuple[int, int, int]:
        return tuple(int(b) for b in self.boundary)

    def numpy(self) -> np.ndarray:
        return self.values.numpy().copy()

    def assign(self, array: np.ndarray) -> None:
        """Overwrite the values (Dirichlet faces are re-imposed afterwards)."""
        array = np.ascontiguousarray(array, dtype=np.float32)
        if array.shape != tuple(self.geometry.shape):
            raise ValueError(f"expected shape {self.geometry.shape}, got {array.shape}")
        self.values.assign(array)
        self.apply_dirichlet()

    def apply_dirichlet(self) -> None:
        """Re-impose ``boundary_value`` on every Dirichlet face (host round trip; not per step)."""
        if all(b == Boundary.NEUMANN for b in self.boundary):
            return
        a = self.values.numpy()
        for axis, b in enumerate(self.boundary):
            if b == Boundary.DIRICHLET:
                idx = [slice(None)] * 3
                idx[axis] = 0
                a[tuple(idx)] = self.boundary_value
                idx[axis] = -1
                a[tuple(idx)] = self.boundary_value
        self.values.assign(a)

    def interior_mask(self) -> np.ndarray:
        """Mask of nodes that solvers update (everything except Dirichlet faces)."""
        mask = np.ones(self.geometry.shape, dtype=bool)
        for axis, b in enumerate(self.boundary):
            if b == Boundary.DIRICHLET:
                idx = [slice(None)] * 3
                idx[axis] = 0
                mask[tuple(idx)] = False
                idx[axis] = -1
                mask[tuple(idx)] = False
        return mask
