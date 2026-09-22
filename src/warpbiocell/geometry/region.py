"""A tissue region: a signed-distance field on a regular grid, negative inside.

Built from a synthetic shape (``shapes.py``) or from a segmentation mask (``masks.py``). Used
for seeding cells (``seeding.py``), for confining them (``kernels/geometry_kernels.py``) and
for fixing the oxygen boundary on the tissue surface (``fields/oxygen.py``).
"""

from __future__ import annotations

import numpy as np
import warp as wp

from warpbiocell.fields.scalar_field import GridGeometry
from warpbiocell.geometry.shapes import evaluate_on_grid
from warpbiocell.kernels.geometry_kernels import sample_sdf


def node_gradient(values: np.ndarray, dx: float) -> np.ndarray:
    """Central differences inside, one-sided at the faces; shape (nx, ny, nz, 3)."""
    grads = np.gradient(values.astype(np.float64), dx, edge_order=1)
    return np.stack(grads, axis=-1).astype(np.float32)


class TissueRegion:
    def __init__(self, geometry: GridGeometry, sdf: np.ndarray, device: wp.context.Device | str | None = None, name: str = "region"):
        sdf = np.ascontiguousarray(sdf, dtype=np.float32)
        if sdf.shape != tuple(geometry.shape):
            raise ValueError(f"sdf shape {sdf.shape} does not match the grid {geometry.shape}")
        self.geometry = geometry
        self.name = name
        self.device = wp.get_device(device)
        self._sdf_np = sdf
        self.sdf = wp.array(sdf, dtype=wp.float32, device=self.device)
        self.gradient = wp.array(node_gradient(sdf, geometry.dx), dtype=wp.vec3, device=self.device)

    @classmethod
    def from_shape(cls, shape, geometry: GridGeometry, device=None) -> TissueRegion:
        return cls(geometry, evaluate_on_grid(shape, geometry), device=device, name=type(shape).__name__.lower())

    # ---- host-side queries -------------------------------------------------------------------

    def sdf_numpy(self) -> np.ndarray:
        return self._sdf_np.copy()

    def inside_mask(self) -> np.ndarray:
        """Nodes strictly inside the tissue."""
        return self._sdf_np < 0.0

    def volume(self) -> float:
        """Tissue volume [um^3] counted in whole voxels."""
        return float(self.inside_mask().sum()) * self.geometry.voxel_volume

    def sample(self, positions: np.ndarray) -> np.ndarray:
        """Trilinear SDF at arbitrary points [um] (host, numpy); points outside the grid are clamped."""
        geom = self.geometry
        u = (np.asarray(positions, dtype=np.float64) - np.asarray(geom.origin)) / geom.dx
        u = np.clip(u, 0.0, np.array(geom.shape) - 1.0)
        base = np.minimum(np.floor(u).astype(int), np.array(geom.shape) - 2)
        f = u - base
        v = self._sdf_np.astype(np.float64)
        i, j, k = base[:, 0], base[:, 1], base[:, 2]
        fx, fy, fz = f[:, 0], f[:, 1], f[:, 2]
        c00 = v[i, j, k] * (1 - fx) + v[i + 1, j, k] * fx
        c10 = v[i, j + 1, k] * (1 - fx) + v[i + 1, j + 1, k] * fx
        c01 = v[i, j, k + 1] * (1 - fx) + v[i + 1, j, k + 1] * fx
        c11 = v[i, j + 1, k + 1] * (1 - fx) + v[i + 1, j + 1, k + 1] * fx
        return (c00 * (1 - fy) + c10 * fy) * (1 - fz) + (c01 * (1 - fy) + c11 * fy) * fz

    def contains(self, positions: np.ndarray, margin: float = 0.0) -> np.ndarray:
        """True where a point lies at least ``margin`` [um] inside the surface."""
        return self.sample(positions) < -margin

    # ---- device-side queries -----------------------------------------------------------------

    def sample_at(self, position: wp.array, count: int, out: wp.array) -> None:
        geom = self.geometry
        if count > 0:
            wp.launch(sample_sdf, dim=count, inputs=[self.sdf, position, wp.vec3(*geom.origin), 1.0 / geom.dx, out], device=self.device)
