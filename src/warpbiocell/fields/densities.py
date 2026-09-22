"""Per-state cell number densities on the grid, deposited once per step and shared by all species."""

from __future__ import annotations

import numpy as np
import warp as wp

from warpbiocell.cells.model import NUM_STATES
from warpbiocell.cells.state import CellPopulation
from warpbiocell.fields.scalar_field import GridGeometry
from warpbiocell.kernels.field_kernels import combine_densities, deposit_trilinear, fixed_point_to_density


class CellDensities:
    """``density[k]`` [cells/um^3] of cells in state ``k`` (trilinear, int64 fixed-point deposit)."""

    def __init__(self, geometry: GridGeometry, device: wp.context.Device | str | None = None):
        if NUM_STATES != 4:
            raise RuntimeError("combine_densities assumes four cell states")
        self.geometry = geometry
        self.device = wp.get_device(device)
        shape = (NUM_STATES, *geometry.shape)
        self._accumulator = wp.zeros(shape, dtype=wp.int64, device=self.device)
        self.density = wp.zeros(shape, dtype=wp.float32, device=self.device)

    def deposit(self, population: CellPopulation) -> None:
        if population.device != self.device:
            raise RuntimeError("population and CellDensities must live on the same device")
        geom = self.geometry
        self._accumulator.zero_()
        if population.count > 0:
            wp.launch(
                deposit_trilinear,
                dim=population.count,
                inputs=[population.position, population.cell_state, wp.vec3(*geom.origin), 1.0 / geom.dx, self._accumulator],
                device=self.device,
            )
        wp.launch(
            fixed_point_to_density,
            dim=(NUM_STATES, *geom.shape),
            inputs=[self._accumulator, 1.0 / geom.voxel_volume, self.density],
            device=self.device,
        )

    def combine(self, weights, out: wp.array) -> None:
        """``out = sum_k weights[k] * density[k]`` (weights indexed by CellState)."""
        weights = [float(w) for w in weights]
        if len(weights) != NUM_STATES:
            raise ValueError(f"need {NUM_STATES} weights, got {len(weights)}")
        wp.launch(combine_densities, dim=self.geometry.shape, inputs=[wp.vec4(*weights), self.density, out], device=self.device)

    def numpy(self) -> np.ndarray:
        return self.density.numpy().copy()

    def total_numpy(self) -> np.ndarray:
        return self.density.numpy().sum(axis=0)
