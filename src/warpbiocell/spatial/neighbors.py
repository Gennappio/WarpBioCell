"""Neighbour search over the active cells of a population using ``wp.HashGrid``.

The grid is rebuilt from scratch before every query pass (positions change every substep).
``dim`` sets the number of hash buckets per axis, not the spatial extent: points anywhere in
space are hashed into ``dim**3`` buckets, so it only needs to be large enough to keep bucket
occupancy low. Bucket bookkeeping costs ``2 * 4 bytes * dim**3`` (16 MB at ``dim=128``).
"""

from __future__ import annotations

import warp as wp

from warpbiocell.cells.state import CellPopulation


class NeighborGrid:
    def __init__(self, cell_size: float, dim: int = 128, device: wp.context.Device | str | None = None):
        """``cell_size`` [um] is the bucket edge and should equal the query radius used in kernels."""
        if cell_size <= 0.0:
            raise ValueError("cell_size must be positive")
        self.cell_size = float(cell_size)
        self.device = wp.get_device(device)
        self._grid = wp.HashGrid(dim, dim, dim, device=self.device)

    @property
    def id(self) -> int:
        """Handle to pass to kernels as ``wp.uint64``."""
        return self._grid.id

    def reserve(self, num_points: int) -> None:
        self._grid.reserve(num_points)

    def build(self, population: CellPopulation) -> None:
        """Index the active prefix ``[0, count)`` only; query results are global slot indices."""
        if population.device != self.device:
            raise RuntimeError("population and NeighborGrid must live on the same device")
        self._grid.build(population.active_position, self.cell_size)
