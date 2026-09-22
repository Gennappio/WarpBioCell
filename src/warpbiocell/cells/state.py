"""Structure-of-arrays cell storage with preallocated capacity.

Layout: slots ``[0, count)`` hold cells that occupy space (alive or dead); slots
``[count, capacity)`` are free. Division (Milestone 3) appends daughters at the end of the
prefix, so no array is ever reallocated inside the simulation loop and every kernel launches
over ``count`` threads only.

Only the fields the current milestone needs are allocated. ``cell_type``, ``cell_state``,
``age`` and ``oxygen_local`` arrive with the lifecycle and oxygen milestones.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp


@dataclass
class CellPopulation:
    capacity: int
    count: int
    device: wp.context.Device
    position: wp.array  # vec3, um
    radius: wp.array  # float32, um
    velocity: wp.array  # vec3, um/h; overdamped, recomputed from contacts each substep
    contact_count: wp.array  # int32, number of overlapping neighbours found in the last substep

    @classmethod
    def from_numpy(
        cls,
        positions: np.ndarray,
        radii: np.ndarray,
        capacity: int | None = None,
        device: wp.context.Device | str | None = None,
    ) -> CellPopulation:
        positions = np.ascontiguousarray(positions, dtype=np.float32)
        radii = np.ascontiguousarray(radii, dtype=np.float32)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError(f"positions must have shape (n, 3), got {positions.shape}")
        if radii.shape != (positions.shape[0],):
            raise ValueError(f"radii must have shape ({positions.shape[0]},), got {radii.shape}")

        count = positions.shape[0]
        capacity = count if capacity is None else capacity
        if capacity < count:
            raise ValueError(f"capacity {capacity} smaller than initial cell count {count}")

        device = wp.get_device(device)
        pos_buf = np.zeros((capacity, 3), dtype=np.float32)
        pos_buf[:count] = positions
        rad_buf = np.zeros(capacity, dtype=np.float32)
        rad_buf[:count] = radii

        return cls(
            capacity=capacity,
            count=count,
            device=device,
            position=wp.array(pos_buf, dtype=wp.vec3, device=device),
            radius=wp.array(rad_buf, dtype=wp.float32, device=device),
            velocity=wp.zeros(capacity, dtype=wp.vec3, device=device),
            contact_count=wp.zeros(capacity, dtype=wp.int32, device=device),
        )

    # Views over the active prefix. Slicing a contiguous 1-D Warp array yields a view, so no copy
    # happens and indices reported by HashGrid queries coincide with global slot indices.
    @property
    def active_position(self) -> wp.array:
        return self.position[: self.count]

    @property
    def active_radius(self) -> wp.array:
        return self.radius[: self.count]

    def positions_numpy(self) -> np.ndarray:
        return self.position.numpy()[: self.count].copy()

    def radii_numpy(self) -> np.ndarray:
        return self.radius.numpy()[: self.count].copy()

    def contact_counts_numpy(self) -> np.ndarray:
        return self.contact_count.numpy()[: self.count].copy()
