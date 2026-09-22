"""Structure-of-arrays cell storage with preallocated capacity.

Layout: slots ``[0, count)`` hold cells that occupy space (alive or dead); slots
``[count, capacity)`` are free. Division appends daughters at the end of the prefix, so no
array is ever reallocated inside the simulation loop and every kernel launches over ``count``
threads only. Nothing is removed in the MVP (dead cells stay), so a cell's slot index is
stable for its whole life and doubles as its id; a separate id array must be added if
compaction is ever introduced.

Randomness: every cell owns a counter-based RNG stream ``rand_init(seed, slot)`` stored in
``rng_state`` and advanced only by that cell's own draws. Results therefore depend on
``(seed, initial cells)`` and never on thread scheduling or capacity.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import warp as wp

from warpbiocell.cells.model import CellState, CellType


@wp.kernel
def _init_rng(seed: int, rng_state: wp.array(dtype=wp.uint32)):
    i = wp.tid()
    rng_state[i] = wp.rand_init(seed, i)


@dataclass
class CellPopulation:
    capacity: int
    count: int
    seed: int
    device: wp.context.Device
    # Persistent per-cell fields
    position: wp.array  # vec3, um
    radius: wp.array  # float32, um
    cell_state: wp.array  # int32, CellState
    cell_type: wp.array  # int32, CellType
    age: wp.array  # float32, h since birth (or since initialization)
    oxygen_local: wp.array  # float32, mmHg sampled at the cell centre (0 until a field writes it)
    glucose_local: wp.array  # float32, mM sampled at the cell centre (0 until a field writes it)
    rng_state: wp.array  # uint32, per-cell RNG stream
    # Per-step outputs and scratch
    velocity: wp.array  # vec3, um/h; overdamped, recomputed from contacts each substep
    neighbor_count: wp.array  # int32, cells within the query radius in the last mechanics substep
    divide_flag: wp.array  # int32, 1 if the cell divides this step
    division_offset: wp.array  # int32, inclusive prefix sum of divide_flag
    fate_flags: wp.array  # int32, bits from the gene network's fate nodes (0 without a network)
    extra_local: dict = field(default_factory=dict)  # name -> float32 array for other sampled species

    @classmethod
    def from_numpy(
        cls,
        positions: np.ndarray,
        radii: np.ndarray,
        capacity: int | None = None,
        device: wp.context.Device | str | None = None,
        seed: int = 0,
        states: np.ndarray | None = None,
    ) -> CellPopulation:
        positions = np.ascontiguousarray(positions, dtype=np.float32)
        radii = np.ascontiguousarray(radii, dtype=np.float32)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError(f"positions must have shape (n, 3), got {positions.shape}")
        count = positions.shape[0]
        if radii.shape != (count,):
            raise ValueError(f"radii must have shape ({count},), got {radii.shape}")

        capacity = count if capacity is None else capacity
        if capacity < count:
            raise ValueError(f"capacity {capacity} smaller than initial cell count {count}")

        if states is None:
            states = np.full(count, int(CellState.PROLIFERATIVE), dtype=np.int32)
        states = np.ascontiguousarray(states, dtype=np.int32)
        if states.shape != (count,):
            raise ValueError(f"states must have shape ({count},), got {states.shape}")

        device = wp.get_device(device)

        def padded(values, dtype, fill=0):
            buf = np.full((capacity,) + values.shape[1:], fill, dtype=dtype)
            buf[:count] = values
            return buf

        pop = cls(
            capacity=capacity,
            count=count,
            seed=int(seed),
            device=device,
            position=wp.array(padded(positions, np.float32), dtype=wp.vec3, device=device),
            radius=wp.array(padded(radii, np.float32), dtype=wp.float32, device=device),
            cell_state=wp.array(padded(states, np.int32, fill=int(CellState.DEAD)), dtype=wp.int32, device=device),
            cell_type=wp.full(capacity, int(CellType.TUMOR), dtype=wp.int32, device=device),
            age=wp.zeros(capacity, dtype=wp.float32, device=device),
            oxygen_local=wp.zeros(capacity, dtype=wp.float32, device=device),
            glucose_local=wp.zeros(capacity, dtype=wp.float32, device=device),
            rng_state=wp.zeros(capacity, dtype=wp.uint32, device=device),
            velocity=wp.zeros(capacity, dtype=wp.vec3, device=device),
            neighbor_count=wp.zeros(capacity, dtype=wp.int32, device=device),
            divide_flag=wp.zeros(capacity, dtype=wp.int32, device=device),
            division_offset=wp.zeros(capacity, dtype=wp.int32, device=device),
            fate_flags=wp.zeros(capacity, dtype=wp.int32, device=device),
        )
        # Streams for free slots are initialized too, so a daughter born into slot j uses the
        # same stream whether it appears at step 3 or step 300.
        wp.launch(_init_rng, dim=capacity, inputs=[pop.seed, pop.rng_state], device=device)
        return pop

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

    def states_numpy(self) -> np.ndarray:
        return self.cell_state.numpy()[: self.count].copy()

    def ages_numpy(self) -> np.ndarray:
        return self.age.numpy()[: self.count].copy()

    def oxygen_numpy(self) -> np.ndarray:
        return self.oxygen_local.numpy()[: self.count].copy()

    def glucose_numpy(self) -> np.ndarray:
        return self.glucose_local.numpy()[: self.count].copy()

    def local_field(self, name: str) -> wp.array:
        """Per-cell sampled values of a species: oxygen and glucose are first-class, others are
        allocated on first use (capacity-sized, so no reallocation in the loop)."""
        if name == "oxygen":
            return self.oxygen_local
        if name == "glucose":
            return self.glucose_local
        if name not in self.extra_local:
            self.extra_local[name] = wp.zeros(self.capacity, dtype=wp.float32, device=self.device)
        return self.extra_local[name]

    def local_numpy(self, name: str) -> np.ndarray:
        return self.local_field(name).numpy()[: self.count].copy()

    def neighbor_counts_numpy(self) -> np.ndarray:
        return self.neighbor_count.numpy()[: self.count].copy()

    def fate_flags_numpy(self) -> np.ndarray:
        return self.fate_flags.numpy()[: self.count].copy()
