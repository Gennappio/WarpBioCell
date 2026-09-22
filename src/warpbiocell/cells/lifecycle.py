"""Cell lifecycle step: death, contact inhibition and division with deterministic slot allocation."""

from __future__ import annotations

from dataclasses import dataclass

import warp as wp
import warp.utils

from warpbiocell.cells.state import CellPopulation
from warpbiocell.kernels.cell_kernels import lifecycle_decide, place_daughters


class CapacityError(RuntimeError):
    """Raised when a step would create more cells than the population was allocated for."""


@dataclass(frozen=True)
class LifecycleParams:
    """Stochastic, rate-based lifecycle.

    division_rate          lambda [1/h]   illustrative; ln(2)/24 ~ 0.0289 corresponds to a 24 h
                                          doubling time in the absence of inhibition and death
    death_rate             [1/h]          placeholder until oxygen-dependent death exists
    inhibition_threshold   [-]            a cell with at least this many neighbours within the
                                          mechanics query radius is QUIESCENT; illustrative
    placement_factor       [-]            centre-to-centre distance of the new pair as a
                                          multiple of the parent radius (1.0 = one radius apart,
                                          i.e. an initial overlap of one radius that the
                                          mechanics relaxes)
    """

    division_rate: float = 0.0289
    death_rate: float = 0.0
    inhibition_threshold: int = 8
    placement_factor: float = 1.0

    def __post_init__(self):
        if self.division_rate < 0.0 or self.death_rate < 0.0:
            raise ValueError("rates must be non-negative")
        if self.inhibition_threshold < 1:
            raise ValueError("inhibition_threshold must be at least 1")
        if self.placement_factor <= 0.0:
            raise ValueError("placement_factor must be positive")


def lifecycle_step(population: CellPopulation, params: LifecycleParams, dt: float) -> int:
    """Advance death/state/division decisions by ``dt`` [h] and append daughters.

    Returns the number of daughters created. Contains exactly one host-device synchronization
    (reading that number). Raises :class:`CapacityError`, leaving no daughter placed, if the
    population would exceed its capacity; decisions taken in this step (deaths, state changes,
    consumed random draws) are kept, so the caller should stop the run rather than retry.
    """
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    n = population.count
    if n == 0:
        return 0
    device = population.device

    wp.launch(
        lifecycle_decide,
        dim=n,
        inputs=[
            dt,
            params.death_rate,
            params.division_rate,
            params.inhibition_threshold,
            population.cell_state,
            population.age,
            population.neighbor_count,
            population.rng_state,
        ],
        outputs=[population.divide_flag],
        device=device,
    )

    warp.utils.array_scan(population.divide_flag[:n], population.division_offset[:n], inclusive=True)
    n_daughters = int(population.division_offset[n - 1 : n].numpy()[0])  # the single sync
    if n_daughters == 0:
        return 0
    if n + n_daughters > population.capacity:
        raise CapacityError(
            f"{n_daughters} divisions would bring the population to {n + n_daughters} cells, "
            f"above the allocated capacity of {population.capacity}"
        )

    wp.launch(
        place_daughters,
        dim=n,
        inputs=[
            n,
            params.placement_factor,
            population.divide_flag,
            population.division_offset,
            population.position,
            population.radius,
            population.cell_state,
            population.cell_type,
            population.age,
            population.rng_state,
            population.velocity,
            population.neighbor_count,
        ],
        device=device,
    )
    population.count = n + n_daughters
    return n_daughters
