"""Overdamped contact mechanics: parameters, stability rule and the substep driver."""

from __future__ import annotations

from dataclasses import dataclass

import warp as wp

from warpbiocell.cells.state import CellPopulation
from warpbiocell.kernels.geometry_kernels import confinement_velocities
from warpbiocell.kernels.mechanics_kernels import contact_velocities, integrate_positions
from warpbiocell.spatial.neighbors import NeighborGrid

# Explicit Euler on a single overlapping pair contracts the overlap by (1 - 2*rate*dt) per
# substep; a cell pushed by several aligned neighbours contracts faster. Above this product the
# pair itself overshoots, so we refuse it.
MAX_RATE_DT = 0.5


@dataclass(frozen=True)
class ContactParams:
    """Linear repulsion between overlapping spheres, overdamped.

    stiffness   k      [force / um]      illustrative
    damping     gamma  [force * h / um]  illustrative
    max_radius         [um]              largest cell radius present; sets the query radius
    margin             [um]              added to 2 * max_radius so the grid also sees near-contacts
    wall_rate          [1/h]             confinement rate against a tissue surface (the
                                         k/gamma of the wall contact); used only with a region

    Only ``rate = k / gamma`` [1/h] enters the dynamics: it is the inverse relaxation time of an
    isolated overlapping pair (overlap ~ exp(-2 * rate * t)).
    """

    stiffness: float
    damping: float
    max_radius: float
    margin: float = 0.0
    wall_rate: float = 10.0

    def __post_init__(self):
        if self.stiffness <= 0.0 or self.damping <= 0.0:
            raise ValueError("stiffness and damping must be positive")
        if self.max_radius <= 0.0:
            raise ValueError("max_radius must be positive")
        if self.margin < 0.0:
            raise ValueError("margin must be non-negative")
        if self.wall_rate < 0.0:
            raise ValueError("wall_rate must be non-negative")

    @property
    def rate(self) -> float:
        return self.stiffness / self.damping

    @property
    def query_radius(self) -> float:
        return 2.0 * self.max_radius + self.margin

    def check_substep(self, dt: float) -> None:
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.rate * dt > MAX_RATE_DT:
            raise ValueError(
                f"rate*dt = {self.rate * dt:.3g} exceeds {MAX_RATE_DT}: explicit contact relaxation "
                "would overshoot. Reduce dt_mechanics or stiffness/damping."
            )
        if self.wall_rate * dt > MAX_RATE_DT:
            raise ValueError(f"wall_rate*dt = {self.wall_rate * dt:.3g} exceeds {MAX_RATE_DT}: reduce dt_mechanics or wall_rate.")


def make_neighbor_grid(params: ContactParams, device, dim: int = 128) -> NeighborGrid:
    return NeighborGrid(cell_size=params.query_radius, dim=dim, device=device)


def contact_substep(population: CellPopulation, grid: NeighborGrid, params: ContactParams, dt: float, region=None) -> None:
    """One explicit substep: rebuild the grid, gather contact velocities (plus the tissue-wall
    push when a ``region`` is given), move the cells."""
    n = population.count
    if n == 0:
        return
    grid.build(population)
    wp.launch(
        contact_velocities,
        dim=n,
        inputs=[
            grid.id,
            population.position,
            population.radius,
            params.query_radius,
            params.rate,
        ],
        outputs=[population.velocity, population.neighbor_count],
        device=population.device,
    )
    if region is not None:
        geom = region.geometry
        wp.launch(
            confinement_velocities,
            dim=n,
            inputs=[region.sdf, region.gradient, wp.vec3(*geom.origin), 1.0 / geom.dx, params.wall_rate, population.position, population.radius],
            outputs=[population.velocity],
            device=population.device,
        )
    wp.launch(
        integrate_positions,
        dim=n,
        inputs=[population.position, population.velocity, dt],
        device=population.device,
    )


def relax_contacts(
    population: CellPopulation,
    grid: NeighborGrid,
    params: ContactParams,
    dt: float,
    substeps: int,
    region=None,
) -> None:
    params.check_substep(dt)
    for _ in range(substeps):
        contact_substep(population, grid, params, dt, region)
