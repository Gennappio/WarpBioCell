"""One cell step = lifecycle decisions, then mechanical relaxation.

Operator splitting (AGENTS.md "Simulation cycle", steps 1-5 arrive with the oxygen milestone):

    6-7. lifecycle_step      states, deaths, divisions; daughters appended     (1 host sync)
    8-9. relax_contacts      `mechanics_substeps` explicit substeps of dt_mechanics, each
                             rebuilding the hash grid; leaves neighbor_count for the next
                             lifecycle decision                                (no host sync)

The crowding a cell sees when deciding at step n+1 is therefore the one left by the mechanics
at the end of step n, i.e. after this step's daughters have been pushed apart.
"""

from __future__ import annotations

from dataclasses import dataclass

from warpbiocell.cells.lifecycle import LifecycleParams, lifecycle_step
from warpbiocell.cells.mechanics import ContactParams, relax_contacts
from warpbiocell.cells.state import CellPopulation
from warpbiocell.spatial.neighbors import NeighborGrid


@dataclass(frozen=True)
class TimeStepping:
    """dt_cells [h] per lifecycle update; dt_mechanics [h] per contact substep."""

    dt_cells: float = 0.25
    dt_mechanics: float = 0.025

    def __post_init__(self):
        if self.dt_cells <= 0.0 or self.dt_mechanics <= 0.0:
            raise ValueError("time steps must be positive")
        if self.dt_mechanics > self.dt_cells:
            raise ValueError("dt_mechanics cannot exceed dt_cells")

    @property
    def mechanics_substeps(self) -> int:
        return max(1, int(round(self.dt_cells / self.dt_mechanics)))


def cell_step(
    population: CellPopulation,
    grid: NeighborGrid,
    lifecycle: LifecycleParams,
    contact: ContactParams,
    stepping: TimeStepping,
) -> int:
    """Advance the population by ``stepping.dt_cells``. Returns the number of daughters created."""
    n_daughters = lifecycle_step(population, lifecycle, stepping.dt_cells)
    relax_contacts(population, grid, contact, stepping.dt_mechanics, stepping.mechanics_substeps)
    return n_daughters
