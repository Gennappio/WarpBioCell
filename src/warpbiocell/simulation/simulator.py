"""One cell step = oxygen update, lifecycle decisions, mechanical relaxation.

Operator splitting (AGENTS.md "Simulation cycle"):

    1-5. oxygen.update       deposit living cells on the grid, solve the quasi-steady field
                             from the previous solution (warm start), sample it at the cells
                             (one host sync per residual check)             [optional]
    6-7. lifecycle_step      states, deaths, divisions; daughters appended     (1 host sync)
    8-9. relax_contacts      `mechanics_substeps` explicit substeps of dt_mechanics, each
                             rebuilding the hash grid; leaves neighbor_count for the next
                             lifecycle decision                                (no host sync)

The lifecycle at step n therefore sees the oxygen of the configuration at the start of step
n and the crowding left by the mechanics at the end of step n-1. Daughters inherit the
parent's sampled oxygen until the next field update.
"""

from __future__ import annotations

from dataclasses import dataclass

from warpbiocell.cells.lifecycle import LifecycleParams, lifecycle_step
from warpbiocell.cells.mechanics import ContactParams, relax_contacts
from warpbiocell.cells.state import CellPopulation
from warpbiocell.fields.oxygen import OxygenField
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


@dataclass(frozen=True)
class StepReport:
    daughters: int
    field_sweeps: int = 0
    field_residual: float = 0.0
    field_converged: bool = True


def cell_step(
    population: CellPopulation,
    grid: NeighborGrid,
    lifecycle: LifecycleParams,
    contact: ContactParams,
    stepping: TimeStepping,
    oxygen: OxygenField | None = None,
    region=None,
) -> StepReport:
    """Advance the population by ``stepping.dt_cells``. ``region`` (a TissueRegion) confines the cells."""
    if oxygen is not None:
        field = oxygen.update(population)
        n_daughters = lifecycle_step(population, lifecycle, stepping.dt_cells)
        relax_contacts(population, grid, contact, stepping.dt_mechanics, stepping.mechanics_substeps, region)
        return StepReport(n_daughters, field.sweeps, field.residual, field.converged)
    n_daughters = lifecycle_step(population, lifecycle, stepping.dt_cells)
    relax_contacts(population, grid, contact, stepping.dt_mechanics, stepping.mechanics_substeps, region)
    return StepReport(n_daughters)
