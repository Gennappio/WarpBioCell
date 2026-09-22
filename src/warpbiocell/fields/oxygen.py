"""Oxygen field: parameters, grid workspace and the cell <-> field coupling.

Unit convention: oxygen partial pressure in **mmHg**. Gas-phase percentages used in culture
(and by MicroC) convert at 37 C, 1 atm, humidified as ``pO2 = 7.13 mmHg per % O2``:
3% = 21 mmHg, 5% = 36 mmHg, 6% = 43 mmHg, 9% = 64 mmHg, 21% (air) = 150 mmHg.

Parameter provenance (AGENTS.md "Parameters and sources"):

    diffusion_coefficient  7.2e6 um^2/h = 2000 um^2/s        literature-derived, order of
                           magnitude for O2 in tissue (e.g. Grote et al. 1977; PhysiCell
                           uses 1e5 um^2/min = 1667 um^2/s)
    uptake_max             1.4e8 mmHg um^3/h per cell        estimated: 5e-17 mol/cell/s
                           (mid-range of Wagner et al. 2011) divided by the O2 solubility
                           alpha ~ 1.3 uM/mmHg = 1.3e-21 mol/(um^3 mmHg)
    michaelis_k            3.4 mmHg (0.45% O2)               estimated from MicroC's
                           half-saturation coefficient (Jayathilake et al. 2024)
    boundary_value         38 mmHg (~5% O2)                  illustrative tissue pO2

References are order-of-magnitude anchors, not fitted values; every one of them is a
candidate for the first sensitivity study.
"""

from __future__ import annotations

from dataclasses import dataclass

import warp as wp

from warpbiocell.cells.model import CellState
from warpbiocell.cells.state import CellPopulation
from warpbiocell.fields.densities import CellDensities
from warpbiocell.fields.diffusion import SolverSettings, UptakeKinetics
from warpbiocell.fields.scalar_field import Boundary, GridGeometry
from warpbiocell.fields.species import SpeciesField, SpeciesParams, by_state

MMHG_PER_PERCENT_O2 = 7.13


@dataclass(frozen=True)
class OxygenParams:
    diffusion_coefficient: float = 7.2e6  # um^2/h
    uptake_max: float = 1.4e8  # mmHg um^3/h per living cell
    michaelis_k: float = 3.4  # mmHg
    boundary_value: float = 38.0  # mmHg

    def __post_init__(self):
        if self.boundary_value < 0.0:
            raise ValueError("boundary_value must be non-negative")
        self.kinetics  # validates the rest

    @property
    def kinetics(self) -> UptakeKinetics:
        return UptakeKinetics(self.diffusion_coefficient, self.uptake_max, self.michaelis_k)


class OxygenField(SpeciesField):
    """Oxygen as a SpeciesField with uniform uptake over living states (optionally scaled per
    state with ``uptake_state_factors``, e.g. ``{"hypoxic": 0.5}``); samples into
    ``population.oxygen_local``."""

    def __init__(
        self,
        geometry: GridGeometry,
        params: OxygenParams,
        settings: SolverSettings = SolverSettings(),
        boundary: tuple[Boundary, Boundary, Boundary] = (Boundary.DIRICHLET,) * 3,
        device: wp.context.Device | str | None = None,
        fixed_outside=None,
        densities: CellDensities | None = None,
        uptake_state_factors: dict | None = None,
    ):
        """``fixed_outside`` (a TissueRegion on the same grid) pins every node outside the tissue
        to ``boundary_value``: the tissue surface becomes the oxygen source."""
        self.oxygen_params = params
        super().__init__(
            geometry,
            species_params(params, uptake_state_factors),
            settings,
            boundary=boundary,
            device=device,
            fixed_outside=fixed_outside,
            densities=densities,
        )

    @property
    def params(self) -> OxygenParams:  # type: ignore[override]
        return self.oxygen_params

    def sample_at_cells(self, population: CellPopulation, out: wp.array | None = None) -> None:
        super().sample_at_cells(population, out if out is not None else population.oxygen_local)


def species_params(params: OxygenParams, uptake_state_factors: dict | None = None) -> SpeciesParams:
    factors = by_state(uptake_state_factors, default=1.0) if uptake_state_factors else (1.0, 1.0, 1.0, 0.0)
    uptake = tuple(params.uptake_max * f if k != int(CellState.DEAD) else 0.0 for k, f in enumerate(factors))
    return SpeciesParams(
        name="oxygen",
        unit="mmHg",
        diffusion=params.diffusion_coefficient,
        michaelis_k=params.michaelis_k,
        boundary_value=params.boundary_value,
        uptake_max=uptake,
    )
