"""Several diffusible species on one grid, sharing the per-state cell densities.

Oxygen is always the first species (the lifecycle reads ``oxygen_local``); glucose, when
present, is read by the lifecycle too (``glucose_local``); any other species (lactate, ...)
is sampled into ``population.extra_local[name]``. Species are solved in list order and a
species may use an *earlier* one as its production source (lactate from glucose).

The rule-based metabolic phenotype lives entirely in the per-state rate tables of the
species (``SpeciesParams.uptake_max`` / ``production``): HYPOXIC cells are the glycolytic
ones. This stands in for MicroC's metabolic network until Milestone 11.
"""

from __future__ import annotations

from dataclasses import dataclass

import warp as wp

from warpbiocell.cells.state import CellPopulation
from warpbiocell.fields.densities import CellDensities
from warpbiocell.fields.diffusion import SolveReport, SolverSettings
from warpbiocell.fields.oxygen import OxygenField, OxygenParams
from warpbiocell.fields.scalar_field import Boundary, GridGeometry
from warpbiocell.fields.species import SpeciesField, SpeciesParams


@dataclass(frozen=True)
class MetabolicReport:
    reports: dict  # species name -> SolveReport

    @property
    def sweeps(self) -> int:
        return sum(r.sweeps for r in self.reports.values())

    @property
    def residual(self) -> float:
        return max(r.residual for r in self.reports.values())

    @property
    def converged(self) -> bool:
        return all(r.converged for r in self.reports.values())


class MetabolicFields:
    def __init__(
        self,
        geometry: GridGeometry,
        oxygen_params: OxygenParams,
        species: list[SpeciesParams],
        settings: SolverSettings = SolverSettings(),
        boundary: tuple[Boundary, Boundary, Boundary] = (Boundary.DIRICHLET,) * 3,
        device: wp.context.Device | str | None = None,
        fixed_outside=None,
        oxygen_state_factors: dict | None = None,
    ):
        self.device = wp.get_device(device)
        self.densities = CellDensities(geometry, device=self.device)
        self.oxygen = OxygenField(
            geometry, oxygen_params, settings, boundary=boundary, device=self.device, fixed_outside=fixed_outside,
            densities=self.densities, uptake_state_factors=oxygen_state_factors,
        )
        self.species_fields: dict[str, SpeciesField] = {"oxygen": self.oxygen}
        for params in species:
            if params.name in self.species_fields:
                raise ValueError(f"duplicate species {params.name!r}")
            if params.production_source is not None and params.production_source not in self.species_fields:
                raise ValueError(f"species {params.name!r}: production source {params.production_source!r} must be an earlier species")
            self.species_fields[params.name] = SpeciesField(
                geometry, params, settings, boundary=boundary, device=self.device, fixed_outside=fixed_outside, densities=self.densities
            )
        self.last_report: MetabolicReport | None = None

    @property
    def geometry(self) -> GridGeometry:
        return self.oxygen.geometry

    @property
    def names(self) -> list[str]:
        return list(self.species_fields)

    @property
    def field(self):  # ScalarField of oxygen, for code that expects an OxygenField
        return self.oxygen.field

    def numpy(self):
        return self.oxygen.numpy()

    def __getitem__(self, name: str) -> SpeciesField:
        return self.species_fields[name]

    def update(self, population: CellPopulation) -> MetabolicReport:
        """Deposit the per-state densities once, then combine, solve and sample every species in order."""
        self.densities.deposit(population)
        reports = {}
        for name, species in self.species_fields.items():
            species.combine()
            source = self.species_fields[species.species.production_source] if species.species.production_source else None
            reports[name] = species.solve_steady_state(source)
            species.sample_at_cells(population)
        self.last_report = MetabolicReport(reports)
        # Mirror the oxygen report where OxygenField users expect it.
        self.oxygen.last_report = reports["oxygen"]
        return self.last_report
