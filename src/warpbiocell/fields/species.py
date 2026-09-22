"""A diffusible species with per-state uptake and production, solved on a ScalarField.

Each species has, per cell state ``k`` (PROLIFERATIVE, QUIESCENT, HYPOXIC, DEAD):

    uptake_max[k]     [conc um^3 / h per cell]   Michaelis-Menten uptake q = uptake_max * C/(K + C)
    production[k]     [conc um^3 / h per cell]   zero-order production capacity, optionally
                                                 multiplied by S/(K_S + S) of a source species
                                                 (lactate is made from glucose)

The cell densities per state are deposited once per step (``CellDensities``) and combined
into this species' effective consuming density and production capacity before its solve.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import warp as wp

from warpbiocell.cells.model import NUM_STATES, CellState
from warpbiocell.cells.state import CellPopulation
from warpbiocell.fields.densities import CellDensities
from warpbiocell.fields.diffusion import (
    SolveReport,
    SolverSettings,
    UptakeKinetics,
    explicit_step,
    solve_steady_state,
)
from warpbiocell.fields.scalar_field import Boundary, GridGeometry, ScalarField
from warpbiocell.kernels.field_kernels import sample_trilinear


def by_state(values: dict | float | None, default: float = 0.0) -> tuple[float, float, float, float]:
    """Per-state tuple from a scalar, a mapping keyed by state name (lower case) or None."""
    if values is None:
        return (default,) * NUM_STATES
    if isinstance(values, (int, float)):
        return (float(values),) * (NUM_STATES - 1) + (0.0,)  # living states only
    out = [default] * NUM_STATES
    for key, value in values.items():
        out[int(CellState[str(key).upper()])] = float(value)
    return tuple(out)


@dataclass(frozen=True)
class SpeciesParams:
    name: str
    unit: str
    diffusion: float  # um^2/h
    michaelis_k: float  # concentration unit
    boundary_value: float
    uptake_max: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)  # per state
    production: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)  # per state
    production_source: str | None = None  # name of the species whose MM factor scales production
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.name:
            raise ValueError("species needs a name")
        if len(self.uptake_max) != NUM_STATES or len(self.production) != NUM_STATES:
            raise ValueError(f"uptake_max and production need {NUM_STATES} per-state values")
        if min(self.uptake_max) < 0.0 or min(self.production) < 0.0:
            raise ValueError("rates must be non-negative")
        if self.boundary_value < 0.0:
            raise ValueError("boundary_value must be non-negative")
        UptakeKinetics(self.diffusion, max(self.uptake_max), self.michaelis_k)

    @property
    def uptake_scale(self) -> float:
        """The largest per-state uptake; the kernel's ``uptake_max``."""
        return max(self.uptake_max) if self.consumes else 0.0

    @property
    def uptake_weights(self) -> tuple[float, ...]:
        """Per-state weights so that ``density = sum_k w_k n_k`` is the consuming density in
        units of cells/um^3 at the scale rate (uniform living rates give weights of 1)."""
        scale = self.uptake_scale
        return tuple(u / scale if scale > 0.0 else 0.0 for u in self.uptake_max)

    @property
    def kinetics(self) -> UptakeKinetics:
        return UptakeKinetics(self.diffusion, self.uptake_scale, self.michaelis_k)

    @property
    def consumes(self) -> bool:
        return max(self.uptake_max) > 0.0

    @property
    def produces(self) -> bool:
        return max(self.production) > 0.0


class SpeciesField:
    """One species: its ScalarField, effective density, production capacity and solver buffers."""

    def __init__(
        self,
        geometry: GridGeometry,
        params: SpeciesParams,
        settings: SolverSettings = SolverSettings(),
        boundary: tuple[Boundary, Boundary, Boundary] = (Boundary.DIRICHLET,) * 3,
        device: wp.context.Device | str | None = None,
        fixed_outside=None,
        densities: CellDensities | None = None,
    ):
        self.species = params
        self.settings = settings
        self.field = ScalarField.create(geometry, params.boundary_value, boundary=boundary, device=device)
        self.device = self.field.device
        if fixed_outside is not None:
            if tuple(fixed_outside.geometry.shape) != tuple(geometry.shape):
                raise ValueError("the tissue region must be defined on the field grid")
            self.field.set_fixed(~fixed_outside.inside_mask())
        self.densities = densities if densities is not None else CellDensities(geometry, device=self.device)
        self.owns_densities = densities is None
        # Consuming density [cells/um^3, state-weighted] and production capacity [conc/h].
        self.density = wp.zeros(geometry.shape, dtype=wp.float32, device=self.device)
        self.production = wp.zeros(geometry.shape, dtype=wp.float32, device=self.device)
        self._scratch = wp.zeros(geometry.shape, dtype=wp.float32, device=self.device)
        self._residual = wp.zeros(1, dtype=wp.float32, device=self.device)
        self.last_report: SolveReport | None = None

    @property
    def params(self) -> SpeciesParams:
        return self.species

    @property
    def name(self) -> str:
        return self.species.name

    @property
    def geometry(self) -> GridGeometry:
        return self.field.geometry

    @property
    def values(self) -> wp.array:
        return self.field.values

    def numpy(self) -> np.ndarray:
        return self.field.numpy()

    # ---- cell <-> field transfer -------------------------------------------------------------

    def deposit_uptake(self, population: CellPopulation) -> None:
        """Deposit the per-state densities (when this species owns them) and combine them for this species."""
        if self.owns_densities:
            self.densities.deposit(population)
        self.combine()

    def combine(self) -> None:
        self.densities.combine(self.species.uptake_weights, self.density)
        if self.species.produces:
            self.densities.combine(self.species.production, self.production)

    def sample_at_cells(self, population: CellPopulation, out: wp.array | None = None) -> None:
        """Trilinear interpolation into ``out`` (default: ``population.local_field(name)``)."""
        if population.count == 0:
            return
        target = out if out is not None else population.local_field(self.name)
        geom = self.geometry
        wp.launch(
            sample_trilinear,
            dim=population.count,
            inputs=[self.field.values, population.position, wp.vec3(*geom.origin), 1.0 / geom.dx, target],
            device=self.device,
        )

    # ---- solvers --------------------------------------------------------------------------

    def solve_steady_state(self, source: SpeciesField | None = None) -> SolveReport:
        report = solve_steady_state(
            self.field, self.density, self.species.kinetics, self.settings, self._residual,
            production=self.production if self.species.produces else None,
            source=source.field.values if source is not None else None,
            source_k=source.species.michaelis_k if source is not None else 1.0,
        )
        self.last_report = report
        return report

    def explicit_step(self, dt: float, source: SpeciesField | None = None) -> None:
        explicit_step(
            self.field, self.density, self.species.kinetics, dt, self._scratch,
            production=self.production if self.species.produces else None,
            source=source.field.values if source is not None else None,
            source_k=source.species.michaelis_k if source is not None else 1.0,
        )

    def update(self, population: CellPopulation, source: SpeciesField | None = None, out: wp.array | None = None) -> SolveReport:
        """Deposit (if owner), combine, solve to steady state, sample: the per-cell-step update."""
        self.deposit_uptake(population)
        report = self.solve_steady_state(source)
        self.sample_at_cells(population, out)
        return report
