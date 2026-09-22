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

import numpy as np
import warp as wp

from warpbiocell.cells.state import CellPopulation
from warpbiocell.fields.diffusion import (
    SolveReport,
    SolverSettings,
    UptakeKinetics,
    explicit_step,
    solve_steady_state,
)
from warpbiocell.fields.scalar_field import Boundary, GridGeometry, ScalarField
from warpbiocell.kernels.field_kernels import deposit_trilinear, fixed_point_to_density, sample_trilinear

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


class OxygenField:
    """A ScalarField plus the consuming-cell density and the buffers the solvers need."""

    def __init__(
        self,
        geometry: GridGeometry,
        params: OxygenParams,
        settings: SolverSettings = SolverSettings(),
        boundary: tuple[Boundary, Boundary, Boundary] = (Boundary.DIRICHLET,) * 3,
        device: wp.context.Device | str | None = None,
        fixed_outside=None,
    ):
        """``fixed_outside`` (a TissueRegion on the same grid) pins every node outside the tissue
        to ``boundary_value``: the tissue surface becomes the oxygen source."""
        self.params = params
        self.settings = settings
        self.field = ScalarField.create(geometry, params.boundary_value, boundary=boundary, device=device)
        self.device = self.field.device
        if fixed_outside is not None:
            if tuple(fixed_outside.geometry.shape) != tuple(geometry.shape):
                raise ValueError("the tissue region must be defined on the oxygen grid")
            self.field.set_fixed(~fixed_outside.inside_mask())
        self.density = wp.zeros(geometry.shape, dtype=wp.float32, device=self.device)  # cells / um^3
        self._accumulator = wp.zeros(geometry.shape, dtype=wp.int64, device=self.device)
        self._scratch = wp.zeros(geometry.shape, dtype=wp.float32, device=self.device)
        self._residual = wp.zeros(1, dtype=wp.float32, device=self.device)
        self.last_report: SolveReport | None = None

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
        """Rebuild ``density`` from the living cells (dead cells do not consume)."""
        if population.device != self.device:
            raise RuntimeError("population and OxygenField must live on the same device")
        geom = self.geometry
        self._accumulator.zero_()
        if population.count > 0:
            wp.launch(
                deposit_trilinear,
                dim=population.count,
                inputs=[population.position, population.cell_state, wp.vec3(*geom.origin), 1.0 / geom.dx, self._accumulator],
                device=self.device,
            )
        wp.launch(
            fixed_point_to_density,
            dim=geom.shape,
            inputs=[self._accumulator, 1.0 / geom.voxel_volume, self.density],
            device=self.device,
        )

    def sample_at_cells(self, population: CellPopulation) -> None:
        """Trilinear interpolation of the field into ``population.oxygen_local``."""
        if population.count == 0:
            return
        geom = self.geometry
        wp.launch(
            sample_trilinear,
            dim=population.count,
            inputs=[self.field.values, population.position, wp.vec3(*geom.origin), 1.0 / geom.dx, population.oxygen_local],
            device=self.device,
        )

    # ---- solvers --------------------------------------------------------------------------

    def solve_steady_state(self) -> SolveReport:
        report = solve_steady_state(self.field, self.density, self.params.kinetics, self.settings, self._residual)
        self.last_report = report
        return report

    def explicit_step(self, dt: float) -> None:
        explicit_step(self.field, self.density, self.params.kinetics, dt, self._scratch)

    def update(self, population: CellPopulation) -> SolveReport:
        """Deposit, solve to steady state, sample: the per-cell-step field update."""
        self.deposit_uptake(population)
        report = self.solve_steady_state()
        self.sample_at_cells(population)
        return report
