"""Reaction-diffusion solvers on a ScalarField with Michaelis-Menten uptake.

``solve_steady_state`` is the production path (the field is quasi-steady on the cell time
scale, AGENTS.md "Time scales"). ``explicit_step`` is the time-accurate reference used for
numerical validation only.
"""

from __future__ import annotations

from dataclasses import dataclass

import warp as wp

from warpbiocell.fields.scalar_field import ScalarField
from warpbiocell.kernels.field_kernels import ftcs_step, sor_sweep, steady_state_residual


@dataclass(frozen=True)
class UptakeKinetics:
    """``q(O) = uptake_max * O / (michaelis_k + O)`` per consuming cell; ``diffusion`` D [um^2/h]."""

    diffusion: float
    uptake_max: float
    michaelis_k: float

    def __post_init__(self):
        if self.diffusion <= 0.0:
            raise ValueError("diffusion coefficient must be positive")
        if self.uptake_max < 0.0 or self.michaelis_k <= 0.0:
            raise ValueError("uptake_max must be non-negative and michaelis_k positive")


@dataclass(frozen=True)
class SolverSettings:
    """Red-black SOR settings.

    omega        relaxation factor; 1.0 is Gauss-Seidel, ~1.8 is near-optimal for pure
                 diffusion on ~50 nodes per axis
    max_sweeps   hard budget per solve (each sweep is two colour passes)
    tolerance    dimensionless max-norm residual (see kernels) at which to stop. In float32
                 the residual bottoms out around 5e-7 * omega / (2 - omega) (round-off
                 amplified by over-relaxation), so tolerances below ~5e-6 may never be met.
                 The solution error can exceed the residual by the problem's condition
                 number; on the spheroid problem the warm start from the previous cell step
                 keeps it far below the biological thresholds.
    check_every  sweeps between residual checks; each check is one host sync
    """

    omega: float = 1.5
    max_sweeps: int = 2000
    tolerance: float = 1.0e-5
    check_every: int = 20

    def __post_init__(self):
        if not 0.0 < self.omega < 2.0:
            raise ValueError("omega must lie in (0, 2)")
        if self.max_sweeps < 1 or self.check_every < 1:
            raise ValueError("max_sweeps and check_every must be positive")
        if self.tolerance <= 0.0:
            raise ValueError("tolerance must be positive")


@dataclass(frozen=True)
class SolveReport:
    sweeps: int
    residual: float
    converged: bool


def solve_steady_state(
    field: ScalarField,
    density: wp.array,
    kinetics: UptakeKinetics,
    settings: SolverSettings,
    residual_buffer: wp.array,
) -> SolveReport:
    """Iterate SOR sweeps in place, starting from the field's current values (warm start)."""
    geom = field.geometry
    dx2_over_D = geom.dx**2 / kinetics.diffusion
    bc_x, bc_y, bc_z = field.boundary_flags
    reference = field.boundary_value if field.boundary_value > 0.0 else 1.0
    dev = field.device

    sweeps = 0
    residual = float("inf")
    while sweeps < settings.max_sweeps:
        for _ in range(min(settings.check_every, settings.max_sweeps - sweeps)):
            for color in (0, 1):
                wp.launch(
                    sor_sweep,
                    dim=geom.shape,
                    inputs=[color, settings.omega, dx2_over_D, kinetics.uptake_max, kinetics.michaelis_k, bc_x, bc_y, bc_z, field.fixed, density, field.values],
                    device=dev,
                )
            sweeps += 1
        residual_buffer.zero_()
        wp.launch(
            steady_state_residual,
            dim=geom.shape,
            inputs=[dx2_over_D, kinetics.uptake_max, kinetics.michaelis_k, 1.0 / reference, bc_x, bc_y, bc_z, field.fixed, density, field.values, residual_buffer],
            device=dev,
        )
        residual = float(residual_buffer.numpy()[0])
        if residual < settings.tolerance:
            return SolveReport(sweeps=sweeps, residual=residual, converged=True)
    return SolveReport(sweeps=sweeps, residual=residual, converged=False)


def max_stable_dt(field: ScalarField, kinetics: UptakeKinetics) -> float:
    """Explicit-scheme stability limit ``dx^2 / (6 D)`` [h]."""
    return field.geometry.dx**2 / (6.0 * kinetics.diffusion)


def explicit_step(
    field: ScalarField,
    density: wp.array,
    kinetics: UptakeKinetics,
    dt: float,
    scratch: wp.array,
) -> None:
    """One FTCS step; writes into ``scratch`` then copies back so ``field.values`` stays the same array."""
    limit = max_stable_dt(field, kinetics)
    if dt > limit * (1.0 + 1.0e-6):
        raise ValueError(f"dt = {dt:.3g} h exceeds the explicit stability limit dx^2/(6D) = {limit:.3g} h")
    geom = field.geometry
    bc_x, bc_y, bc_z = field.boundary_flags
    wp.launch(
        ftcs_step,
        dim=geom.shape,
        inputs=[dt, kinetics.diffusion / geom.dx**2, kinetics.uptake_max, kinetics.michaelis_k, bc_x, bc_y, bc_z, field.fixed, density, field.values, scratch],
        device=field.device,
    )
    wp.copy(field.values, scratch)
