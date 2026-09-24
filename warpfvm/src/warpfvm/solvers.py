"""Solvers with FiPy's names and tolerance semantics, all running preconditioned CG on the device.

Every system warpfvm assembles is symmetric (FiPy's diffusion discretization on a uniform
grid, sources on the diagonal or on the right-hand side), and positive definite as soon as a
boundary value is fixed, a sink or a transient term is present. Conjugate gradient is the
Krylov method for such systems, so every solver name runs it (warpfvm.krylov): BiCGSTAB and
GMRES would reach the same solution in more work. The names exist so that FiPy code runs
unchanged:

    LinearPCGSolver / LinearCGSolver   CG with the requested tolerance
    LinearBicgstabSolver               same, FiPy's name kept
    LinearGMRESSolver                  same, FiPy's name kept (MicroC picks it for problems
                                       without a fixed boundary value)
    LinearLUSolver                     no factorization exists in Warp. As FiPy's LU solver, it
                                       returns the starting values untouched when their
                                       residual already meets ``tolerance``; otherwise it runs
                                       CG to near machine precision (relative residual 1e-12
                                       in float64, 1e-6 in float32). ``iterations`` (LU
                                       refinement steps in FiPy) becomes a budget that grows
                                       with the grid

Tolerance, as FiPy >= 4 with the default criterion ("RHS"): stop when
``||L x - b|| <= tolerance * ||b||``. Also available: "unscaled" (``||L x - b|| <= tolerance``),
"initial" (relative to the residual of the starting values) and "legacy" (= "RHS" with FiPy's
legacy default tolerance 1e-10). The residual is the true one, recomputed at the end.

A steady problem without any fixed boundary value, sink or transient term is singular: its
solutions differ by a constant. The solvers then remove from the right-hand side its component
along the constants (so a problem whose sources do not balance still gets the least-squares
answer, with a warning) and return the solution whose mean equals the mean of the starting
values.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
import warp as wp

from warpfvm import kernels
from warpfvm.krylov import pcg


class ConvergenceWarning(UserWarning):
    """A solve reached its iteration limit above its tolerance."""


class SingularSystemWarning(UserWarning):
    """A singular system's sources do not balance: no exact steady state exists."""


@dataclass(frozen=True)
class Convergence:
    """Outcome of the last solve, in FiPy's units (``L`` and ``b`` as FiPy assembles them)."""

    solver: str
    iterations: int
    residual: float  # ||L x - b|| after the solve (the true residual)
    rhs_norm: float  # ||b|| (after the projection, for a singular system)
    threshold: float  # the absolute stopping threshold on ||L x - b||
    converged: bool
    restarts: int = 0
    singular: bool = False

    @property
    def relative_residual(self) -> float:
        return self.residual / self.rhs_norm if self.rhs_norm > 0.0 else self.residual


class Solver:
    """Base class: preconditioned conjugate gradient with FiPy's constructor arguments."""

    DEFAULT_TOLERANCE = 1e-5  # FiPy's default with the "default" criterion
    LEGACY_TOLERANCE = 1e-10  # FiPy's default with the "legacy" criterion
    DEFAULT_ITERATIONS = 1000  # FiPy's default
    _criteria = ("default", "RHS", "unscaled", "initial", "legacy")

    def __init__(
        self,
        tolerance="default",
        criterion="default",
        iterations="default",
        precon="default",
        absolute_tolerance=0.0,
        *,
        check_every=10,
        use_cuda_graph=False,
        **ignored,
    ):
        if criterion not in self._criteria:
            raise NotImplementedError(f"criterion {criterion!r} is not supported; use one of {self._criteria}")
        self.criterion = criterion
        default = self.LEGACY_TOLERANCE if criterion == "legacy" else self.DEFAULT_TOLERANCE
        self.tolerance = default if tolerance in ("default", None) else float(tolerance)
        self.iterations = self.DEFAULT_ITERATIONS if iterations in ("default", None) else int(iterations)
        if precon in ("default", "jacobi", "diag", "diagonal"):
            precon = "jacobi"
        elif precon in (None, "none", "id"):
            precon = None
        else:
            raise NotImplementedError(f"preconditioner {precon!r} is not supported; use 'jacobi' or None")
        self.precon = precon
        self.absolute_tolerance = float(absolute_tolerance)
        self.check_every = int(check_every)
        self.use_cuda_graph = bool(use_cuda_graph)
        self.convergence: Convergence | None = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __repr__(self) -> str:
        return f"{type(self).__name__}(tolerance={self.tolerance:g}, iterations={self.iterations}, precon={self.precon!r})"

    def _limits(self, system, x):
        """(relative tolerance, absolute tolerance, iteration budget) in the scaled system."""
        scale = abs(system.scale)
        absolute = self.absolute_tolerance / scale
        if self.criterion in ("default", "RHS", "legacy"):
            return self.tolerance, absolute, self.iterations
        if self.criterion == "unscaled":
            return 0.0, max(self.tolerance / scale, absolute), self.iterations
        return 0.0, max(self.tolerance * system.residual_norm(x), absolute), self.iterations  # "initial"

    def _start_threshold(self, system, x):
        """Absolute residual at or below which the starting values are returned untouched
        (None: only the stopping threshold applies)."""
        return None

    def _solve(self, system, var) -> Convergence:
        x = var._array
        mean_before = None
        if system.singular:
            mean_before = _project_out_constants(system, x)
        start_threshold = self._start_threshold(system, x)
        rtol, atol, maxiter = self._limits(system, x)
        result = pcg(
            system, x, rtol, atol, maxiter,
            precondition=self.precon == "jacobi", check_every=self.check_every, use_cuda_graph=self.use_cuda_graph,
            start_threshold=start_threshold,
        )
        if mean_before is not None:
            _shift_to_mean(x, mean_before)
        scale = abs(system.scale)
        self.convergence = Convergence(
            solver=type(self).__name__,
            iterations=result.iterations,
            residual=scale * result.residual,
            rhs_norm=scale * result.rhs_norm,
            threshold=scale * result.threshold,
            converged=result.converged,
            restarts=result.restarts,
            singular=system.singular,
        )
        if not result.converged:
            c = self.convergence
            warnings.warn(
                f"{c.solver} stopped after {c.iterations} iterations with residual {c.residual:.3g} "
                f"above the threshold {c.threshold:.3g}",
                ConvergenceWarning,
                stacklevel=4,
            )
        return self.convergence


def _mean(values: wp.array) -> float:
    return float(wp.utils.array_sum(values)) / values.shape[0]


def _shift_to_mean(values: wp.array, target: float) -> None:
    shift = target - _mean(values)
    wp.launch(kernels.get(values.dtype).add_constant, dim=values.shape[0], inputs=[values, values.dtype(shift)], device=values.device)


def _project_out_constants(system, x) -> float:
    """Remove the constant component of the right-hand side (the null space of a singular
    all-Neumann operator is the constants); warn when it was not negligible. Returns the mean
    of the starting values, which the solution keeps."""
    mean_rhs = _mean(system.rhs)
    norm = system.rhs_norm()
    removed = abs(mean_rhs) * math.sqrt(system.size)
    if norm > 0.0 and removed > 1e-8 * norm:
        warnings.warn(
            f"the steady problem has no fixed boundary value, sink or transient term and its sources do not "
            f"balance ({removed / norm:.2g} of the right-hand side lies along the constants): no steady state "
            "exists, returning the least-squares solution",
            SingularSystemWarning,
            stacklevel=5,
        )
    _shift_to_mean(system.rhs, 0.0)
    return _mean(x)


class LinearPCGSolver(Solver):
    """Jacobi-preconditioned conjugate gradient."""


LinearCGSolver = LinearPCGSolver


class LinearBicgstabSolver(Solver):
    """FiPy's name, kept for drop-in use: runs the same preconditioned CG (see the module docstring)."""


class LinearGMRESSolver(Solver):
    """FiPy's name, kept for drop-in use: runs the same preconditioned CG (see the module docstring)."""

    def __init__(self, tolerance="default", criterion="default", iterations="default", precon="default", absolute_tolerance=0.0, *, restart=None, **kwargs):
        super().__init__(tolerance, criterion, iterations, precon, absolute_tolerance, **kwargs)


class LinearLUSolver(LinearPCGSolver):
    """FiPy's direct solver, reproduced: FiPy's LinearLUSolver first compares the residual of the
    starting values with ``tolerance`` (under ``criterion``) and returns them untouched when it
    is small enough; otherwise one LU solve gives the exact discrete solution. warpfvm makes the
    same check, then replaces the LU solve by preconditioned CG to near machine precision.

    The check matters: with MicroC's ``tolerance=1e-6`` and the default criterion, a field whose
    starting residual is below 1e-6 ||b|| is not updated at all, and ||b|| contains the boundary
    terms, so a weak interior source can leave the field exactly unchanged in FiPy (and here).
    """

    RTOL_FLOAT64 = 1e-12
    RTOL_FLOAT32 = 1e-6

    def _start_threshold(self, system, x):
        rtol, atol, _ = Solver._limits(self, system, x)
        return max(rtol * system.rhs_norm(), atol)

    def _limits(self, system, x):
        rtol = self.RTOL_FLOAT64 if system.dtype == wp.float64 else self.RTOL_FLOAT32
        budget = max(2000, 50 * max(system.mesh.shape))
        return rtol, self.absolute_tolerance / abs(system.scale), budget


# FiPy's defaults with the scipy suite (the one MicroCpy installs): an equation solved without
# a solver uses the LU solver.
DefaultSolver = LinearLUSolver
DefaultAsymmetricSolver = LinearLUSolver
GeneralSolver = LinearLUSolver
