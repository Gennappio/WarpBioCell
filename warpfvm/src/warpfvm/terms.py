"""FiPy's term algebra for one scalar unknown.

    DiffusionTerm(coeff=D)          div(D grad phi), D a number
    TransientTerm(coeff=c)          c d(phi)/dt, backward Euler over ``dt``
    ImplicitSourceTerm(coeff=S)     S phi, S a number or one value per cell
    a number, array or CellVariable q   an explicit source q

Terms combine with ``+``, ``-``, unary ``-``, multiplication and division by numbers, and
``==`` (``a == b`` means ``a - b = 0``), exactly as in FiPy::

    eq = DiffusionTerm(coeff=D) - ImplicitSourceTerm(coeff=k) == -source
    eq.solve(var=phi, solver=LinearPCGSolver(tolerance=1e-8))

    eq = TransientTerm() == DiffusionTerm(coeff=D) + source
    eq.solve(var=phi, dt=dt)

Not supported, with an explicit error: coupled equations (terms bound to several variables),
convection terms, per-cell or tensor diffusion coefficients, higher-order diffusion,
old-style ``boundaryConditions=`` arguments.
"""

from __future__ import annotations

import math
import numbers

import numpy as np
import warp as wp

from warpfvm.system import LinearSystem, assemble
from warpfvm.variables import CellVariable, _ScaledVariable


def _require_scalar(coeff, name: str) -> float:
    if isinstance(coeff, numbers.Real) or (isinstance(coeff, np.ndarray) and coeff.ndim == 0):
        return float(coeff)
    raise NotImplementedError(
        f"{name} needs a number as coefficient: per-cell, per-face and tensor coefficients are "
        "not supported (MicroC uses a constant diffusion coefficient)"
    )


def _as_term(other) -> Term:
    if isinstance(other, Term):
        return other
    if isinstance(other, (numbers.Real, np.ndarray, list, tuple, CellVariable, _ScaledVariable, wp.array)):
        return _ExplicitSourceTerm(other)
    raise TypeError(f"cannot combine a {type(other).__name__} with a term")


def _default_solver(solver):
    from warpfvm.solvers import DefaultSolver, Solver

    if solver is None:
        return DefaultSolver()
    if not isinstance(solver, Solver):
        raise TypeError(f"{type(solver).__module__}.{type(solver).__name__} is not a warpfvm solver: use warpfvm.solvers")
    return solver


def _no_boundary_conditions(boundaryConditions):
    if boundaryConditions not in ((), None, []):
        raise NotImplementedError("old-style boundaryConditions are not supported: use var.constrain(value, where=faces)")


class Term:
    """Base of every term and equation."""

    __hash__ = object.__hash__

    def _items(self) -> list:
        raise NotImplementedError

    # --- algebra ------------------------------------------------------------------------------

    def __add__(self, other):
        if isinstance(other, numbers.Real) and other == 0:
            return self
        return _Equation(self._items() + _as_term(other)._items())

    def __radd__(self, other):
        if isinstance(other, numbers.Real) and other == 0:
            return self
        return _Equation(_as_term(other)._items() + self._items())

    def __neg__(self):
        return _Equation([(-scale, term) for scale, term in self._items()])

    def __pos__(self):
        return self

    def __sub__(self, other):
        return self + (-_as_term(other))

    def __rsub__(self, other):
        return _as_term(other) + (-self)

    def __eq__(self, other):
        return self - other

    def __mul__(self, other):
        if isinstance(other, numbers.Real):
            return _Equation([(float(other) * scale, term) for scale, term in self._items()])
        return NotImplemented

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, numbers.Real):
            return self * (1.0 / float(other))
        return NotImplemented

    # --- solving ------------------------------------------------------------------------------

    def _resolve_var(self, var) -> CellVariable:
        bound = {id(term.var): term.var for _, term in self._items() if term.var is not None}
        if var is None:
            if len(bound) == 1:
                var = next(iter(bound.values()))
            elif not bound:
                raise ValueError("the solution variable must be given: eq.solve(var=...)")
            else:
                raise NotImplementedError("coupled equations (terms on several variables) are not supported")
        elif any(other is not var for other in bound.values()):
            raise NotImplementedError("terms bound to another variable (coupled equations) are not supported")
        if not isinstance(var, CellVariable):
            raise TypeError(f"the solution variable must be a warpfvm CellVariable, got {type(var).__name__}")
        return var

    def assemble(self, var=None, dt=None, underRelaxation=None) -> LinearSystem:
        """The scaled device system ``A x = rhs`` for this equation (see warpfvm.system)."""
        return assemble(self._items(), self._resolve_var(var), dt, underRelaxation)

    def solve(self, var=None, solver=None, boundaryConditions=(), dt=None) -> None:
        """Assemble and solve once, in place on ``var`` (FiPy's ``solve``; returns None)."""
        _no_boundary_conditions(boundaryConditions)
        var = self._resolve_var(var)
        system = assemble(self._items(), var, dt)
        _default_solver(solver)._solve(system, var)

    def sweep(
        self, var=None, solver=None, boundaryConditions=(), dt=None, underRelaxation=None,
        residualFn=None, cacheResidual=False, cacheError=False,
    ) -> float:
        """Assemble, solve once, and return ``||L x - b||`` of the values *before* the solve
        (FiPy's sweep residual, in FiPy's units)."""
        _no_boundary_conditions(boundaryConditions)
        if residualFn is not None or cacheError:
            raise NotImplementedError("residualFn and cacheError are not supported")
        var = self._resolve_var(var)
        system = assemble(self._items(), var, dt, underRelaxation)
        r = system.residual(var._array)
        residual = abs(system.scale) * math.sqrt(max(float(wp.utils.array_inner(r, r)), 0.0))
        self.residualVector = -system.scale * r.numpy().astype(np.float64) if cacheResidual else None
        _default_solver(solver)._solve(system, var)
        return residual

    def justResidualVector(self, var=None, solver=None, boundaryConditions=(), dt=None, underRelaxation=None, residualFn=None) -> np.ndarray:
        """``L x - b`` for the current values, without solving (FiPy's units)."""
        _no_boundary_conditions(boundaryConditions)
        if residualFn is not None:
            raise NotImplementedError("residualFn is not supported")
        var = self._resolve_var(var)
        system = assemble(self._items(), var, dt, underRelaxation)
        return -system.scale * system.residual(var._array).numpy().astype(np.float64)


class _UnaryTerm(Term):
    _kind = ""

    def __init__(self, coeff=1.0, var=None):
        if var is not None and not isinstance(var, CellVariable):
            raise TypeError("var must be a warpfvm CellVariable")
        self.coeff = coeff
        self.var = var

    def _items(self) -> list:
        return [(1.0, self)]

    def __repr__(self) -> str:
        return f"{type(self).__name__}(coeff={self.coeff!r})"


class DiffusionTerm(_UnaryTerm):
    """div(D grad phi) with a constant D (FiPy's second-order ``DiffusionTerm``)."""

    _kind = "diffusion"

    def __init__(self, coeff=1.0, var=None):
        if isinstance(coeff, (tuple, list)):
            if len(coeff) != 1:
                raise NotImplementedError("higher-order diffusion (several coefficients) is not supported")
            coeff = coeff[0]
        self._scalar_coeff = _require_scalar(coeff, "DiffusionTerm")
        super().__init__(coeff, var)


ImplicitDiffusionTerm = DiffusionTerm


class TransientTerm(_UnaryTerm):
    """c d(phi)/dt, discretized as c V (phi - phi_old) / dt (backward Euler)."""

    _kind = "transient"

    def __init__(self, coeff=1.0, var=None):
        self._scalar_coeff = _require_scalar(coeff, "TransientTerm")
        super().__init__(coeff, var)


class ImplicitSourceTerm(_UnaryTerm):
    """S phi with S a number or one value per cell (CellVariable, numpy or Warp array).

    As in FiPy, the coefficient goes on the matrix diagonal only in the cells where it makes
    the diagonal stronger (a sink in a diffusion or transient equation); elsewhere it is
    evaluated with the current values of the unknown and moved to the right-hand side.
    """

    _kind = "implicit"


class _ExplicitSourceTerm(_UnaryTerm):
    _kind = "explicit"

    def __repr__(self) -> str:
        return f"source({self.coeff!r})"


class _Equation(Term):
    def __init__(self, items):
        self._list = list(items)

    def _items(self) -> list:
        return list(self._list)

    def __repr__(self) -> str:
        return " + ".join(f"{scale:g}*{term!r}" for scale, term in self._list) or "0"
