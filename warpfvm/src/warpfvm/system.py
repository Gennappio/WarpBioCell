"""Assembly of a FiPy-style equation into a scaled 7-point system ``A x = rhs`` on the device.

FiPy writes an equation as a sum of terms equal to zero and discretizes it as ``L phi = b``
(fipy/terms). Per cell P of volume V, with T_f = D A_f / d_f the transmissibility of face f
(d_f = centre-to-centre distance, or centre-to-face = d/2 on the boundary):

    DiffusionTerm(D)          L_PP -= sum_f T_f         L_PN += T_f (interior faces)
                              Dirichlet face:           b    -= T_f * value
    TransientTerm(c)          L_PP += c V / dt          b    += c V old_P / dt
    ImplicitSourceTerm(S)     L_PP += S_P V             where s * S_P >= 0, otherwise lagged:
                                                        b    -= S_P V phi_P
    explicit source q         b    -= q_P V

s is FiPy's "diagonal sign" of the whole equation (nonDiffusionTerm._getDiagonalSign): +1 when
it has a positive transient coefficient, otherwise -1 when its diffusion coefficient is
positive. Unconstrained boundary faces carry no flux.

warpfvm solves the same system multiplied by s / a_ref, where a_ref is the diagonal of an
interior cell (sum over the axes of 2 |D| A/d, plus |c| V/dt). The matrix then has a positive
diagonal of order one (it is symmetric positive definite for every well-posed MicroC problem,
so CG applies), and the numbers stay far from the float32 underflow that SI units would cause
in the solver's dot products (D ~ 1e-9 m^2/s, V ~ 1e-13 m^3 give T ~ 1e-14).
``LinearSystem.scale`` converts back: ``L = scale * A`` and ``b = scale * rhs``.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass

import numpy as np
import warp as wp
from warp.optim.linear import LinearOperator, aslinearoperator

from warpfvm import kernels
from warpfvm._config import numpy_dtype
from warpfvm.mesh import UniformGrid
from warpfvm.variables import CellVariable, _ScaledVariable


@dataclass
class LinearSystem:
    """``A x = rhs`` with ``A = diag + tx (x neighbours) + ty (y) + tz (z)``, on the device."""

    mesh: UniformGrid
    diag: wp.array
    rhs: wp.array
    inv_diag: wp.array
    tx: float
    ty: float
    tz: float
    scale: float  # FiPy's matrix and right-hand side are scale * A and scale * rhs
    singular: bool = False  # no fixed value, sink or transient term: solutions differ by a constant

    @property
    def device(self):
        return self.diag.device

    @property
    def dtype(self):
        return self.diag.dtype

    @property
    def size(self) -> int:
        return self.diag.shape[0]

    def matvec(self, x, y, z, alpha, beta) -> None:
        """z = alpha A x + beta y, one kernel launch (CUDA-graph capturable)."""
        mesh, s = self.mesh, self.dtype
        wp.launch(
            kernels.get(s).stencil_matvec,
            dim=(mesh.nz, mesh.ny, mesh.nx),
            inputs=[mesh.nx, mesh.ny, mesh.nz, self.diag, s(self.tx), s(self.ty), s(self.tz), x, y, z, s(alpha), s(beta)],
            device=self.device,
        )

    def operator(self) -> LinearOperator:
        return LinearOperator((self.size, self.size), self.dtype, self.device, self.matvec)

    def jacobi(self) -> LinearOperator:
        return aslinearoperator(self.inv_diag)

    def residual(self, x: wp.array, out: wp.array | None = None) -> wp.array:
        """rhs - A x."""
        if out is None:
            out = wp.empty_like(self.rhs)
        self.matvec(x, self.rhs, out, -1.0, 1.0)
        return out

    def residual_norm(self, x: wp.array) -> float:
        r = self.residual(x)
        return math.sqrt(max(float(wp.utils.array_inner(r, r)), 0.0))

    def rhs_norm(self) -> float:
        return math.sqrt(max(float(wp.utils.array_inner(self.rhs, self.rhs)), 0.0))

    def to_scipy(self):
        """FiPy's ``(L, b)`` on the host: a scipy CSR matrix and a numpy vector (for checks)."""
        import scipy.sparse as sp

        mesh, n = self.mesh, self.size
        ids = np.arange(n).reshape(mesh.nz, mesh.ny, mesh.nx)
        rows, cols, vals = [np.arange(n)], [np.arange(n)], [self.diag.numpy().astype(np.float64)]
        for coefficient, axis in ((self.tx, 2), (self.ty, 1), (self.tz, 0)):
            count = ids.shape[axis]
            if count < 2 or coefficient == 0.0:
                continue
            lower = np.take(ids, np.arange(count - 1), axis=axis).ravel()
            upper = np.take(ids, np.arange(1, count), axis=axis).ravel()
            rows += [lower, upper]
            cols += [upper, lower]
            vals += [np.full(lower.size, coefficient)] * 2
        matrix = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
        return self.scale * matrix, self.scale * self.rhs.numpy().astype(np.float64)


def diagonal_sign(has_diffusion: bool, diffusion: float, has_transient: bool, transient: float) -> int:
    """FiPy's ``_getDiagonalSign`` for scalar coefficients (fipy/terms/nonDiffusionTerm.py)."""
    diffusion_sign = 1 if diffusion <= 0.0 else -1
    transient_sign = 1 if transient >= 0.0 else -1
    if has_transient and has_diffusion:
        return diffusion_sign if transient == 0.0 else transient_sign
    if has_transient:
        return transient_sign
    if has_diffusion:
        return diffusion_sign
    return 1


def _dummy(mesh: UniformGrid) -> wp.array:
    key = ("dummy", mesh.device.alias, mesh.dtype.__name__)
    if key not in mesh._topology:
        mesh._topology[key] = wp.zeros(1, dtype=mesh.dtype, device=mesh.device)
    return mesh._topology[key]


def _coefficient(coeff, mesh: UniformGrid):
    """(use_array, scalar value, per-cell array, extra scale) for a source coefficient."""
    if isinstance(coeff, numbers.Real) or (isinstance(coeff, np.ndarray) and coeff.ndim == 0):
        return 0, float(coeff), _dummy(mesh), 1.0
    if isinstance(coeff, _ScaledVariable):
        use_array, value, array, extra = _coefficient(coeff.var, mesh)
        return use_array, value, array, extra * coeff.scale
    if isinstance(coeff, CellVariable):
        if not mesh._same_cells(coeff.mesh):
            raise ValueError(f"coefficient {coeff.name!r} lives on a different mesh, device or precision")
        return 1, 0.0, coeff._array, 1.0
    n = mesh.numberOfCells
    if isinstance(coeff, wp.array):
        if coeff.shape == (n,) and coeff.device == mesh.device and coeff.dtype == mesh.dtype:
            return 1, 0.0, coeff, 1.0
        coeff = coeff.numpy()
    host = np.asarray(coeff, dtype=np.float64).reshape(-1)
    if host.size != n:
        raise ValueError(f"a per-cell coefficient needs {n} values, got {host.size}")
    array = wp.array(host.astype(numpy_dtype(mesh.dtype)), dtype=mesh.dtype, device=mesh.device)
    return 1, 0.0, array, 1.0


def assemble(items, var: CellVariable, dt=None, underRelaxation=None) -> LinearSystem:
    """Build the system for the equation ``sum(scale * term) = 0`` in the unknown ``var``.

    ``items`` is a list of ``(scale, term)`` where ``term._kind`` is ``diffusion``,
    ``transient``, ``implicit`` or ``explicit`` (see warpfvm.terms). Implicit sources read the
    current ``var`` values where FiPy's rule lags them; the transient term reads ``var.old``.
    """
    mesh = var.mesh
    diffusion = transient = 0.0
    has_diffusion = has_transient = False
    sources = []
    for scale, term in items:
        kind = term._kind
        if kind == "diffusion":
            has_diffusion = True
            diffusion += scale * term._scalar_coeff
        elif kind == "transient":
            has_transient = True
            transient += scale * term._scalar_coeff
        elif kind in ("implicit", "explicit"):
            sources.append((kind == "implicit", scale, term.coeff))
        else:
            raise TypeError(f"unknown term kind {kind!r}")
    if has_transient:
        if dt is None:
            raise TypeError("`dt` must be specified.")
        if np.ndim(dt) != 0:
            raise TypeError("`dt` must be a single number")
        dt = float(dt)
        if not dt > 0.0:
            raise ValueError("`dt` must be positive")

    sign = diagonal_sign(has_diffusion, diffusion, has_transient, transient)
    volume = mesh._cell_volume
    areas, spacing = mesh._axis_areas, mesh._spacing
    reference = abs(diffusion) * sum(2.0 * areas[a] / spacing[a] for a in range(mesh.dim))
    if has_transient:
        reference += abs(transient) * volume / dt
    if not (reference > 0.0 and math.isfinite(reference)):
        reference = volume
    factor = sign / reference
    t = [factor * diffusion * areas[a] / spacing[a] if a < mesh.dim else 0.0 for a in range(3)]
    t_transient = factor * transient * volume / dt if has_transient else 0.0

    s, device, n = mesh.dtype, mesh.device, mesh.numberOfCells
    k = kernels.get(s)
    diag = wp.empty(n, dtype=s, device=device)
    rhs = wp.empty(n, dtype=s, device=device)
    fixed, values = var._boundary_arrays()
    wp.launch(
        k.assemble_base,
        dim=(mesh.nz, mesh.ny, mesh.nx),
        inputs=[
            mesh.nx, mesh.ny, mesh.nz, mesh.dim, s(t[0]), s(t[1]), s(t[2]), *mesh._side_offsets,
            fixed, values, int(has_transient), s(t_transient), var.old._array, diag, rhs,
        ],
        device=device,
    )
    for implicit, scale, coeff in sources:
        use_array, value, array, extra = _coefficient(coeff, mesh)
        wp.launch(
            k.add_source,
            dim=n,
            inputs=[array, use_array, s(value), s(factor * scale * extra * volume), int(implicit), var._array, diag, rhs],
            device=device,
        )
    if underRelaxation is not None:
        wp.launch(k.under_relax, dim=n, inputs=[diag, rhs, var._array, s(float(underRelaxation))], device=device)
    inv_diag = wp.empty(n, dtype=s, device=device)
    wp.launch(k.invert_diagonal, dim=n, inputs=[diag, inv_diag], device=device)
    singular = _is_singular(var, diffusion, has_transient and transient != 0.0, sources, factor, volume, underRelaxation)
    return LinearSystem(mesh, diag, rhs, inv_diag, t[0], t[1], t[2], scale=sign * reference, singular=singular)


def _is_singular(var, diffusion, transient, sources, factor, volume, underRelaxation) -> bool:
    """True for the all-Neumann steady operator: diffusion only on the diagonal, whose null space
    is the constants. Any fixed boundary value, transient term, under-relaxation or implicit sink
    that reaches the diagonal in at least one cell makes the matrix positive definite."""
    if diffusion == 0.0 or transient or (underRelaxation is not None and underRelaxation != 1.0):
        return False
    if any(constraint.mask.any() for constraint in var._constraints):
        return False
    for implicit, scale, coeff in sources:
        if not implicit:
            continue
        if isinstance(coeff, numbers.Real) or (isinstance(coeff, np.ndarray) and coeff.ndim == 0):
            values = np.asarray(float(coeff))
        else:
            values = np.asarray(coeff.value if hasattr(coeff, "value") else coeff.numpy() if isinstance(coeff, wp.array) else coeff, dtype=np.float64)
        if np.any(factor * scale * values * volume > 0.0):
            return False
    return True
