"""Warp kernels for the 7-point finite-volume stencil, generated once per scalar type.

The linear system is stored as a diagonal ``diag[N]``, a right-hand side ``rhs[N]`` and one
off-diagonal coefficient per axis (``tx, ty, tz``): with a scalar diffusion coefficient on a
uniform grid every interior face of an axis has the same transmissibility, so the matrix never
exists in memory. Boundary conditions only change ``diag`` and ``rhs``.

Kernels over cells are launched with ``dim = (nz, ny, nx)``: consecutive threads handle
consecutive cells along x, which are consecutive in memory (FiPy's numbering).
"""

# No "from __future__ import annotations" here: Warp must see the closure type `scalar` in the
# kernel annotations as an object, not as a string it cannot resolve.
import functools
from types import SimpleNamespace

import warp as wp


@functools.cache
def get(scalar) -> SimpleNamespace:
    """The kernels for ``scalar`` = ``wp.float64`` or ``wp.float32``."""

    @wp.kernel(module="unique")
    def assemble_base(
        nx: int,
        ny: int,
        nz: int,
        dim: int,
        tx: scalar,  # off-diagonal of A across an x face (a boundary face counts twice: d/2)
        ty: scalar,
        tz: scalar,
        off_xmin: int,
        off_xmax: int,
        off_ymin: int,
        off_ymax: int,
        off_zmin: int,
        off_zmax: int,
        bc_fixed: wp.array(dtype=wp.int32),  # compact exterior faces: 1 where the value is fixed
        bc_value: wp.array(dtype=scalar),
        has_transient: int,
        tc: scalar,  # diagonal of A from the transient term, c V / dt (scaled)
        old: wp.array(dtype=scalar),
        diag: wp.array(dtype=scalar),
        rhs: wp.array(dtype=scalar),
    ):
        k, j, i = wp.tid()
        p = i + nx * (j + ny * k)
        two = scalar(2.0)
        d = scalar(0.0)
        r = scalar(0.0)

        if i > 0:
            d -= tx
        else:
            e = off_xmin + j + ny * k
            if bc_fixed[e] != 0:
                d -= two * tx
                r -= two * tx * bc_value[e]
        if i < nx - 1:
            d -= tx
        else:
            e = off_xmax + j + ny * k
            if bc_fixed[e] != 0:
                d -= two * tx
                r -= two * tx * bc_value[e]

        if dim >= 2:
            if j > 0:
                d -= ty
            else:
                e = off_ymin + i + nx * k
                if bc_fixed[e] != 0:
                    d -= two * ty
                    r -= two * ty * bc_value[e]
            if j < ny - 1:
                d -= ty
            else:
                e = off_ymax + i + nx * k
                if bc_fixed[e] != 0:
                    d -= two * ty
                    r -= two * ty * bc_value[e]

        if dim >= 3:
            if k > 0:
                d -= tz
            else:
                e = off_zmin + i + nx * j
                if bc_fixed[e] != 0:
                    d -= two * tz
                    r -= two * tz * bc_value[e]
            if k < nz - 1:
                d -= tz
            else:
                e = off_zmax + i + nx * j
                if bc_fixed[e] != 0:
                    d -= two * tz
                    r -= two * tz * bc_value[e]

        if has_transient != 0:
            d += tc
            r += tc * old[p]

        diag[p] = d
        rhs[p] = r

    @wp.kernel(module="unique")
    def add_source(
        values: wp.array(dtype=scalar),
        use_array: int,
        value: scalar,
        factor: scalar,  # (diagonal sign / reference) * term scale * cell volume
        implicit: int,
        x: wp.array(dtype=scalar),
        diag: wp.array(dtype=scalar),
        rhs: wp.array(dtype=scalar),
    ):
        """Explicit source s: rhs -= factor s. Implicit source S (FiPy's rule, per cell): on the
        diagonal where it reinforces it (factor S >= 0), otherwise lagged, rhs -= factor S x."""
        p = wp.tid()
        s = value
        if use_array != 0:
            s = values[p]
        sv = factor * s
        if implicit != 0:
            if sv >= scalar(0.0):
                diag[p] = diag[p] + sv
            else:
                rhs[p] = rhs[p] - sv * x[p]
        else:
            rhs[p] = rhs[p] - sv

    @wp.kernel(module="unique")
    def under_relax(
        diag: wp.array(dtype=scalar),
        rhs: wp.array(dtype=scalar),
        x: wp.array(dtype=scalar),
        relaxation: scalar,
    ):
        """FiPy's under-relaxation: diag /= w, rhs += (1 - w) diag x."""
        p = wp.tid()
        d = diag[p] / relaxation
        diag[p] = d
        rhs[p] = rhs[p] + (scalar(1.0) - relaxation) * d * x[p]

    @wp.kernel(module="unique")
    def invert_diagonal(diag: wp.array(dtype=scalar), inv_diag: wp.array(dtype=scalar)):
        p = wp.tid()
        d = diag[p]
        if d != scalar(0.0):
            inv_diag[p] = scalar(1.0) / d
        else:
            inv_diag[p] = scalar(1.0)

    @wp.kernel(module="unique")
    def stencil_matvec(
        nx: int,
        ny: int,
        nz: int,
        diag: wp.array(dtype=scalar),
        tx: scalar,
        ty: scalar,
        tz: scalar,
        x: wp.array(dtype=scalar),
        y: wp.array(dtype=scalar),
        z: wp.array(dtype=scalar),
        alpha: scalar,
        beta: scalar,
    ):
        """z = alpha A x + beta y (y and z may alias; x must not alias z)."""
        k, j, i = wp.tid()
        p = i + nx * (j + ny * k)
        acc = diag[p] * x[p]
        if i > 0:
            acc += tx * x[p - 1]
        if i < nx - 1:
            acc += tx * x[p + 1]
        if j > 0:
            acc += ty * x[p - nx]
        if j < ny - 1:
            acc += ty * x[p + nx]
        if k > 0:
            acc += tz * x[p - nx * ny]
        if k < nz - 1:
            acc += tz * x[p + nx * ny]
        out = alpha * acc
        if beta != scalar(0.0):
            out += beta * y[p]
        z[p] = out

    @wp.kernel(module="unique")
    def cg_start(
        r: wp.array(dtype=scalar),
        inv_diag: wp.array(dtype=scalar),
        precondition: int,
        z: wp.array(dtype=scalar),
        p: wp.array(dtype=scalar),
    ):
        """z = M r (Jacobi, or z = r), p = z."""
        i = wp.tid()
        zi = r[i]
        if precondition != 0:
            zi = zi * inv_diag[i]
        z[i] = zi
        p[i] = zi

    @wp.kernel(module="unique")
    def cg_update_solution(
        rz: wp.array(dtype=scalar),
        pq: wp.array(dtype=scalar),
        x: wp.array(dtype=scalar),
        r: wp.array(dtype=scalar),
        p: wp.array(dtype=scalar),
        q: wp.array(dtype=scalar),
        inv_diag: wp.array(dtype=scalar),
        precondition: int,
        z: wp.array(dtype=scalar),
    ):
        """alpha = (r.z) / (p.Ap); x += alpha p; r -= alpha Ap; z = M r. Scalars stay on the device."""
        i = wp.tid()
        alpha = scalar(0.0)
        if pq[0] != scalar(0.0):
            alpha = rz[0] / pq[0]
        x[i] = x[i] + alpha * p[i]
        ri = r[i] - alpha * q[i]
        r[i] = ri
        if precondition != 0:
            z[i] = ri * inv_diag[i]
        else:
            z[i] = ri

    @wp.kernel(module="unique")
    def cg_update_direction(
        rz_new: wp.array(dtype=scalar),
        rz_old: wp.array(dtype=scalar),
        z: wp.array(dtype=scalar),
        p: wp.array(dtype=scalar),
    ):
        """beta = (r.z)_new / (r.z)_old; p = z + beta p."""
        i = wp.tid()
        beta = scalar(0.0)
        if rz_old[0] != scalar(0.0):
            beta = rz_new[0] / rz_old[0]
        p[i] = z[i] + beta * p[i]

    @wp.kernel(module="unique")
    def add_constant(values: wp.array(dtype=scalar), shift: scalar):
        i = wp.tid()
        values[i] = values[i] + shift

    return SimpleNamespace(
        assemble_base=assemble_base,
        add_source=add_source,
        under_relax=under_relax,
        invert_diagonal=invert_diagonal,
        stencil_matvec=stencil_matvec,
        cg_start=cg_start,
        cg_update_solution=cg_update_solution,
        cg_update_direction=cg_update_direction,
        add_constant=add_constant,
    )
