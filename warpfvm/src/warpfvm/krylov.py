"""Preconditioned conjugate gradient on the device.

Why not ``warp.optim.linear.cg``: on the CPU its tiled dot product costs about a thousand
times a native reduction (53 ms against 0.05 ms for 32^3 values with warp-lang 1.17 on an
Apple M-series CPU), which makes every iteration ~160 ms. This loop uses ``array_inner`` (a
native reduction on both CPU and CUDA) with a device output, and keeps alpha and beta on the
device: between two convergence checks there is no host synchronization.

Per iteration: one stencil product, two inner products, two fused update kernels. Every
``check_every`` iterations the squared residual is read on the host. When the recursive
residual says "converged", the true residual ``rhs - A x`` is recomputed; if it is above the
threshold (drift of the recursion), CG restarts from the current iterate, at most
``max_restarts`` times.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import warp as wp

from warpfvm import kernels


@dataclass(frozen=True)
class KrylovResult:
    iterations: int
    residual: float  # true ||rhs - A x|| at exit (scaled system)
    rhs_norm: float  # ||rhs|| (scaled system)
    threshold: float  # absolute threshold applied to ||rhs - A x|| (scaled system)
    converged: bool
    restarts: int


class _Workspace:
    """Solver vectors and device scalars, reused while the size, device and precision match."""

    def __init__(self, n, dtype, device):
        self.key = (n, dtype, device.alias)
        self.r = wp.empty(n, dtype=dtype, device=device)
        self.z = wp.empty(n, dtype=dtype, device=device)
        self.p = wp.empty(n, dtype=dtype, device=device)
        self.q = wp.empty(n, dtype=dtype, device=device)
        self.scalars = wp.zeros(4, dtype=dtype, device=device)
        # rz ping-pong (two slots), p.Ap, r.r: one-element views that array_inner writes into
        self.rz = (self.scalars[0:1], self.scalars[1:2])
        self.pq = self.scalars[2:3]
        self.rr = self.scalars[3:4]


_WORKSPACES: dict = {}


def _workspace(n, dtype, device) -> _Workspace:
    key = (n, dtype, device.alias)
    if key not in _WORKSPACES:
        if len(_WORKSPACES) >= 8:
            _WORKSPACES.pop(next(iter(_WORKSPACES)))
        _WORKSPACES[key] = _Workspace(n, dtype, device)
    return _WORKSPACES[key]


def _norm(a) -> float:
    return math.sqrt(max(float(wp.utils.array_inner(a, a)), 0.0))


def pcg(
    system,
    x: wp.array,
    rtol: float,
    atol: float,
    maxiter: int,
    precondition: bool = True,
    check_every: int = 10,
    max_restarts: int = 3,
    use_cuda_graph: bool = False,
    start_threshold: float | None = None,
) -> KrylovResult:
    """Solve ``system.A x = system.rhs`` in place, starting from ``x``.

    Stops when ``||rhs - A x|| <= max(rtol ||rhs||, atol)`` (true residual) or after
    ``maxiter`` iterations. With ``start_threshold`` (absolute, scaled system), a starting
    residual at or below it returns at once, untouched: FiPy's LU solver does exactly that.
    """
    n, dtype, device = system.size, system.dtype, system.device
    k = kernels.get(dtype)
    ws = _workspace(n, dtype, device)
    r, z, p, q = ws.r, ws.z, ws.p, ws.q
    use_precond = 1 if precondition else 0
    inv_diag = system.inv_diag

    rhs_norm = _norm(system.rhs)
    threshold = max(rtol * rhs_norm, atol, 0.0)
    threshold_sq = threshold * threshold
    check_every = max(2, int(check_every) + (int(check_every) % 2))  # even: the rz slots realign

    def start():
        system.matvec(x, system.rhs, r, -1.0, 1.0)
        wp.launch(k.cg_start, dim=n, inputs=[r, inv_diag, use_precond, z, p], device=device)
        wp.utils.array_inner(r, z, out=ws.rz[0])

    def iteration(parity):
        rz_old, rz_new = ws.rz[parity], ws.rz[1 - parity]
        system.matvec(p, q, q, 1.0, 0.0)
        wp.utils.array_inner(p, q, out=ws.pq)
        wp.launch(k.cg_update_solution, dim=n, inputs=[rz_old, ws.pq, x, r, p, q, inv_diag, use_precond, z], device=device)
        wp.utils.array_inner(r, z, out=rz_new)
        wp.launch(k.cg_update_direction, dim=n, inputs=[rz_new, rz_old, z, p], device=device)

    def block():
        for step in range(check_every):
            iteration(step % 2)

    graph = None
    iterations = restarts = 0
    residual = math.inf
    converged = False
    while True:
        start()
        recursive_sq = float(wp.utils.array_inner(r, r))
        if start_threshold is not None and iterations == 0 and restarts == 0 and recursive_sq <= start_threshold**2:
            return KrylovResult(0, math.sqrt(recursive_sq), rhs_norm, start_threshold, True, 0)
        while recursive_sq > threshold_sq and iterations < maxiter:
            remaining = maxiter - iterations
            if remaining >= check_every:
                if use_cuda_graph and device.is_cuda:
                    if graph is None:
                        block()  # run eagerly once, so every module is loaded before the capture
                        with wp.ScopedCapture(device=device, force_module_load=False) as capture:
                            block()  # recorded, not executed
                        graph = capture.graph
                    else:
                        wp.capture_launch(graph)
                else:
                    block()
                iterations += check_every
            else:
                for step in range(remaining):
                    iteration(step % 2)
                iterations += remaining
                if remaining % 2:  # keep r.z in slot 0 for a possible restart
                    wp.copy(ws.rz[0], ws.rz[1])
            wp.utils.array_inner(r, r, out=ws.rr)
            recursive_sq = float(ws.rr.numpy()[0])
        residual = _norm(system.residual(x, out=q))
        converged = residual <= threshold * (1.0 + 1e-12)
        if converged or iterations >= maxiter or restarts >= max_restarts:
            break
        restarts += 1
    return KrylovResult(iterations, residual, rhs_norm, threshold, converged, restarts)
