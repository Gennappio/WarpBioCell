"""Timing of the oxygen field operations at several grid sizes and cell counts.

Measures separately: cell -> grid deposit, one SOR sweep (two colour passes), one residual
check (includes its host sync), grid -> cell sampling, and a full warm-started
``OxygenField.update`` on a static spheroid (reports the sweeps it needed). Every record
carries the device.

    python benchmarks/bench_field.py --device cpu --nodes 41 81 --cells 10000 100000
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import time
from pathlib import Path

import numpy as np
import warp as wp

import warpbiocell
from warpbiocell.cells.initialization import spherical_cluster
from warpbiocell.cells.state import CellPopulation
from warpbiocell.device import resolve_device, synchronize
from warpbiocell.fields.diffusion import SolverSettings
from warpbiocell.fields.oxygen import OxygenField, OxygenParams
from warpbiocell.fields.scalar_field import GridGeometry
from warpbiocell.kernels.field_kernels import sor_sweep, steady_state_residual

RADIUS_UM = 8.0
DX_UM = 20.0


def _timed(fn, device, repeats):
    times = []
    for _ in range(repeats):
        synchronize(device)
        t0 = time.perf_counter()
        fn()
        synchronize(device)
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def bench_grid(nodes, n_cells, device, repeats, warmup):
    geom = GridGeometry(origin=(-0.5 * (nodes - 1) * DX_UM,) * 3, dx=DX_UM, shape=(nodes,) * 3)
    params = OxygenParams()
    ox = OxygenField(geom, params, SolverSettings(omega=1.8, tolerance=1e-5, check_every=10), device=device)
    pos = spherical_cluster(n_cells, RADIUS_UM, spacing_factor=1.0, jitter=0.1, seed=0)
    pop = CellPopulation.from_numpy(pos, np.full(n_cells, RADIUS_UM), device=device)
    kin = params.kinetics
    dx2_over_D = geom.dx**2 / kin.diffusion
    bc = ox.field.boundary_flags

    def deposit():
        ox.deposit_uptake(pop)

    def sweep():
        for color in (0, 1):
            wp.launch(sor_sweep, dim=geom.shape, inputs=[color, 1.8, dx2_over_D, kin.uptake_max, kin.michaelis_k, *bc, ox.field.fixed, ox.density, ox.production, 0, ox.values, 1.0, ox.values], device=device)

    def residual():
        ox._residual.zero_()
        wp.launch(steady_state_residual, dim=geom.shape, inputs=[dx2_over_D, kin.uptake_max, kin.michaelis_k, params.boundary_value, *bc, ox.field.fixed, ox.density, ox.production, 0, ox.values, 1.0, ox.values, ox._residual], device=device)
        return float(ox._residual.numpy()[0])

    def sample():
        ox.sample_at_cells(pop)

    for _ in range(warmup):
        deposit()
        sweep()
        residual()
        sample()
    deposit_ms = _timed(deposit, device, repeats) * 1e3
    sweep_ms = _timed(sweep, device, repeats) * 1e3
    residual_ms = _timed(residual, device, repeats) * 1e3
    sample_ms = _timed(sample, device, repeats) * 1e3

    cold = ox.update(pop)  # from the boundary-value initial field
    update_ms = _timed(lambda: ox.update(pop), device, repeats) * 1e3
    warm = ox.last_report

    return {
        "nodes_per_axis": nodes,
        "nodes": geom.node_count,
        "cells": n_cells,
        "deposit_ms": deposit_ms,
        "sor_sweep_ms": sweep_ms,
        "residual_check_ms": residual_ms,
        "sample_ms": sample_ms,
        "cold_solve_sweeps": cold.sweeps,
        "warm_update_ms": update_ms,
        "warm_update_sweeps": warm.sweeps,
        "nodes_per_second_per_sweep": geom.node_count / (sweep_ms * 1e-3),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--nodes", type=int, nargs="+", default=[41, 81], help="nodes per axis (dx = 20 um)")
    parser.add_argument("--cells", type=int, nargs="+", default=[10_000, 100_000])
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    wp.config.quiet = True
    device = resolve_device(args.device)

    records = [bench_grid(n, c, device, args.repeats, args.warmup) for n in args.nodes for c in args.cells]

    print(f"device={device.alias} ({device.name})  warp={wp.__version__}")
    print(f"{'nodes':>7} {'cells':>7} {'deposit ms':>11} {'sweep ms':>9} {'resid ms':>9} {'sample ms':>10} {'cold sw':>8} {'warm ms':>8} {'warm sw':>8}")
    for r in records:
        print(
            f"{r['nodes_per_axis']:>4}^3 {r['cells']:>7} {r['deposit_ms']:>11.3f} {r['sor_sweep_ms']:>9.3f} {r['residual_check_ms']:>9.3f} "
            f"{r['sample_ms']:>10.3f} {r['cold_solve_sweeps']:>8} {r['warm_update_ms']:>8.1f} {r['warm_update_sweeps']:>8}"
        )

    result = {
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "warpbiocell_version": warpbiocell.__version__,
        "warp_version": wp.__version__,
        "device": {"alias": device.alias, "name": device.name, "is_cuda": device.is_cuda},
        "platform": platform.platform(),
        "python": platform.python_version(),
        "parameters": {"radius_um": RADIUS_UM, "dx_um": DX_UM, "omega": 1.8, "tolerance": 1e-5, "repeats": args.repeats, "warmup": args.warmup},
        "records": records,
    }
    out = args.out or Path(__file__).parent / "results" / f"field_{device.alias.replace(':', '')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"written {out}")


if __name__ == "__main__":
    main()
