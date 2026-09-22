"""Per-kernel timing of the contact-mechanics substep at several population sizes.

Measures separately: hash-grid build, contact kernel, position integration, and the whole
substep. Every record carries the device, so CPU numbers can never be mistaken for GPU ones.

    python benchmarks/bench_mechanics.py --device cpu --sizes 1000 10000 100000
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
from warpbiocell.cells.mechanics import ContactParams, contact_substep, make_neighbor_grid
from warpbiocell.cells.state import CellPopulation
from warpbiocell.device import resolve_device, synchronize
from warpbiocell.kernels.mechanics_kernels import contact_velocities, integrate_positions

RADIUS_UM = 8.0
DT_H = 0.005


def _timed(fn, device, repeats):
    times = []
    for _ in range(repeats):
        synchronize(device)
        t0 = time.perf_counter()
        fn()
        synchronize(device)
        times.append(time.perf_counter() - t0)
    return float(np.median(times)), float(np.min(times))


def bench_size(n, device, substeps, warmup):
    pos = spherical_cluster(n, RADIUS_UM, spacing_factor=0.9, jitter=0.1, seed=0)
    pop = CellPopulation.from_numpy(pos, np.full(n, RADIUS_UM), device=device)
    params = ContactParams(stiffness=10.0, damping=1.0, max_radius=RADIUS_UM, margin=0.5)
    grid = make_neighbor_grid(params, device)

    for _ in range(warmup):
        contact_substep(pop, grid, params, DT_H)

    def build():
        grid.build(pop)

    def contacts():
        wp.launch(
            contact_velocities,
            dim=n,
            inputs=[grid.id, pop.position, pop.radius, params.query_radius, params.rate],
            outputs=[pop.velocity, pop.contact_count],
            device=device,
        )

    def integrate():
        wp.launch(integrate_positions, dim=n, inputs=[pop.position, pop.velocity, DT_H], device=device)

    def full():
        contact_substep(pop, grid, params, DT_H)

    build_med, _ = _timed(build, device, substeps)
    contact_med, _ = _timed(contacts, device, substeps)
    integrate_med, _ = _timed(integrate, device, substeps)
    full_med, full_min = _timed(full, device, substeps)

    return {
        "cells": n,
        "mean_contacts_per_cell": float(pop.contact_counts_numpy().mean()),
        "grid_build_ms": build_med * 1e3,
        "contact_kernel_ms": contact_med * 1e3,
        "integrate_ms": integrate_med * 1e3,
        "substep_ms_median": full_med * 1e3,
        "substep_ms_min": full_min * 1e3,
        "cells_per_second": n / full_med,
        "substeps_per_second": 1.0 / full_med,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--sizes", type=int, nargs="+", default=[1_000, 10_000, 100_000])
    parser.add_argument("--substeps", type=int, default=20, help="timed repetitions per measurement")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--out", type=Path, default=None, help="JSON output path (default: benchmarks/results/)")
    args = parser.parse_args()

    wp.config.quiet = True
    device = resolve_device(args.device)

    records = [bench_size(n, device, args.substeps, args.warmup) for n in args.sizes]

    header = f"{'cells':>8} {'contacts':>9} {'build ms':>9} {'contact ms':>11} {'integ ms':>9} {'substep ms':>11} {'cells/s':>12}"
    print(f"device={device.alias} ({device.name})  warp={wp.__version__}")
    print(header)
    for r in records:
        print(
            f"{r['cells']:>8} {r['mean_contacts_per_cell']:>9.2f} {r['grid_build_ms']:>9.2f} "
            f"{r['contact_kernel_ms']:>11.2f} {r['integrate_ms']:>9.3f} {r['substep_ms_median']:>11.2f} "
            f"{r['cells_per_second']:>12.3e}"
        )

    result = {
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "warpbiocell_version": warpbiocell.__version__,
        "warp_version": wp.__version__,
        "device": {"alias": device.alias, "name": device.name, "is_cuda": device.is_cuda},
        "platform": platform.platform(),
        "python": platform.python_version(),
        "parameters": {
            "radius_um": RADIUS_UM,
            "dt_h": DT_H,
            "spacing_factor": 0.9,
            "timed_repetitions": args.substeps,
            "warmup": args.warmup,
        },
        "records": records,
    }
    out = args.out or Path(__file__).parent / "results" / f"mechanics_{device.alias.replace(':', '')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"written {out}")


if __name__ == "__main__":
    main()
