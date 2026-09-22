"""Timing of the lifecycle step and of a full cell step at several population sizes.

Measures separately: the decision kernel, the prefix scan plus its host read (the one sync
per step), daughter placement with every cell dividing (worst case), a realistic
``lifecycle_step`` (~1% of cells dividing) and a full ``cell_step`` with 10 mechanics
substeps. Every record carries the device.

    python benchmarks/bench_lifecycle.py --device cpu --sizes 1000 10000 100000
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
import warp.utils

import warpbiocell
from warpbiocell.cells.initialization import spherical_cluster
from warpbiocell.cells.lifecycle import LifecycleParams, lifecycle_step
from warpbiocell.cells.mechanics import ContactParams, contact_substep, make_neighbor_grid
from warpbiocell.cells.state import CellPopulation
from warpbiocell.device import resolve_device, synchronize
from warpbiocell.kernels.cell_kernels import lifecycle_decide, place_daughters
from warpbiocell.simulation.simulator import TimeStepping, cell_step

RADIUS_UM = 8.0
STEPPING = TimeStepping(dt_cells=0.25, dt_mechanics=0.025)
REALISTIC = LifecycleParams(division_rate=0.04, death_rate=0.001, inhibition_threshold=10**6)  # p ~ 1%/step
CERTAIN = LifecycleParams(division_rate=1.0e3, death_rate=0.0, inhibition_threshold=10**6)


def _timed(fn, device, repeats):
    times = []
    for _ in range(repeats):
        synchronize(device)
        t0 = time.perf_counter()
        fn()
        synchronize(device)
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def _fresh(n, device, capacity):
    pos = spherical_cluster(n, RADIUS_UM, spacing_factor=1.0, jitter=0.1, seed=0)
    return CellPopulation.from_numpy(pos, np.full(n, RADIUS_UM), capacity=capacity, device=device, seed=0)


def bench_size(n, device, repeats, warmup):
    contact = ContactParams(stiffness=10.0, damping=1.0, max_radius=RADIUS_UM, margin=2.0)
    grid = make_neighbor_grid(contact, device)

    # Decision kernel alone (state unchanged between repeats apart from ages and RNG draws).
    pop = _fresh(n, device, capacity=4 * n)
    contact_substep(pop, grid, contact, 0.0)

    def decide():
        wp.launch(
            lifecycle_decide,
            dim=n,
            inputs=[
                0.25, 0.0, 0.0, 0.0, 10**6, 0.0, 0.0, 0.0, 0.0, 0,
                pop.cell_state, pop.age, pop.neighbor_count, pop.oxygen_local, pop.glucose_local, pop.rng_state,
            ],
            outputs=[pop.divide_flag],
            device=device,
        )

    def scan_and_read():
        warp.utils.array_scan(pop.divide_flag[:n], pop.division_offset[:n], inclusive=True)
        return int(pop.division_offset[n - 1 : n].numpy()[0])

    for _ in range(warmup):
        decide()
        scan_and_read()
    decide_ms = _timed(decide, device, repeats) * 1e3
    scan_ms = _timed(scan_and_read, device, repeats) * 1e3

    # Placement with every cell dividing; count is not advanced so slots [n, 2n) are rewritten.
    pop.divide_flag[:n].fill_(1)
    warp.utils.array_scan(pop.divide_flag[:n], pop.division_offset[:n], inclusive=True)

    def place_all():
        wp.launch(
            place_daughters,
            dim=n,
            inputs=[
                n, 1.0, pop.divide_flag, pop.division_offset, pop.position, pop.radius, pop.cell_state,
                pop.cell_type, pop.age, pop.oxygen_local, pop.glucose_local, pop.rng_state, pop.velocity, pop.neighbor_count,
            ],
            device=device,
        )

    for _ in range(warmup):
        place_all()
    place_ms = _timed(place_all, device, repeats) * 1e3

    # Realistic lifecycle step and full cell step on fresh populations (count grows slowly).
    pop = _fresh(n, device, capacity=4 * n)
    contact_substep(pop, grid, contact, 0.0)
    for _ in range(warmup):
        lifecycle_step(pop, REALISTIC, STEPPING.dt_cells)
    lifecycle_ms = _timed(lambda: lifecycle_step(pop, REALISTIC, STEPPING.dt_cells), device, repeats) * 1e3

    pop = _fresh(n, device, capacity=4 * n)
    for _ in range(warmup):
        cell_step(pop, grid, REALISTIC, contact, STEPPING)
    step_ms = _timed(lambda: cell_step(pop, grid, REALISTIC, contact, STEPPING), device, repeats) * 1e3

    return {
        "cells": n,
        "decide_kernel_ms": decide_ms,
        "scan_and_host_read_ms": scan_ms,
        "place_all_daughters_ms": place_ms,
        "lifecycle_step_ms": lifecycle_ms,
        "cell_step_ms": step_ms,
        "mechanics_substeps_per_cell_step": STEPPING.mechanics_substeps,
        "cell_steps_per_second": 1e3 / step_ms,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--sizes", type=int, nargs="+", default=[1_000, 10_000, 100_000])
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    wp.config.quiet = True
    device = resolve_device(args.device)

    records = [bench_size(n, device, args.repeats, args.warmup) for n in args.sizes]

    print(f"device={device.alias} ({device.name})  warp={wp.__version__}")
    print(f"{'cells':>8} {'decide ms':>10} {'scan+read ms':>13} {'place-all ms':>13} {'lifecycle ms':>13} {'cell step ms':>13}")
    for r in records:
        print(
            f"{r['cells']:>8} {r['decide_kernel_ms']:>10.3f} {r['scan_and_host_read_ms']:>13.3f} "
            f"{r['place_all_daughters_ms']:>13.3f} {r['lifecycle_step_ms']:>13.3f} {r['cell_step_ms']:>13.2f}"
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
            "dt_cells_h": STEPPING.dt_cells,
            "dt_mechanics_h": STEPPING.dt_mechanics,
            "realistic_division_rate": REALISTIC.division_rate,
            "repeats": args.repeats,
            "warmup": args.warmup,
        },
        "records": records,
    }
    out = args.out or Path(__file__).parent / "results" / f"lifecycle_{device.alias.replace(':', '')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"written {out}")


if __name__ == "__main__":
    main()
