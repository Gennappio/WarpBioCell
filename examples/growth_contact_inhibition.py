"""Milestone 3 demonstrator: a cluster grows by stochastic division under contact inhibition.

No oxygen yet: cells divide at a constant rate unless crowded, dead cells (optional constant
death rate) stay in place, and contact mechanics pushes daughters apart. Reports the
population by state and the spheroid radius over time; optionally saves a CSV of the
metrics and an .npz snapshot of the final configuration.

    python examples/growth_contact_inhibition.py --days 5 --device cpu
    python examples/growth_contact_inhibition.py --csv runs/growth.csv --save runs/growth.npz
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import warp as wp

from warpbiocell.cells.initialization import spherical_cluster
from warpbiocell.cells.lifecycle import LifecycleParams
from warpbiocell.cells.mechanics import ContactParams, contact_substep, make_neighbor_grid
from warpbiocell.cells.state import CellPopulation
from warpbiocell.device import resolve_device, synchronize
from warpbiocell.metrics.population import population_summary
from warpbiocell.simulation.simulator import TimeStepping, cell_step


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cells", type=int, default=500, help="initial cells")
    parser.add_argument("--capacity", type=int, default=50_000)
    parser.add_argument("--radius", type=float, default=8.0, help="cell radius [um]")
    parser.add_argument("--days", type=float, default=5.0)
    parser.add_argument("--dt-cells", type=float, default=0.25, help="[h]")
    parser.add_argument("--dt-mechanics", type=float, default=0.025, help="[h]")
    parser.add_argument("--doubling-time", type=float, default=24.0, help="free doubling time [h]; illustrative")
    parser.add_argument("--death-rate", type=float, default=0.0, help="[1/h]; placeholder, oxygen-independent")
    parser.add_argument("--inhibition-threshold", type=int, default=8, help="neighbours within query radius")
    parser.add_argument("--margin", type=float, default=2.0, help="query margin beyond 2r [um]")
    parser.add_argument("--report-every", type=float, default=12.0, help="[h]")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--save", type=Path, default=None)
    args = parser.parse_args()

    wp.config.quiet = True
    device = resolve_device(args.device)

    pos = spherical_cluster(args.cells, args.radius, spacing_factor=1.0, jitter=0.1, seed=args.seed)
    pop = CellPopulation.from_numpy(
        pos, np.full(args.cells, args.radius), capacity=args.capacity, device=device, seed=args.seed
    )
    contact = ContactParams(stiffness=10.0, damping=1.0, max_radius=args.radius, margin=args.margin)
    grid = make_neighbor_grid(contact, device)
    lifecycle = LifecycleParams(
        division_rate=np.log(2.0) / args.doubling_time,
        death_rate=args.death_rate,
        inhibition_threshold=args.inhibition_threshold,
    )
    stepping = TimeStepping(dt_cells=args.dt_cells, dt_mechanics=args.dt_mechanics)
    contact.check_substep(stepping.dt_mechanics)

    steps_per_report = max(1, int(round(args.report_every / stepping.dt_cells)))
    n_steps = int(round(args.days * 24.0 / stepping.dt_cells))

    print(
        f"device={device.alias}  cells={args.cells}  lambda={lifecycle.division_rate:.4f} 1/h  "
        f"threshold={lifecycle.inhibition_threshold}  dt_cells={stepping.dt_cells} h  "
        f"mechanics substeps={stepping.mechanics_substeps}"
    )
    header = ["time_h", "cells", "living_cells", "dead_cells", "proliferative_cells", "quiescent_cells", "spheroid_radius_um", "wall_s"]
    print(f"{'t [h]':>7} {'cells':>7} {'prolif':>7} {'quiesc':>7} {'dead':>6} {'R99 [um]':>9} {'wall [s]':>9}")

    contact_substep(pop, grid, contact, dt=0.0)  # neighbour counts of the initial configuration
    rows = []
    wall = 0.0
    t = 0.0

    def record():
        s = population_summary(pop)
        rows.append([t, s["cells"], s["living_cells"], s["dead_cells"], s["proliferative_cells"], s["quiescent_cells"], s["spheroid_radius"], wall])
        print(
            f"{t:>7.1f} {s['cells']:>7d} {s['proliferative_cells']:>7d} {s['quiescent_cells']:>7d} "
            f"{s['dead_cells']:>6d} {s['spheroid_radius']:>9.1f} {wall:>9.1f}"
        )

    record()
    for step in range(1, n_steps + 1):
        t0 = time.perf_counter()
        cell_step(pop, grid, lifecycle, contact, stepping)
        synchronize(device)
        wall += time.perf_counter() - t0
        t += stepping.dt_cells
        if step % steps_per_report == 0 or step == n_steps:
            record()

    print(f"{n_steps} cell steps in {wall:.1f} s on {device.alias}")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
        print(f"saved {args.csv}")
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.save,
            positions=pop.positions_numpy(),
            radii=pop.radii_numpy(),
            states=pop.states_numpy(),
            ages=pop.ages_numpy(),
            neighbors=pop.neighbor_counts_numpy(),
        )
        print(f"saved {args.save}")


if __name__ == "__main__":
    main()
