"""Milestone 4 demonstrator: a spheroid grows while consuming oxygen from a diffusive field.

Cells divide unless crowded or oxygen-starved, become HYPOXIC below the hypoxia threshold,
die faster below the death threshold, and dead cells stay in place. The oxygen field is
solved to steady state every cell step. Reports population by state, oxygen at the cells
and in the grid, and prints the final radial profile.

    python examples/spheroid_oxygen.py --days 4 --device cpu
    python examples/spheroid_oxygen.py --boundary-o2 150 --box 1200   # culture-like conditions
    python examples/spheroid_oxygen.py --csv runs/spheroid.csv --save runs/spheroid.npz
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
from warpbiocell.fields.diffusion import SolverSettings
from warpbiocell.fields.oxygen import OxygenField, OxygenParams
from warpbiocell.fields.scalar_field import GridGeometry
from warpbiocell.metrics.population import population_summary, radial_profile
from warpbiocell.simulation.simulator import TimeStepping, cell_step

COLUMNS = [
    "time_h", "cells", "living_cells", "dead_cells", "proliferative_cells", "quiescent_cells", "hypoxic_cells",
    "spheroid_radius_um", "oxygen_cells_mean", "oxygen_cells_min", "oxygen_grid_min", "field_sweeps", "wall_s",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cells", type=int, default=1000, help="initial cells")
    parser.add_argument("--capacity", type=int, default=100_000)
    parser.add_argument("--radius", type=float, default=8.0, help="cell radius [um]")
    parser.add_argument("--days", type=float, default=4.0)
    parser.add_argument("--dt-cells", type=float, default=0.25, help="[h]")
    parser.add_argument("--dt-mechanics", type=float, default=0.025, help="[h]")
    parser.add_argument("--doubling-time", type=float, default=24.0, help="free doubling time [h]; illustrative")
    parser.add_argument("--inhibition-threshold", type=int, default=8)
    parser.add_argument("--hypoxia-threshold", type=float, default=8.0, help="[mmHg]")
    parser.add_argument("--death-threshold", type=float, default=2.0, help="[mmHg]")
    parser.add_argument("--anoxic-death-rate", type=float, default=0.5, help="[1/h]")
    parser.add_argument("--boundary-o2", type=float, default=38.0, help="oxygen at the box faces [mmHg]; 38 ~ 5%% O2 tissue, 150 ~ air")
    parser.add_argument("--box", type=float, default=800.0, help="cubic domain edge [um]")
    parser.add_argument("--dx", type=float, default=20.0, help="grid spacing [um]")
    parser.add_argument("--omega", type=float, default=1.8)
    parser.add_argument("--report-every", type=float, default=12.0, help="[h]")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--save", type=Path, default=None)
    args = parser.parse_args()

    wp.config.quiet = True
    device = resolve_device(args.device)

    pos = spherical_cluster(args.cells, args.radius, spacing_factor=1.0, jitter=0.1, seed=args.seed)
    pop = CellPopulation.from_numpy(pos, np.full(args.cells, args.radius), capacity=args.capacity, device=device, seed=args.seed)
    contact = ContactParams(stiffness=10.0, damping=1.0, max_radius=args.radius, margin=2.0)
    grid = make_neighbor_grid(contact, device)
    lifecycle = LifecycleParams(
        division_rate=np.log(2.0) / args.doubling_time,
        anoxic_death_rate=args.anoxic_death_rate,
        inhibition_threshold=args.inhibition_threshold,
        hypoxia_threshold=args.hypoxia_threshold,
        death_threshold=args.death_threshold,
    )
    stepping = TimeStepping(dt_cells=args.dt_cells, dt_mechanics=args.dt_mechanics)
    contact.check_substep(stepping.dt_mechanics)
    geometry = GridGeometry.centered_cube(args.box, args.dx)
    oxygen = OxygenField(geometry, OxygenParams(boundary_value=args.boundary_o2), SolverSettings(omega=args.omega, tolerance=1e-5, check_every=10), device=device)

    steps_per_report = max(1, int(round(args.report_every / stepping.dt_cells)))
    n_steps = int(round(args.days * 24.0 / stepping.dt_cells))

    print(
        f"device={device.alias}  cells={args.cells}  grid={geometry.shape} dx={args.dx} um  O2 boundary={args.boundary_o2} mmHg  "
        f"lambda={lifecycle.division_rate:.4f} 1/h  thresholds hyp/death={args.hypoxia_threshold}/{args.death_threshold} mmHg"
    )
    print(f"{'t [h]':>6} {'cells':>7} {'prolif':>7} {'quiesc':>7} {'hypox':>7} {'dead':>6} {'R99':>6} {'O2 cells':>9} {'O2 min':>7} {'sweeps':>6} {'wall':>7}")

    contact_substep(pop, grid, contact, dt=0.0)
    oxygen.update(pop)
    rows = []
    wall = 0.0
    t = 0.0

    def record():
        s = population_summary(pop, oxygen)
        rows.append([t, s["cells"], s["living_cells"], s["dead_cells"], s["proliferative_cells"], s["quiescent_cells"], s["hypoxic_cells"],
                     s["spheroid_radius"], s["oxygen_cells_mean"], s["oxygen_cells_min"], s["oxygen_grid_min"], s["field_sweeps"], wall])
        print(
            f"{t:>6.1f} {s['cells']:>7d} {s['proliferative_cells']:>7d} {s['quiescent_cells']:>7d} {s['hypoxic_cells']:>7d} "
            f"{s['dead_cells']:>6d} {s['spheroid_radius']:>6.0f} {s['oxygen_cells_mean']:>9.1f} {s['oxygen_cells_min']:>7.1f} "
            f"{s['field_sweeps']:>6d} {wall:>6.1f}s"
        )

    record()
    for step in range(1, n_steps + 1):
        t0 = time.perf_counter()
        report = cell_step(pop, grid, lifecycle, contact, stepping, oxygen=oxygen)
        synchronize(device)
        wall += time.perf_counter() - t0
        t += stepping.dt_cells
        if not report.field_converged:
            print(f"warning: field solve hit max_sweeps at t={t:.2f} h (residual {report.field_residual:.2e})")
        if step % steps_per_report == 0 or step == n_steps:
            record()

    print(f"{n_steps} cell steps in {wall:.1f} s on {device.alias}")

    profile = radial_profile(pop, n_bins=8)
    print("\nradial profile (shell outer radius [um], cells, O2 mean [mmHg], fraction proliferative / hypoxic / dead)")
    for b in range(len(profile["r_outer"])):
        print(
            f"{profile['r_outer'][b]:>7.0f} {profile['cells'][b]:>6d} {profile['oxygen_mean'][b]:>8.1f}   "
            f"{profile['fraction_proliferative'][b]:>5.2f} / {profile['fraction_hypoxic'][b]:>5.2f} / {profile['fraction_dead'][b]:>5.2f}"
        )

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(COLUMNS)
            writer.writerows(rows)
        print(f"saved {args.csv}")
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.save,
            positions=pop.positions_numpy(), radii=pop.radii_numpy(), states=pop.states_numpy(),
            oxygen=pop.oxygen_numpy(), field=oxygen.numpy(), origin=np.array(geometry.origin), dx=geometry.dx,
        )
        print(f"saved {args.save}")


if __name__ == "__main__":
    main()
