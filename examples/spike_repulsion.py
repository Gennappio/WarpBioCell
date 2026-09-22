"""Technical spike: 10k spherical cells + HashGrid neighbour search + overdamped overlap repulsion.

A cluster initialised with 10% compression relaxes under contact forces alone. Reports how
the mean contact count and the cluster radius evolve. No oxygen, no proliferation.

    python examples/spike_repulsion.py --cells 10000 --hours 2 --device cpu
    python examples/spike_repulsion.py --save runs/spike.npz     # positions for later plotting
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import warp as wp

from warpbiocell.cells.initialization import spherical_cluster
from warpbiocell.cells.mechanics import ContactParams, contact_substep, make_neighbor_grid, relax_contacts
from warpbiocell.cells.state import CellPopulation
from warpbiocell.device import resolve_device, synchronize


def cluster_radius(pop: CellPopulation) -> float:
    x = pop.positions_numpy()
    return float(np.linalg.norm(x - x.mean(axis=0), axis=1).max())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cells", type=int, default=10_000)
    parser.add_argument("--radius", type=float, default=8.0, help="cell radius [um]")
    parser.add_argument("--compression", type=float, default=0.9, help="initial lattice spacing / cell diameter")
    parser.add_argument("--hours", type=float, default=2.0, help="simulated time [h]")
    parser.add_argument("--dt", type=float, default=0.005, help="mechanics substep [h]")
    parser.add_argument("--stiffness", type=float, default=10.0, help="k [force/um], illustrative")
    parser.add_argument("--damping", type=float, default=1.0, help="gamma [force*h/um], illustrative")
    parser.add_argument("--report-every", type=float, default=0.25, help="report interval [h]")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save", type=Path, default=None, help="write final positions/radii to this .npz")
    args = parser.parse_args()

    wp.config.quiet = True
    device = resolve_device(args.device)

    pos = spherical_cluster(args.cells, args.radius, spacing_factor=args.compression, jitter=0.1, seed=args.seed)
    pop = CellPopulation.from_numpy(pos, np.full(args.cells, args.radius), capacity=args.cells, device=device)
    params = ContactParams(stiffness=args.stiffness, damping=args.damping, max_radius=args.radius, margin=0.5)
    params.check_substep(args.dt)
    grid = make_neighbor_grid(params, device)

    substeps_per_report = max(1, int(round(args.report_every / args.dt)))
    n_reports = int(round(args.hours / (substeps_per_report * args.dt)))

    print(f"device={device.alias}  cells={args.cells}  radius={args.radius} um  rate={params.rate:.3g} 1/h  dt={args.dt} h")
    print(f"{'t [h]':>7} {'mean contacts':>14} {'cells in contact':>17} {'cluster radius [um]':>20} {'wall [s]':>9}")

    # dt = 0 evaluates contacts on the initial configuration without moving anything (and
    # triggers kernel compilation, which is kept out of the wall-clock figures).
    contact_substep(pop, grid, params, dt=0.0)
    t_sim = 0.0
    wall = 0.0
    for k in range(n_reports + 1):
        if k > 0:
            t0 = time.perf_counter()
            relax_contacts(pop, grid, params, dt=args.dt, substeps=substeps_per_report)
            synchronize(device)
            wall += time.perf_counter() - t0
            t_sim += substeps_per_report * args.dt
        contacts = pop.neighbor_counts_numpy()
        print(
            f"{t_sim:>7.2f} {contacts.mean():>14.2f} {np.mean(contacts > 0):>17.1%} "
            f"{cluster_radius(pop):>20.1f} {wall:>9.2f}"
        )

    total_substeps = n_reports * substeps_per_report
    print(f"{total_substeps} substeps in {wall:.2f} s  ->  {1e3 * wall / total_substeps:.2f} ms/substep on {device.alias}")

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.save, positions=pop.positions_numpy(), radii=pop.radii_numpy(), contacts=pop.neighbor_counts_numpy())
        print(f"saved {args.save}")


if __name__ == "__main__":
    main()
