"""Summarise a sweep directory: mean ± std over seeds per swept value, table and figure.

    python examples/analyze_sweep.py runs/<timestamp>_oxygen_boundary
    python examples/analyze_sweep.py runs/..._oxygen_boundary --zero-order-oxygen --figure docs/results/figures/oxygen_boundary_sweep.png

``--zero-order-oxygen`` overlays the critical radius from the zero-order diffusion estimate
(uniform consumption rho*q inside a sphere, Dirichlet box at distance L = box/2 from the
centre):  O(0) = O_b - rho q R^3/(3D) (1/R - 1/L) - rho q R^2/(6D) = hypoxia threshold.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml

QUANTITIES = [
    ("hypoxia_onset_radius_um", "hypoxia onset radius [um]"),
    ("necrosis_onset_radius_um", "necrosis onset radius [um]"),
    ("final_living_cells", "living cells at the end"),
    ("final_spheroid_radius_um", "spheroid radius at the end [um]"),
    ("final_viable_rim_um", "viable rim at the end [um]"),
    ("final_necrotic_fraction", "necrotic fraction at the end"),
    ("final_oxygen_drop_mmHg", "O2 boundary - O2 min at cells [mmHg]"),
]


def load_summary(directory: Path):
    with (directory / "summary.csv").open() as f:
        rows = list(csv.DictReader(f))
    meta = json.loads((directory / "metadata.json").read_text())
    keys = list(meta["spec"]["grid"])
    if len(keys) != 1:
        raise SystemExit("this script handles one swept parameter")
    key = keys[0]
    for r in rows:
        for k, v in list(r.items()):
            try:
                r[k] = float(v)
            except (TypeError, ValueError):
                pass
        r["final_necrotic_fraction"] = r["final_dead_cells"] / r["final_cells"] if r["final_cells"] else float("nan")
        r["final_oxygen_drop_mmHg"] = r[key] - r["final_oxygen_cells_min_mmHg"] if key.endswith("mmHg") else float("nan")
    return key, rows, meta


def aggregate(key, rows):
    values = sorted({r[key] for r in rows})
    table = {}
    for v in values:
        group = [r for r in rows if r[key] == v]
        table[v] = {q: (np.nanmean([g[q] for g in group]), np.nanstd([g[q] for g in group]), sum(np.isnan(g[q]) for g in group)) for q, _ in QUANTITIES}
        table[v]["n"] = len(group)
    return values, table


def zero_order_radius(base: dict, boundary: float, rho: float) -> float:
    ox, lc = base["oxygen"], base["lifecycle"]
    D, q = float(ox["diffusion_um2_per_h"]), float(ox["uptake_max_mmHg_um3_per_h"])
    L = 0.5 * float(ox["grid"]["box_um"])
    target = float(lc["hypoxia_threshold_mmHg"])
    radii = np.linspace(1.0, L - 1.0, 4000)
    centre = boundary - rho * q * radii**3 / (3 * D) * (1 / radii - 1 / L) - rho * q * radii**2 / (6 * D)
    below = np.where(centre <= target)[0]
    return float(radii[below[0]]) if below.size else float("nan")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--figure", type=Path, default=None)
    parser.add_argument("--zero-order-oxygen", action="store_true")
    parser.add_argument("--rho", type=float, default=2.3e-4, help="cell density [cells/um^3] for the zero-order estimate")
    args = parser.parse_args()

    key, rows, meta = load_summary(args.directory)
    values, table = aggregate(key, rows)
    statuses = {r["status"] for r in rows}
    print(f"sweep {meta['name']}: {len(rows)} runs, statuses {sorted(statuses)}, swept {key} = {values}")
    header = f"{key:>22} " + " ".join(f"{label[:26]:>28}" for _, label in QUANTITIES)
    print(header)
    for v in values:
        cells = []
        for q, _ in QUANTITIES:
            mean, std, n_nan = table[v][q]
            cells.append(f"{'n/a':>28}" if np.isnan(mean) else f"{mean:>14.1f} ± {std:<8.1f}{('(' + str(n_nan) + ' nan)') if n_nan else '':>4}")
        print(f"{v:>22g} " + " ".join(cells))

    estimate = None
    if args.zero_order_oxygen:
        base = yaml.safe_load((args.directory / "base_config.yaml").read_text())
        estimate = {v: zero_order_radius(base, v, args.rho) for v in values}
        print("zero-order hypoxia radius [um]: " + ", ".join(f"{v:g}: {r:.0f}" for v, r in estimate.items()))

    if args.figure:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        panels = [
            ("hypoxia_onset_radius_um", "hypoxia onset radius [um]"),
            ("final_living_cells", "living cells at the end"),
            ("final_necrotic_fraction", "necrotic fraction at the end"),
            ("final_viable_rim_um", "viable rim at the end [um]"),
        ]
        for ax, (q, label) in zip(axes.ravel(), panels):
            means = [table[v][q][0] for v in values]
            stds = [table[v][q][1] for v in values]
            ax.errorbar(values, means, yerr=stds, fmt="o-", color="#2a9d8f", capsize=3, label="simulation (mean ± sd over seeds)")
            if q == "hypoxia_onset_radius_um" and estimate:
                ax.plot(values, [estimate[v] for v in values], "k--", label="zero-order estimate")
                ax.legend(frameon=False, fontsize=8)
            ax.set_xlabel(key)
            ax.set_ylabel(label)
        fig.suptitle(f"sweep {meta['name']}")
        fig.tight_layout()
        args.figure.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.figure, dpi=130)
        print(f"figure written to {args.figure}")


if __name__ == "__main__":
    main()
