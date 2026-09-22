"""Command-line sweep runner.

    python -m warpbiocell.sweep --config configs/tumor_spheroid.yaml --sweep configs/sweeps/oxygen_boundary.yaml
    python -m warpbiocell.sweep --config ... --sweep ... --dry-run          # list the runs only
    python -m warpbiocell.sweep --config ... --sweep ... --device cuda:0 --figures

Writes runs/<timestamp>_<sweep name>/ with sweep.yaml, base_config.yaml, summary.csv,
metadata.json and one run directory per combination and seed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from warpbiocell.io.run_output import make_run_dir
from warpbiocell.simulation.config import ConfigError, apply_overrides
from warpbiocell.simulation.sweep import Sweep, load_sweep


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True, help="base experiment configuration")
    parser.add_argument("--sweep", type=Path, required=True, help="sweep specification")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE", help="override applied to the base config before the sweep")
    parser.add_argument("--out", type=Path, default=Path("runs"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--figures", action="store_true", help="also render figures for every run")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    try:
        with args.config.open() as f:
            base = yaml.safe_load(f) or {}
        apply_overrides(base, args.overrides)
        spec = load_sweep(args.sweep)
        sweep = Sweep(spec, base, base_path=args.config, sweep_path=args.sweep)
    except (ConfigError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    log = (lambda _: None) if args.quiet else print
    if args.dry_run:
        for label, values, config in sweep.configs:
            print(f"{label}: {values}  ({config.n_steps} steps)")
        print(f"{len(sweep.configs)} runs")
        return 0

    out_dir = make_run_dir(args.out, spec.name)
    log(f"sweep directory: {out_dir}  ({len(sweep.configs)} runs)")
    rows = sweep.run(out_dir, device=args.device, figures=args.figures, log=log)
    failed = [r["label"] for r in rows if r["status"] != "completed"]
    log(f"done: {len(rows) - len(failed)} completed, {len(failed)} not completed -> {out_dir}/summary.csv")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
