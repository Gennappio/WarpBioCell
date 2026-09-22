"""Command-line experiment runner.

    python -m warpbiocell.run --config configs/tumor_spheroid.yaml
    python -m warpbiocell.run --config configs/tumor_spheroid.yaml --set oxygen.boundary_mmHg=150 --set simulation.duration_h=48
    python -m warpbiocell.run --config configs/tumor_spheroid.yaml --device cpu --out runs/ --no-figures

Writes runs/<timestamp>_<name>/ with config.yaml, metadata.json, metrics.csv, profiles.csv,
checkpoint/ and figures/ (see io/run_output.py).
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

from warpbiocell.io.run_output import make_run_dir
from warpbiocell.simulation.config import ConfigError, load_config
from warpbiocell.simulation.experiment import Experiment


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE", help="override a config value, e.g. oxygen.boundary_mmHg=150")
    parser.add_argument("--out", type=Path, default=Path("runs"), help="base directory for run outputs")
    parser.add_argument("--device", default=None, help="cpu, cuda:0 or auto (default: from the config)")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config, args.overrides)
        if args.no_figures:
            config = dataclasses.replace(config, output=dataclasses.replace(config.output, figures=False))
        experiment = Experiment(config, device=args.device, config_path=args.config)
    except (ConfigError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    run_dir = make_run_dir(args.out, config.name)
    log = (lambda _: None) if args.quiet else print
    log(f"run directory: {run_dir}")
    log(
        f"device={experiment.device.alias}  cells={config.cells.initial_count}->max {config.cells.max_cells}  "
        f"duration={config.simulation.duration_h} h  dt_cells={config.simulation.dt_cells_h} h  "
        f"oxygen={'on' if config.oxygen.enabled else 'off'}  seed={config.simulation.seed}"
    )
    result = experiment.run(run_dir, log=log)
    log(f"status={result.status}  steps={result.summary['steps']}  wall={result.wall_s:.1f} s  -> {run_dir}")
    return 0 if result.status == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
