"""Parameter sweeps: a base configuration, a grid of overrides and a list of seeds.

Sweep file (YAML):

    name: oxygen_boundary
    seeds: [0, 1, 2]
    overrides:                       # applied to every run
      simulation.duration_h: 288
    grid:                            # cartesian product, one run per combination and seed
      oxygen.boundary_mmHg: [20, 38, 60, 100, 150]

Every run is an ordinary run directory under ``runs/<timestamp>_<name>/<label>/`` and
``summary.csv`` collects, per run, the swept values, the seed, the status, onset times and
radii of hypoxia and necrosis, and the final metrics row. Runs are sequential; no scheduler.
"""

from __future__ import annotations

import copy
import csv
import dataclasses
import datetime as dt
import itertools
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import yaml

from warpbiocell.io.run_output import METRIC_COLUMNS
from warpbiocell.metrics.timeseries import onset_summary
from warpbiocell.simulation.config import ConfigError, apply_overrides, config_from_dict
from warpbiocell.simulation.experiment import Experiment

ONSET_COLUMNS = ["hypoxia_onset_time_h", "hypoxia_onset_radius_um", "necrosis_onset_time_h", "necrosis_onset_radius_um"]


@dataclass(frozen=True)
class SweepSpec:
    name: str
    seeds: tuple[int, ...] = (0,)
    overrides: dict = field(default_factory=dict)  # dotted key -> value
    grid: dict = field(default_factory=dict)  # dotted key -> list of values

    def __post_init__(self):
        if not self.name:
            raise ConfigError("sweep needs a name")
        if not self.seeds:
            raise ConfigError("sweep needs at least one seed")
        for key, values in self.grid.items():
            if not isinstance(values, (list, tuple)) or not values:
                raise ConfigError(f"grid.{key} must be a non-empty list")

    @property
    def grid_keys(self) -> list[str]:
        return list(self.grid)

    def runs(self) -> list[tuple[str, dict]]:
        """``(label, {dotted key: value})`` for every grid combination and seed, in order."""
        out = []
        combos = itertools.product(*(self.grid[k] for k in self.grid_keys)) if self.grid else [()]
        for combo in combos:
            values = dict(zip(self.grid_keys, combo))
            for seed in self.seeds:
                parts = [f"{k.split('.')[-1]}={_fmt(v)}" for k, v in values.items()] + [f"seed={seed}"]
                out.append(("_".join(parts), {**values, "simulation.seed": seed}))
        return out


def _fmt(value) -> str:
    return f"{value:g}" if isinstance(value, (int, float)) and not isinstance(value, bool) else str(value)


def load_sweep(path: str | Path) -> SweepSpec:
    with Path(path).open() as f:
        data = yaml.safe_load(f) or {}
    unknown = set(data) - {"name", "seeds", "overrides", "grid"}
    if unknown:
        raise ConfigError(f"sweep: unknown keys {sorted(unknown)}")
    return SweepSpec(
        name=str(data.get("name", Path(path).stem)),
        seeds=tuple(int(s) for s in data.get("seeds", [0])),
        overrides=dict(data.get("overrides") or {}),
        grid=dict(data.get("grid") or {}),
    )


def build_run_config(base: dict, spec: SweepSpec, label: str, values: dict):
    data = copy.deepcopy(base)
    apply_overrides(data, [f"{k}={_yaml(v)}" for k, v in spec.overrides.items()])
    apply_overrides(data, [f"{k}={_yaml(v)}" for k, v in values.items()])
    data["name"] = label
    return config_from_dict(data)


def _yaml(value) -> str:
    return yaml.safe_dump(value, default_flow_style=True).strip().removesuffix("\n...")


class Sweep:
    def __init__(self, spec: SweepSpec, base: dict, base_path: Path | None = None, sweep_path: Path | None = None):
        self.spec = spec
        self.base = base
        self.base_path = base_path
        self.sweep_path = sweep_path
        # Validate every configuration before running anything.
        self.configs = [(label, values, build_run_config(base, spec, label, values)) for label, values in spec.runs()]
        for _, _, config in self.configs:
            config.validate()

    def run(self, out_dir: str | Path, device: str | None = None, figures: bool = False, log: Callable[[str], None] | None = None) -> list[dict]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if self.sweep_path is not None:
            shutil.copyfile(self.sweep_path, out_dir / "sweep.yaml")
        if self.base_path is not None:
            shutil.copyfile(self.base_path, out_dir / "base_config.yaml")
        metadata = {
            "name": self.spec.name,
            "started": dt.datetime.now(dt.timezone.utc).isoformat(),
            "finished": None,
            "spec": dataclasses.asdict(self.spec),
            "runs": [],
        }
        columns = ["label", "run_dir", "seed", *self.spec.grid_keys, "status", "wall_s", *ONSET_COLUMNS, *[f"final_{c}" for c in METRIC_COLUMNS]]
        rows = []
        with (out_dir / "summary.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for index, (label, values, config) in enumerate(self.configs, start=1):
                if not figures:
                    config = dataclasses.replace(config, output=dataclasses.replace(config.output, figures=False))
                if log:
                    log(f"[{index}/{len(self.configs)}] {label}")
                run_dir = out_dir / label
                experiment = Experiment(config, device=device)
                result = experiment.run(run_dir, log=None)
                row = {
                    "label": label,
                    "run_dir": str(run_dir),
                    "seed": config.simulation.seed,
                    **{k: values[k] for k in self.spec.grid_keys},
                    "status": result.status,
                    "wall_s": result.wall_s,
                    **onset_summary(result.metrics),
                    **{f"final_{k}": v for k, v in result.metrics[-1].items()},
                }
                rows.append(row)
                writer.writerow(row)
                f.flush()
                metadata["runs"].append({"label": label, "status": result.status, "wall_s": result.wall_s})
                (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
                if log:
                    log(
                        f"    {result.status}  cells={result.metrics[-1]['cells']}  R={result.metrics[-1]['spheroid_radius_um']:.0f} um  "
                        f"hypoxia onset R={row['hypoxia_onset_radius_um']:.0f} um  wall={result.wall_s:.0f} s"
                    )
        metadata["finished"] = dt.datetime.now(dt.timezone.utc).isoformat()
        (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
        return rows
