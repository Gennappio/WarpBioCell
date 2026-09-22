"""Run directory layout (AGENTS.md / docs/roadmap.md "Experiment runner"):

    runs/<timestamp>_<name>/
        config.yaml        verbatim copy of the input configuration (provenance comments included)
        metadata.json      resolved config, seed, versions, git commit, device, timing, status, summary
        metrics.csv        one row per metrics interval
        profiles.csv       radial profiles, one row per shell per profile interval
        checkpoint/        state_<time>h.npz
        figures/           PNGs (when matplotlib is available and output.figures is true)

CSV files are flushed after every row so an interrupted run still leaves usable data.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import platform
import shutil
import subprocess
from pathlib import Path

import warp as wp
import yaml

import warpbiocell

METRIC_COLUMNS = [
    "time_h",
    "cells",
    "living_cells",
    "dead_cells",
    "proliferative_cells",
    "quiescent_cells",
    "hypoxic_cells",
    "mean_radius_um",
    "spheroid_radius_um",
    "necrotic_radius_um",
    "hypoxic_radius_um",
    "non_proliferative_radius_um",
    "viable_rim_um",
    "proliferating_rim_um",
    "oxygen_cells_mean_mmHg",
    "oxygen_cells_min_mmHg",
    "oxygen_grid_mean_mmHg",
    "oxygen_grid_min_mmHg",
    "glucose_cells_mean_mM",
    "glucose_cells_min_mM",
    "lactate_cells_mean_mM",
    "lactate_cells_max_mM",
    "network_proliferation_cells",
    "network_apoptosis_cells",
    "network_growth_arrest_cells",
    "network_necrosis_cells",
    "field_sweeps",
    "cells_outside_grid",
    "cells_outside_tissue",
    "wall_s",
]

PROFILE_COLUMNS = [
    "time_h",
    "shell",
    "r_outer_um",
    "cells",
    "oxygen_mean_mmHg",
    "glucose_mean_mM",
    "lactate_mean_mM",
    "fraction_proliferative",
    "fraction_quiescent",
    "fraction_hypoxic",
    "fraction_dead",
]


def git_commit() -> str | None:
    try:
        root = Path(warpbiocell.__file__).resolve().parents[2]
        out = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def make_run_dir(base: str | Path, name: str) -> Path:
    stamp = dt.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    path = Path(base) / f"{stamp}_{name}"
    suffix = 1
    while path.exists():
        path = Path(base) / f"{stamp}_{name}_{suffix}"
        suffix += 1
    path.mkdir(parents=True)
    return path


class RunOutput:
    def __init__(self, directory: Path, config_path: Path | None, config_dict: dict, device: wp.context.Device, seed: int):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "checkpoint").mkdir(exist_ok=True)
        if config_path is not None:
            shutil.copyfile(config_path, self.directory / "config.yaml")
        else:  # no source file (programmatic or sweep run): dump the resolved values instead
            (self.directory / "config.yaml").write_text(yaml.safe_dump(config_dict, sort_keys=False))
        self.metadata = {
            "name": config_dict.get("name"),
            "status": "running",
            "started": dt.datetime.now(dt.timezone.utc).isoformat(),
            "finished": None,
            "wall_s": None,
            "seed": seed,
            "warpbiocell_version": warpbiocell.__version__,
            "git_commit": git_commit(),
            "warp_version": wp.__version__,
            "device": {"alias": device.alias, "name": device.name, "is_cuda": device.is_cuda},
            "platform": platform.platform(),
            "python": platform.python_version(),
            "config": config_dict,
            "warnings": [],
            "summary": None,
        }
        self._metrics_file = (self.directory / "metrics.csv").open("w", newline="")
        self._metrics = csv.DictWriter(self._metrics_file, fieldnames=METRIC_COLUMNS)
        self._metrics.writeheader()
        self._profiles_file = (self.directory / "profiles.csv").open("w", newline="")
        self._profiles = csv.DictWriter(self._profiles_file, fieldnames=PROFILE_COLUMNS)
        self._profiles.writeheader()
        self.write_metadata()

    def write_metrics(self, row: dict) -> None:
        self._metrics.writerow({key: row.get(key, "") for key in METRIC_COLUMNS})
        self._metrics_file.flush()

    def write_profile(self, time_h: float, profile: dict) -> None:
        for shell in range(len(profile["r_outer"])):
            self._profiles.writerow(
                {
                    "time_h": time_h,
                    "shell": shell,
                    "r_outer_um": float(profile["r_outer"][shell]),
                    "cells": int(profile["cells"][shell]),
                    "oxygen_mean_mmHg": float(profile["oxygen_mean"][shell]),
                    "glucose_mean_mM": float(profile["glucose_mean"][shell]) if "glucose_mean" in profile else "",
                    "lactate_mean_mM": float(profile["lactate_mean"][shell]) if "lactate_mean" in profile else "",
                    "fraction_proliferative": float(profile["fraction_proliferative"][shell]),
                    "fraction_quiescent": float(profile["fraction_quiescent"][shell]),
                    "fraction_hypoxic": float(profile["fraction_hypoxic"][shell]),
                    "fraction_dead": float(profile["fraction_dead"][shell]),
                }
            )
        self._profiles_file.flush()

    def checkpoint_path(self, time_h: float) -> Path:
        return self.directory / "checkpoint" / f"state_{time_h:07.2f}h.npz"

    def write_metadata(self) -> None:
        (self.directory / "metadata.json").write_text(json.dumps(self.metadata, indent=2, default=str))

    def finish(self, status: str, wall_s: float, summary: dict | None) -> None:
        self.metadata.update(
            {"status": status, "finished": dt.datetime.now(dt.timezone.utc).isoformat(), "wall_s": wall_s, "summary": summary}
        )
        self.write_metadata()
        self._metrics_file.close()
        self._profiles_file.close()
