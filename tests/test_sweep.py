import csv
import json
import math
from pathlib import Path

import pytest
import yaml

from warpbiocell import sweep as sweep_cli
from warpbiocell.metrics.timeseries import onset, onset_summary
from warpbiocell.simulation.config import ConfigError
from warpbiocell.simulation.sweep import Sweep, SweepSpec, load_sweep

REPO_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "tumor_spheroid.yaml"
REPO_SWEEP = Path(__file__).resolve().parents[1] / "configs" / "sweeps" / "oxygen_boundary.yaml"

TINY_BASE = {
    "simulation": {"duration_h": 2.0, "dt_cells_h": 0.5, "dt_mechanics_h": 0.05, "device": "cpu"},
    "cells": {"initial_count": 200, "max_cells": 2000},
    "lifecycle": {"division_rate_per_h": 0.3},
    "oxygen": {"grid": {"box_um": 400.0}},
    "output": {"metrics_every_h": 1.0, "figures": False},
}


def test_repository_sweep_expands_to_the_expected_runs():
    spec = load_sweep(REPO_SWEEP)
    runs = spec.runs()
    assert spec.name == "oxygen_boundary"
    assert len(runs) == 5 * 3
    assert runs[0] == ("boundary_mmHg=20_seed=0", {"oxygen.boundary_mmHg": 20, "simulation.seed": 0})
    assert runs[-1][0] == "boundary_mmHg=150_seed=2"


def test_sweep_spec_validation(tmp_path):
    with pytest.raises(ConfigError):
        SweepSpec(name="x", grid={"a.b": []})
    with pytest.raises(ConfigError):
        SweepSpec(name="x", seeds=())
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: x\nvariants: []\n")
    with pytest.raises(ConfigError, match="unknown keys"):
        load_sweep(bad)


def test_onset_helpers():
    rows = [
        {"time_h": 0.0, "hypoxic_cells": 0, "dead_cells": 0, "spheroid_radius_um": 100.0},
        {"time_h": 1.0, "hypoxic_cells": 3, "dead_cells": 0, "spheroid_radius_um": 110.0},
        {"time_h": 2.0, "hypoxic_cells": 9, "dead_cells": 0, "spheroid_radius_um": 120.0},
    ]
    assert onset(rows, "hypoxic_cells") == (1.0, 110.0)
    summary = onset_summary(rows)
    assert summary["hypoxia_onset_radius_um"] == 110.0
    assert math.isnan(summary["necrosis_onset_time_h"])


def test_tiny_sweep_runs_and_summarises(tmp_path):
    spec = SweepSpec(name="tiny", seeds=(0, 1), overrides={"simulation.duration_h": 1.0}, grid={"oxygen.boundary_mmHg": [10, 60]})
    sweep = Sweep(spec, TINY_BASE)

    rows = sweep.run(tmp_path / "sweep", device="cpu")

    assert [r["label"] for r in rows] == ["boundary_mmHg=10_seed=0", "boundary_mmHg=10_seed=1", "boundary_mmHg=60_seed=0", "boundary_mmHg=60_seed=1"]
    assert all(r["status"] == "completed" for r in rows)
    with (tmp_path / "sweep" / "summary.csv").open() as f:
        summary = list(csv.DictReader(f))
    assert len(summary) == 4
    assert {r["oxygen.boundary_mmHg"] for r in summary} == {"10", "60"}
    assert float(summary[0]["final_time_h"]) == 1.0
    # Low boundary oxygen makes the initial cluster hypoxic within the first metrics interval
    # (onset resolution = metrics_every_h = 1 h); high does not.
    assert float(summary[0]["hypoxia_onset_time_h"]) == 1.0
    assert summary[2]["hypoxia_onset_time_h"] == "nan"
    run_dir = Path(summary[0]["run_dir"])
    assert (run_dir / "metrics.csv").exists()
    resolved = yaml.safe_load((run_dir / "config.yaml").read_text())
    assert resolved["oxygen"]["boundary_mmHg"] == 10.0 and resolved["simulation"]["seed"] == 0 and resolved["name"] == "boundary_mmHg=10_seed=0"
    meta = json.loads((tmp_path / "sweep" / "metadata.json").read_text())
    assert meta["finished"] is not None and len(meta["runs"]) == 4


def test_sweep_cli_dry_run_and_errors(tmp_path, capsys):
    assert sweep_cli.main(["--config", str(REPO_CONFIG), "--sweep", str(REPO_SWEEP), "--dry-run"]) == 0
    assert "15 runs" in capsys.readouterr().out
    assert sweep_cli.main(["--config", str(REPO_CONFIG), "--sweep", str(tmp_path / "missing.yaml")]) == 2
    assert not any(tmp_path.iterdir())
