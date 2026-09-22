"""End-to-end experiment runs on tiny configurations: outputs, reproducibility, failure modes."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from warpbiocell import run as run_module
from warpbiocell.io.checkpoints import load_checkpoint
from warpbiocell.io.run_output import METRIC_COLUMNS, PROFILE_COLUMNS
from warpbiocell.simulation.config import config_from_dict
from warpbiocell.simulation.experiment import Experiment

REPO_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "tumor_spheroid.yaml"


def _tiny(**overrides):
    data = {
        "name": "tiny",
        "simulation": {"duration_h": 3.0, "dt_cells_h": 0.5, "dt_mechanics_h": 0.05, "seed": 3, "device": "cpu"},
        "cells": {"initial_count": 300, "max_cells": 3000},
        "lifecycle": {"division_rate_per_h": 0.3},
        "oxygen": {"grid": {"box_um": 400.0, "dx_um": 20.0}},
        "output": {"metrics_every_h": 1.0, "profile_every_h": 1.5, "checkpoint_every_h": 3.0, "figures": True},
    }
    for key, value in overrides.items():
        section, name = key.split(".")
        data.setdefault(section, {})[name] = value
    return config_from_dict(data)


def _read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def test_run_writes_the_full_directory_layout(tmp_path):
    config = _tiny()
    result = Experiment(config, config_path=REPO_CONFIG).run(tmp_path / "run")

    run_dir = tmp_path / "run"
    assert result.status == "completed"
    assert (run_dir / "config.yaml").read_text() == REPO_CONFIG.read_text()

    metrics = _read_csv(run_dir / "metrics.csv")
    assert list(metrics[0].keys()) == METRIC_COLUMNS
    assert len(metrics) == 4  # t = 0, 1, 2, 3 h
    assert [float(r["time_h"]) for r in metrics] == [0.0, 1.0, 2.0, 3.0]
    assert int(metrics[-1]["cells"]) > 300
    assert float(metrics[-1]["oxygen_cells_min_mmHg"]) > 0.0
    assert all(int(r["cells_outside_grid"]) == 0 for r in metrics)

    profiles = _read_csv(run_dir / "profiles.csv")
    assert list(profiles[0].keys()) == PROFILE_COLUMNS
    assert {float(r["time_h"]) for r in profiles} == {0.0, 1.5, 3.0}

    checkpoints = sorted(p.name for p in (run_dir / "checkpoint").iterdir())
    assert checkpoints == ["state_0000.00h.npz", "state_0003.00h.npz"]
    state = load_checkpoint(run_dir / "checkpoint" / "state_0003.00h.npz")
    assert int(state["count"]) == int(metrics[-1]["cells"])
    assert state["position"].shape == (int(state["count"]), 3)
    assert state["field"].shape == (21, 21, 21)
    assert "rng_state" in state

    meta = json.loads((run_dir / "metadata.json").read_text())
    assert meta["status"] == "completed"
    assert meta["config"]["cells"]["initial_count"] == 300
    assert meta["device"]["alias"] == "cpu"
    assert meta["summary"]["steps"] == 6
    assert "git_commit" in meta and "warp_version" in meta

    figures = sorted(p.name for p in (run_dir / "figures").iterdir())
    assert figures == ["cells_3d.png", "oxygen_slice.png", "population.png", "radial_profile.png"]


def test_in_memory_runs_are_reproducible_and_seed_dependent():
    a = Experiment(_tiny()).run()
    b = Experiment(_tiny()).run()
    c = Experiment(_tiny(**{"simulation.seed": 4})).run()

    for key in ("cells", "dead_cells", "spheroid_radius_um", "oxygen_cells_min_mmHg"):
        np.testing.assert_array_equal(a.metric(key), b.metric(key))
    assert a.summary["cells"] != c.summary["cells"] or a.summary["spheroid_radius_um"] != c.summary["spheroid_radius_um"]


def test_capacity_overflow_ends_the_run_gracefully(tmp_path):
    config = _tiny(**{"cells.max_cells": 320})
    assert any("max_cells" in w for w in config.validate())

    result = Experiment(config).run(tmp_path / "run")

    assert result.status == "capacity_exceeded"
    meta = json.loads((tmp_path / "run" / "metadata.json").read_text())
    assert meta["status"] == "capacity_exceeded"
    metrics = _read_csv(tmp_path / "run" / "metrics.csv")
    assert len(metrics) >= 2 and int(metrics[-1]["cells"]) <= 320


def test_oxygen_can_be_disabled(tmp_path):
    result = Experiment(_tiny(**{"oxygen.enabled": False})).run(tmp_path / "run")

    assert result.status == "completed"
    assert result.summary["hypoxic_cells"] == 0
    assert np.isnan(result.summary["oxygen_cells_min_mmHg"])
    assert not (tmp_path / "run" / "figures" / "oxygen_slice.png").exists()
    assert (tmp_path / "run" / "figures" / "population.png").exists()


def test_command_line_runner(tmp_path):
    code = run_module.main(
        [
            "--config", str(REPO_CONFIG), "--device", "cpu", "--out", str(tmp_path), "--quiet", "--no-figures",
            "--set", "simulation.duration_h=1", "--set", "cells.initial_count=200", "--set", "output.metrics_every_h=0.5",
        ]
    )
    assert code == 0
    runs = list(tmp_path.iterdir())
    assert len(runs) == 1 and runs[0].name.endswith("_tumor_spheroid")
    assert not (runs[0] / "figures").exists()
    assert len(_read_csv(runs[0] / "metrics.csv")) == 3


def test_command_line_runner_reports_config_errors(tmp_path, capsys):
    code = run_module.main(["--config", str(REPO_CONFIG), "--out", str(tmp_path), "--set", "oxygen.boundary_mmhg=1"])
    assert code == 2
    assert "unknown keys" in capsys.readouterr().err
    assert not any(tmp_path.iterdir())


@pytest.mark.parametrize("bad", [["--config", "does_not_exist.yaml"]])
def test_missing_config_file(bad, tmp_path):
    assert run_module.main(bad + ["--out", str(tmp_path)]) == 2
