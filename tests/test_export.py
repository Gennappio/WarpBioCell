"""OpenUSD export of run checkpoints (optional usd-core) and the no-USD-in-simulation rule."""

import json
from pathlib import Path

import numpy as np
import pytest

from warpbiocell import export_usd as export_cli
from warpbiocell.io.checkpoints import load_checkpoint
from warpbiocell.simulation.config import config_from_dict
from warpbiocell.simulation.experiment import Experiment

SRC = Path(__file__).resolve().parents[1] / "src" / "warpbiocell"


def _tiny_run(tmp_path, geometry=False):
    data = {
        "name": "usd",
        "simulation": {"duration_h": 1.0, "dt_cells_h": 0.5, "dt_mechanics_h": 0.05, "device": "cpu"},
        "cells": {"initial_count": 150, "max_cells": 2000},
        "lifecycle": {"division_rate_per_h": 0.5},
        "oxygen": {"grid": {"box_um": 300.0}},
        "output": {"metrics_every_h": 0.5, "checkpoint_every_h": 0.5, "figures": False},
    }
    if geometry:
        data["geometry"] = {"enabled": True, "shape": "sphere", "radii_um": [100.0]}
    run_dir = tmp_path / "run"
    result = Experiment(config_from_dict(data)).run(run_dir)
    assert result.status == "completed"
    return run_dir


def test_no_simulation_module_imports_usd():
    offenders = [
        p.relative_to(SRC) for p in SRC.rglob("*.py")
        if "pxr" in p.read_text() and p.name not in ("export.py", "export_usd.py")
    ]
    assert offenders == []


def test_export_matches_checkpoints(tmp_path):
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom

    from warpbiocell.io.export import export_usd

    run_dir = _tiny_run(tmp_path, geometry=True)
    checkpoints = sorted(run_dir.glob("checkpoint/state_*.npz"))
    assert len(checkpoints) == 3  # t = 0, 0.5, 1 h

    path = export_usd(run_dir)

    assert path == run_dir / "cells.usdc"
    stage = Usd.Stage.Open(str(path))
    instancer = UsdGeom.PointInstancer(stage.GetPrimAtPath("/World/Cells"))
    samples = instancer.GetPositionsAttr().GetTimeSamples()
    assert samples == [0.0, 0.5, 1.0]
    assert (stage.GetStartTimeCode(), stage.GetEndTimeCode()) == (0.0, 1.0)
    assert UsdGeom.GetStageMetersPerUnit(stage) == 1.0e-6
    for file, t in zip(checkpoints, samples):
        data = load_checkpoint(file)
        n = int(data["count"])
        positions = np.array(instancer.GetPositionsAttr().Get(t))
        assert positions.shape == (n, 3)
        np.testing.assert_allclose(positions, data["position"][:n], atol=1e-5)
        scales = np.array(instancer.GetScalesAttr().Get(t))
        np.testing.assert_allclose(scales[:, 0], data["radius"][:n])
        assert list(instancer.GetIdsAttr().Get(t)) == list(range(n))  # slot ids, stable over time
        color = np.array(UsdGeom.PrimvarsAPI(instancer).GetPrimvar("displayColor").Get(t))
        assert color.shape == (n, 3)
        state = np.array(UsdGeom.PrimvarsAPI(instancer).GetPrimvar("state").Get(t))
        np.testing.assert_array_equal(state, data["cell_state"][:n])
    assert stage.GetPrimAtPath("/World/Domain").IsValid()
    assert stage.GetPrimAtPath("/World/Cells/Prototypes/Cell").IsValid()
    tissue = UsdGeom.Points(stage.GetPrimAtPath("/World/Tissue"))
    surface = np.array(tissue.GetPointsAttr().Get())
    assert 100 < len(surface) < 5000
    assert np.abs(np.linalg.norm(surface, axis=1) - 100.0).max() < 20.0  # on the sphere, within a voxel
    layer_data = stage.GetRootLayer().customLayerData["warpbiocell"]
    meta = json.loads((run_dir / "metadata.json").read_text())
    assert layer_data["name"] == "usd" and layer_data["length_unit"] == "micrometre"
    assert layer_data["git_commit"] == (meta["git_commit"] or "")


def test_export_colour_by_oxygen_and_text_format(tmp_path):
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom

    from warpbiocell.io.export import export_usd

    run_dir = _tiny_run(tmp_path)
    path = export_usd(run_dir, tmp_path / "cells.usda", color_by="oxygen")

    assert path.suffix == ".usda" and path.read_text().startswith("#usda")
    stage = Usd.Stage.Open(str(path))
    instancer = UsdGeom.PointInstancer(stage.GetPrimAtPath("/World/Cells"))
    color = np.array(UsdGeom.PrimvarsAPI(instancer).GetPrimvar("displayColor").Get(0.0))
    oxygen = np.array(UsdGeom.PrimvarsAPI(instancer).GetPrimvar("oxygen").Get(0.0))
    assert color.shape[0] == oxygen.shape[0] and oxygen.max() > 0.0
    assert not stage.GetPrimAtPath("/World/Tissue").IsValid()  # no region in this run
    with pytest.raises(ValueError):
        export_usd(run_dir, tmp_path / "x.usdc", color_by="age")


def test_export_cli(tmp_path, capsys):
    pytest.importorskip("pxr")
    run_dir = _tiny_run(tmp_path)
    assert export_cli.main([str(run_dir)]) == 0
    assert (run_dir / "cells.usdc").exists()
    assert export_cli.main([str(tmp_path / "nothing")]) == 2
    assert "no checkpoint" in capsys.readouterr().err
