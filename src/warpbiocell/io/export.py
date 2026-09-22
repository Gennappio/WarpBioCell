"""OpenUSD export of a run's checkpoints (optional dependency ``usd-core``, extra ``usd``).

Post-processing only: reads ``checkpoint/state_*.npz`` files, never the simulation objects,
so no simulation module depends on USD (AGENTS.md "Design principles").

Stage layout:

    /World                         Xform; stage metadata in customLayerData
    /World/Cells                   UsdGeomPointInstancer, one time sample per checkpoint
    /World/Cells/Prototypes/Cell   unit sphere prototype; per-instance scales carry the radii
    /World/Domain                  the grid box (guide purpose), when the checkpoint has a grid
    /World/Tissue                  UsdGeomPoints on the tissue surface (|sdf| < dx/2), when a region exists

Per instance and per time sample: ``positions`` [um], ``scales`` (radius), ``ids`` (the cell
slot, stable for the whole run), ``primvars:displayColor`` (by state, or by oxygen with
``color_by="oxygen"``), ``primvars:state`` (int), ``primvars:oxygen`` (mmHg). Units: metres
per unit = 1e-6 (micrometres); one time code = one simulated hour, 24 time codes per second
so that one playback second is one simulated day. The default file is binary ``cells.usdc``;
pass a ``.usda`` path for a human-readable stage.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from warpbiocell.cells.model import CellState
from warpbiocell.io.checkpoints import load_checkpoint

STATE_COLORS = {
    CellState.PROLIFERATIVE: (0.165, 0.616, 0.561),
    CellState.QUIESCENT: (0.914, 0.769, 0.416),
    CellState.HYPOXIC: (0.957, 0.635, 0.380),
    CellState.DEAD: (0.424, 0.459, 0.490),
}
TIME_CODES_PER_SECOND = 24.0  # 1 time code = 1 h; one playback second = one simulated day


def _require_pxr():
    try:
        from pxr import Gf, Sdf, Usd, UsdGeom, Vt  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError("USD export needs usd-core: pip install -e '.[usd]'") from exc
    return Gf, Sdf, Usd, UsdGeom, Vt


def checkpoint_files(run_dir: str | Path) -> list[Path]:
    """Checkpoints of a run directory in time order."""
    files = sorted(Path(run_dir).glob("checkpoint/state_*.npz"))
    if not files:
        raise FileNotFoundError(f"no checkpoint/state_*.npz under {run_dir}")
    return files


def _colors(state: np.ndarray, oxygen: np.ndarray, color_by: str, oxygen_max: float) -> np.ndarray:
    if color_by == "state":
        table = np.array([STATE_COLORS[s] for s in CellState], dtype=np.float32)
        return table[np.clip(state, 0, len(CellState) - 1)]
    if color_by == "oxygen":
        t = np.clip(oxygen / max(oxygen_max, 1e-6), 0.0, 1.0)[:, None].astype(np.float32)
        low, high = np.array([0.95, 0.95, 0.95], dtype=np.float32), np.array([0.03, 0.19, 0.42], dtype=np.float32)
        return low + (high - low) * t
    raise ValueError("color_by must be 'state' or 'oxygen'")


def export_usd(run_dir: str | Path, out_path: str | Path | None = None, color_by: str = "state") -> Path:
    """Write ``<run_dir>/cells.usdc`` (or ``out_path``) from the run's checkpoints; return the path."""
    Gf, Sdf, Usd, UsdGeom, Vt = _require_pxr()
    run_dir = Path(run_dir)
    files = checkpoint_files(run_dir)
    out_path = Path(out_path) if out_path else run_dir / "cells.usdc"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {}
    meta_file = run_dir / "metadata.json"
    if meta_file.exists():
        meta = json.loads(meta_file.read_text())
        metadata = {k: meta.get(k) for k in ("name", "seed", "git_commit", "warpbiocell_version", "warp_version", "status")}
        if meta.get("config", {}).get("oxygen"):
            metadata["oxygen_boundary_mmHg"] = meta["config"]["oxygen"].get("boundary_mmHg")
    oxygen_max = float(metadata.get("oxygen_boundary_mmHg") or 0.0)

    stage = Usd.Stage.CreateNew(str(out_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0e-6)
    stage.SetTimeCodesPerSecond(TIME_CODES_PER_SECOND)
    stage.SetFramesPerSecond(TIME_CODES_PER_SECOND)

    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    instancer = UsdGeom.PointInstancer.Define(stage, "/World/Cells")
    proto = UsdGeom.Sphere.Define(stage, "/World/Cells/Prototypes/Cell")
    proto.GetRadiusAttr().Set(1.0)
    instancer.CreatePrototypesRel().SetTargets([proto.GetPath()])
    primvars = UsdGeom.PrimvarsAPI(instancer)
    color_pv = primvars.CreatePrimvar("displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex)
    state_pv = primvars.CreatePrimvar("state", Sdf.ValueTypeNames.IntArray, UsdGeom.Tokens.vertex)
    oxygen_pv = primvars.CreatePrimvar("oxygen", Sdf.ValueTypeNames.FloatArray, UsdGeom.Tokens.vertex)

    times = []
    domain_written = False
    for file in files:
        data = load_checkpoint(file)
        t = float(data["time_h"])
        n = int(data["count"])
        pos = np.asarray(data["position"], dtype=np.float32)[:n]
        radius = np.asarray(data["radius"], dtype=np.float32)[:n]
        state = np.asarray(data["cell_state"], dtype=np.int32)[:n]
        oxygen = np.asarray(data.get("oxygen_local", np.zeros(n)), dtype=np.float32)[:n]
        if not oxygen_max and oxygen.size:
            oxygen_max = float(oxygen.max())

        instancer.GetPositionsAttr().Set(Vt.Vec3fArray.FromNumpy(pos), t)
        instancer.GetScalesAttr().Set(Vt.Vec3fArray.FromNumpy(np.repeat(radius[:, None], 3, axis=1)), t)
        instancer.GetProtoIndicesAttr().Set(Vt.IntArray.FromNumpy(np.zeros(n, dtype=np.int32)), t)
        instancer.GetIdsAttr().Set(Vt.Int64Array.FromNumpy(np.arange(n, dtype=np.int64)), t)
        color_pv.Set(Vt.Vec3fArray.FromNumpy(_colors(state, oxygen, color_by, oxygen_max)), t)
        state_pv.Set(Vt.IntArray.FromNumpy(state), t)
        oxygen_pv.Set(Vt.FloatArray.FromNumpy(oxygen), t)
        times.append(t)

        if not domain_written and "grid_origin" in data:
            origin = np.asarray(data["grid_origin"], dtype=float)
            dx = float(data["grid_dx"])
            grid = data["field"] if "field" in data else data["tissue_sdf"]
            shape = np.asarray(grid.shape)
            size = dx * (shape - 1)
            cube = UsdGeom.Cube.Define(stage, "/World/Domain")
            cube.GetSizeAttr().Set(1.0)
            xform = UsdGeom.Xformable(cube)
            xform.AddTranslateOp().Set(Gf.Vec3d(*(origin + 0.5 * size)))
            xform.AddScaleOp().Set(Gf.Vec3f(*size))
            cube.GetPurposeAttr().Set(UsdGeom.Tokens.guide)  # outline only, never rendered
            if "tissue_sdf" in data:
                sdf = np.asarray(data["tissue_sdf"], dtype=np.float32)
                idx = np.argwhere(np.abs(sdf) < 0.5 * dx)
                surface = (origin + dx * idx).astype(np.float32)
                points = UsdGeom.Points.Define(stage, "/World/Tissue")
                points.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(surface))
                points.GetWidthsAttr().Set(Vt.FloatArray.FromNumpy(np.full(len(surface), 0.5 * dx, dtype=np.float32)))
                UsdGeom.PrimvarsAPI(points).CreatePrimvar("displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.constant).Set([(0.35, 0.35, 0.35)])
            domain_written = True

    stage.SetStartTimeCode(min(times))
    stage.SetEndTimeCode(max(times))
    stage.GetRootLayer().customLayerData = {
        "warpbiocell": {**{k: ("" if v is None else v) for k, v in metadata.items()}, "length_unit": "micrometre", "time_code_unit": "hour", "color_by": color_by}
    }
    stage.GetRootLayer().Save()
    return out_path
