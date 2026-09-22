# OpenCellComms integration (Milestone 10)

WarpBioCell is exposed to OpenCellComms as the **`MicroC_warp` adapter**
(`opencellcomms_adapters/MicroC_warp/` in the OpenCellComms repository, MicroCpy). The
adapter lives there, next to the other plugins, because it follows OpenCellComms' plugin
contract; WarpBioCell itself contains no OpenCellComms code and no LLM calls (AGENTS.md).

## What the adapter offers

GUI-visible nodes that assemble a WarpBioCell configuration in the engine context and run it:

| Node | Fills / does |
|---|---|
| `define_warp_domain` | `simulation`, `cells`, `mechanics`, `oxygen.grid` |
| `define_warp_substance` (cloneable) | `oxygen` or a `species[]` entry (per-state uptake, production, source) |
| `define_warp_cell_rules` | `lifecycle` (division, inhibition, thresholds, MicroC necrosis rule) |
| `define_warp_tissue` | `geometry` (sphere / ellipsoid / union / mask, confinement, oxygen source) |
| `run_warpbiocell_simulation` | validates with `config_from_dict`, runs `Experiment.run`, streams `[OCC_EVENT]` JSON lines |
| `run_warpbiocell_sweep` | `Sweep` over a grid of dotted keys × seeds, `summary.csv` with onsets |
| `summarize_warpbiocell_run` | report + `context['warpbiocell_summary']` |

plus a `warpbiocell` facade kernel for workflows that carry the whole configuration in
`metadata.warpbiocell`. Three workflows ship with it: the metabolic spheroid, the tumour in
an ellipsoidal tissue, and the oxygen-boundary sweep.

## The contract the adapter relies on (keep stable)

* `warpbiocell.simulation.config.config_from_dict(dict) -> ExperimentConfig` and
  `ExperimentConfig.validate() -> warnings`; unknown keys raise `ConfigError`.
* `warpbiocell.simulation.experiment.Experiment(config, device=None)` with
  `.population.count`, `.device.alias`, and `.run(output_dir, log=..., on_metrics=...)`
  returning `RunResult(status, metrics, summary, wall_s)`.
* `warpbiocell.metrics.timeseries.onset_summary(rows)`.
* `warpbiocell.simulation.sweep.SweepSpec` / `Sweep(spec, base_dict).run(dir, device, figures, log)`.
* Metrics column names of `io/run_output.py` (`cells`, `living_cells`, `dead_cells`,
  `hypoxic_cells`, `proliferative_cells`, `quiescent_cells`, `spheroid_radius_um`,
  `oxygen_cells_min_mmHg`, `glucose_cells_min_mM`, `lactate_cells_max_mM`, `wall_s`, ...).

A change to any of these must be mirrored in the adapter (`backend/warpbiocell_backend.py`).

## Installing

The adapter's `requirements.txt` installs WarpBioCell into the OpenCellComms environment:

```text
warpbiocell @ git+https://github.com/Gennappio/WarpBioCell.git
```

(`install.sh` / `install.bat` / the Docker image pick it up.) For development, an editable
install of this checkout into the OpenCellComms `.venv` works as well.

## Running

```bash
cd MicroCpy
python opencellcomms_engine/run_workflow.py --workflow opencellcomms_adapters/MicroC_warp/workflows/microc_warp_spheroid.json
```

Results: `runs/<workflow>/warp_run/warpbiocell/` — an ordinary WarpBioCell run directory
plus `occ_events.jsonl`. Verified on 2026-09-22 (CPU): the spheroid workflow reproduces
`docs/results/spheroid_metabolic.md` exactly (10 368 cells, 2 dead at day 7, 103 s).
