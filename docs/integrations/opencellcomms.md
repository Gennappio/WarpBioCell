# OpenCellComms integration (Milestone 10)

WarpBioCell is exposed to OpenCellComms as the **`MicroC_warp` adapter**
(`opencellcomms_adapters/MicroC_warp/` in the OpenCellComms repository, MicroCpy). The
adapter lives there, next to the other plugins, because it follows OpenCellComms' plugin
contract; WarpBioCell itself contains no OpenCellComms code and no LLM calls (AGENTS.md).

## What the adapter offers

Atomic, GUI-visible nodes in the ABM structure of OpenCellComms' own MicroC adapter: the
engine's `__scheduler__` owns the loop (one iteration = one WarpBioCell cell step) and every
node wraps exactly one library call, so each law is implemented once, here, and every phase
of every iteration is a node the engine's observability records (events, durations, context
reads/writes). There is no run-everything node and no facade kernel (decided 2026-09-22:
the first adapter collapsed the simulation into one node, which hid the loop from the
canvas; it was rebuilt).

| Canvas | Node | Library call |
|---|---|---|
| World | `setup_warp_world` | `GridGeometry.centered_cube`, `TimeStepping`, `SolverSettings`, `RunOutput` |
| World | `define_warp_tissue` (optional) | `TissueRegion.from_shape` / `region_from_mask` |
| World | `define_warp_mechanics` | `ContactParams`, `make_neighbor_grid` |
| Resources (one per substance) | `setup_warp_substance` | `MetabolicFields(...)` for oxygen, `MetabolicFields.add_species` for the others |
| Agent creation | `seed_warp_cells` | `spherical_cluster` / `fill_region`, `CellPopulation.from_numpy`, `contact_substep(dt=0)` |
| Agent creation | `init_warp_gene_networks` (optional) | `load_network`, `NetworkRuntime(...)`, `.initialize` |
| World step | `solve_warp_fields` | `MetabolicFields.update` |
| Agent step | `clamp_warp_network_inputs`, `propagate_warp_network`, `read_warp_fates` | `NetworkRuntime.clamp_inputs`, `.update(..., updates_per_step, time_units_per_h)`, `.read_fates` |
| Agent step | `decide_warp_lifecycle` | `decide_lifecycle` |
| World step | `commit_warp_divisions`, `inherit_warp_networks` | `commit_divisions`, `NetworkRuntime.inherit` |
| World step | `relax_warp_contacts` | `relax_contacts` |
| Agent step | `record_warp_census` | `simulation.experiment.observe` |
| World step | `write_warp_profile`, `save_warp_checkpoint` | `radial_profile`, `save_checkpoint` |
| Processing | `summarize_warpbiocell_run` | `make_run_figures`, `onset_summary`, `RunOutput.write_config`, `.finish` |

The scheduler order is the operator splitting of `cell_step`: fields → gene network →
lifecycle decisions → divisions → mechanics, then the census and the outputs. Sweeps are
Planner tabs (one configuration per tab, replicates = seeds), not a node. Four workflows
ship with it: the metabolic spheroid, the tumour in an ellipsoidal tissue, the spheroid
with MicroC's gene network in every cell, and the oxygen-boundary sweep as Planner tabs.

## The contract the adapter relies on (keep stable)

* The library functions in the table above, with their signatures: in particular
  `lifecycle_step` = `decide_lifecycle(population, params, dt)` +
  `commit_divisions(population, params) -> int` (raises `CapacityError`),
  `MetabolicFields.add_species(SpeciesParams)`, `NetworkRuntime.update(population, dt_h,
  updates_per_step=None, time_units_per_h=None)`, `NetworkRuntime.inherit(population,
  count_before)`, `NetworkRuntime.fate_counts`, `simulation.experiment.observe(population,
  oxygen, time_h, shell_width, region, network, wall_s, field_sweeps) -> (row, profile)`,
  `RunOutput(directory, None, config_dict, device, seed)` with `.write_metrics`,
  `.write_profile`, `.checkpoint_path`, `.write_config`, `.finish`.
* The parameter dataclasses (`ContactParams`, `LifecycleParams`, `OxygenParams`,
  `SpeciesParams` + `by_state`, `NetworkParams` + `InputClamp`, `TimeStepping`,
  `SolverSettings`, `GridGeometry`) validate their own ranges with `ValueError`.
* `CellPopulation.from_numpy(positions, radii, capacity, device, seed)` takes a 31-bit seed
  (the adapter folds the engine's run seed).
* `warpbiocell.metrics.timeseries.onset_summary(rows)`.
* Metrics column names of `io/run_output.py` (`cells`, `living_cells`, `dead_cells`,
  `hypoxic_cells`, `proliferative_cells`, `quiescent_cells`, `spheroid_radius_um`,
  `oxygen_cells_min_mmHg`, `glucose_cells_min_mM`, `lactate_cells_max_mM`,
  `network_{proliferation,apoptosis,growth_arrest,necrosis}_cells` (empty without a
  network), `wall_s`, ...).

A change to any of these must be mirrored in the adapter's nodes
(`opencellcomms_adapters/MicroC_warp/functions/`).

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

Results: `runs/<workflow>_<timestamp>/<tab>/replicate-NNN/warpbiocell/` — an ordinary
WarpBioCell run directory (config.yaml with every node's parameters, metadata.json with the
run seed and the folded WarpBioCell seed, metrics.csv, profiles.csv, checkpoints, figures)
plus `occ_events.jsonl`. The seed is the engine's Planner seed, so the numbers differ from
the seed-0 CLI runs in `docs/results/` statistically only; the same workflow and seed replay
bitwise on the same device.
