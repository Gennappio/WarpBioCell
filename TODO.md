# Current task

Replace this file's content when the task is done.

## Done

* 2026-09-22 — architecture spike (Milestones 0–2): `docs/architecture_spike.md`, cells +
  HashGrid + overdamped repulsion, CPU benchmarks.
* 2026-09-22 — Milestone 3, cell lifecycle: per-cell RNG streams, stochastic division with
  prefix-sum slot allocation, contact inhibition, `cell_step` operator splitting.
* 2026-09-22 — Milestone 4, oxygen: 3-D grid, deterministic deposit, red-black SOR steady
  state with Michaelis–Menten uptake, FTCS reference, hypoxia / oxygen-dependent division /
  anoxic death; numerical, computational and biological tests.
* 2026-09-22 — Milestone 5, experiment runner: YAML config, `Experiment(config).run(dir)`,
  `python -m warpbiocell.run`, run directories, CUDA-vs-CPU test set; first quantitative
  analysis in `docs/results/spheroid_baseline.md`; MicroC parameters recorded in
  `docs/reference/microc_parameters.md` with `configs/microc_oxygen.yaml`.
* 2026-09-22 — Milestone 5.1: sweep runner, oxygen-boundary sensitivity study
  (`docs/results/oxygen_boundary_sweep.md`), `docs/model.md`, GitHub Actions CI.
* 2026-09-22 — Milestone 7, tissue geometry: SDF regions (synthetic shapes, voxel masks),
  seeding with a cell budget, wall confinement, tissue-surface oxygen source,
  `configs/tumor_in_ellipsoid.yaml` + `docs/results/tumor_in_ellipsoid.md`, measured
  scale gap in `docs/vision.md`.
* 2026-09-22 — Milestone 8 (USD part): `python -m warpbiocell.export_usd <run dir>` writes
  a point-instancer stage from the checkpoints; `usd` extra.
* 2026-09-22 — Milestone 9 (glucose and lactate): per-state cell densities shared by all
  species, `SpeciesField` with per-state uptake and production from a source species,
  `MetabolicFields` manager, `species:` config list, glucose thresholds and MicroC's
  necrosis rule in the lifecycle, residual relative to the local field magnitude, species
  metrics/profile columns/figure/USD primvars, 109 CPU tests,
  `configs/tumor_spheroid_metabolic.yaml` + `docs/results/spheroid_metabolic.md`.

## Decisions taken

* Validation path: reproduce a MicroC oxygen-only configuration first, then a published
  spheroid dataset (2026-09-22).
* CUDA: tested at the end, on the user's GPU machine, by uploading the repository there
  (`README.md`, "On a CUDA machine"). No GPU numbers before that.
* Gene regulatory network: last milestone (11), design notes in `docs/vision.md`; no network
  hooks before then (2026-09-22).
* Patient geometry: synthetic shapes first, masks as a thin loader; the tissue boundary is
  rigid until the mechanobiology extension. A real NIfTI segmentation is not needed now.
* Milestone 8 stops at the OpenUSD export; the Isaac demonstrator waits for a machine with
  Isaac (2026-09-22).
* Milestone 9: glucose and lactate as species with per-state rates; MicroC's file value for
  glucose consumption (3e-15 mol/cell/s) is not used, the stoichiometric rates are
  (2026-09-22).

## Open items that need a human decision

* The published spheroid dataset for quantitative validation (cell line, medium O₂, growth
  curve, viable-rim thickness, necrosis onset diameter).
* Whether the remaining Milestone 9 items (lactate uptake / MCT1, pH, multiple cell types)
  are worth doing as rules, or should wait for the network (Milestone 11) where MicroC
  defines them. Proposal: wait.
* Milestone 10 scope (below): confirm or redirect.

## Next: Milestone 10 — the simulator as a tool (OpenCellComms interface)

docs/roadmap.md "Expose the simulation as a scientific tool". The runner, sweeps and
`Experiment` exist; what is missing is a machine-readable contract an agent can use without
reading Python:

1. `python -m warpbiocell.describe`: the configuration schema as JSON Schema (generated from
   the dataclasses, with units and provenance labels from the YAML comments where possible),
   the list of metrics with definitions and units, and the list of bundled configs and sweeps.
2. `warpbiocell.api`: `run(config: dict | path, overrides, out_dir) -> summary dict`,
   `sweep(base, spec, out_dir) -> rows`, `analyze(run_dir) -> onsets + final metrics`, all
   returning plain JSON-serialisable dicts; the CLI commands call these.
3. An experiment request format (`requests/*.yaml`): hypothesis, base config, overrides or
   grid, seeds, observables, expected measurement — the provenance fields AGENTS.md
   "Agent-generated experiments" requires — executed by `python -m warpbiocell.request`.
4. Tests for the schema (every config key documented), the API round trip, and a request
   run end to end.
5. Do not add any LLM call: the agent lives outside (OpenCellComms); WarpBioCell only
   offers the contract.

Then Milestone 6 as soon as the CUDA machine is available; Milestone 11 last.
