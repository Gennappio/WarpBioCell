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

* 2026-09-22 — Milestone 10: the `MicroC_warp` adapter in the OpenCellComms repository
  (`opencellcomms_adapters/MicroC_warp/`: manifest, backend, 7 nodes, `warpbiocell` facade
  kernel, 3 workflows, README); all three workflows verified through the OpenCellComms
  engine on the CPU. WarpBioCell side: `Experiment.run(on_metrics=...)`,
  `docs/integrations/opencellcomms.md`.

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
  (2026-09-22). The remaining Milestone 9 items (MCT1 / lactate uptake, pH, multiple cell
  types) wait for the gene network (Milestone 11) — decided 2026-09-22.
* Milestone 10 is the OpenCellComms adapter; no LLM call inside WarpBioCell (2026-09-22).

## Open items that need a human decision

* The published spheroid dataset for quantitative validation (cell line, medium O₂, growth
  curve, viable-rim thickness, necrosis onset diameter).
* The `MicroC_warp` adapter is created but **not committed** in the OpenCellComms repository
  (that repository has unrelated uncommitted changes of yours); commit it there when ready.
* The MicroC network exported from GINsim (MaBoSS `.bnd/.cfg` or BoolNet `.bnet`) for
  Milestone 11.

## Next

* Milestone 6 — performance on the CUDA machine (README "On a CUDA machine"): run the gpu
  tests, the three benchmarks at 10³–10⁶ cells, commit `benchmarks/results/*_cuda0.json`,
  then profile the contact kernel and the SOR sweeps and decide whether anything merits an
  upstream Warp issue (docs/upstream.md).
* Milestone 11 — gene regulatory network per cell (docs/vision.md design notes), once the
  network file is available.
