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
* 2026-09-22 — Milestone 7, tissue geometry: signed-distance regions from synthetic shapes
  (sphere, ellipsoid, union) or voxel masks (scipy EDT, NIfTI via nibabel, optional `masks`
  extra), seeding with a cell budget, confinement by a wall contact law, oxygen pinned on
  every node outside the tissue, `geometry` config section, cut-away 3-D figure, 95 CPU
  tests, `configs/tumor_in_ellipsoid.yaml` with `docs/results/tumor_in_ellipsoid.md`;
  scale-gap numbers measured in `docs/vision.md`.

## Decisions taken

* Validation path: reproduce a MicroC oxygen-only configuration first, then a published
  spheroid dataset (2026-09-22).
* CUDA: tested at the end, on the user's GPU machine, by uploading the repository there
  (`README.md`, "On a CUDA machine"). No GPU numbers before that.
* Gene regulatory network: last milestone (11), design notes in `docs/vision.md`; no network
  hooks before then (2026-09-22).
* Patient geometry: synthetic shapes first, masks as a thin loader; the tissue boundary is
  rigid until the mechanobiology extension (2026-09-22).

## Open items that need a human decision

* The published spheroid dataset for quantitative validation (cell line, medium O₂, growth
  curve, viable-rim thickness, necrosis onset diameter).
* A real segmentation mask (NIfTI) to exercise `geometry.shape: mask` end to end; only
  synthetic voxel masks have been used so far.
* Milestone 8 needs `usd-core` (pip, ~50 MB) for OpenUSD export; the Isaac for Healthcare
  demonstrator cannot be built on the development Mac — decide whether Milestone 8 stops at
  the USD export (proposal: yes, the Isaac part waits for a machine that has it).

## Next: Milestone 8 — OpenUSD export (Isaac demonstrator deferred)

docs/roadmap.md "Cellular state → USD".

1. `io/export.py`: write a USD stage with a `UsdGeomPointInstancer` (one prototype sphere,
   per-cell positions and scales, per-cell colour from state or oxygen as a primvar) and one
   time sample per checkpoint, plus the tissue surface as a mesh (marching cubes of the SDF
   is more than needed: export the SDF zero level as a point cloud or leave it to the
   consumer) — start with cells only.
2. Optional `usd` extra (`usd-core`); export skipped with a clear message when absent.
3. `python -m warpbiocell.run ... --set output.usd=true` or a post-processing command
   `python -m warpbiocell.export_usd runs/<dir>` that reads the checkpoints (preferred: keeps
   export out of the simulation loop, AGENTS.md "Design principles").
4. Tests: a two-checkpoint run exports a stage whose point count and time samples match the
   checkpoints (skipped without `usd-core`); no simulation module imports USD.
5. Then Milestone 9 (metabolic fields: glucose, lactate; MicroC parameters are recorded) or
   Milestone 6 as soon as the CUDA machine is available.
