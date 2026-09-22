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
  a point-instancer stage from the checkpoints (positions, radii, stable ids, colour by
  state or oxygen, tissue surface, grid box; micrometre units, one time code per hour);
  `usd` extra; checkpoints now carry the tissue SDF; 99 CPU tests.

## Decisions taken

* Validation path: reproduce a MicroC oxygen-only configuration first, then a published
  spheroid dataset (2026-09-22).
* CUDA: tested at the end, on the user's GPU machine, by uploading the repository there
  (`README.md`, "On a CUDA machine"). No GPU numbers before that.
* Gene regulatory network: last milestone (11), design notes in `docs/vision.md`; no network
  hooks before then (2026-09-22).
* Patient geometry: synthetic shapes first, masks as a thin loader; the tissue boundary is
  rigid until the mechanobiology extension. A real NIfTI segmentation is **not** needed now:
  the mask path is covered by synthetic voxel masks and a nibabel roundtrip; the loader's
  known limitation (affine orientation ignored) is documented and waits for an imaging
  pipeline (2026-09-22).
* Milestone 8 stops at the OpenUSD export; the Isaac for Healthcare demonstrator waits for a
  machine that has Isaac (2026-09-22).

## Open items that need a human decision

* The published spheroid dataset for quantitative validation (cell line, medium O₂, growth
  curve, viable-rim thickness, necrosis onset diameter).
* Milestone 9 scope: the proposal below adds glucose as a second field with a rule-based
  metabolic phenotype (no network). Confirm or redirect.

## Next: Milestone 9 — metabolic fields (glucose first)

docs/roadmap.md "Advanced biology"; MicroC's `diffusion-parameters.txt` is the reference set
(`docs/reference/microc_parameters.md`: glucose D = 6.7e-11 m²/s, uptake 3e-15 mol/cell/s,
5 mM boundary, 4 mM activation threshold; lactate produced at 3e-15 mol/cell/s, 1 mM boundary).

1. Generalise the field machinery to several species: `fields/species.py` with a list of
   `SpeciesField` (name, unit, kinetics, boundary), each reusing `ScalarField`, the deposit and
   the SOR solver; per-cell sampled values in a `(capacity, n_species)` array or one array per
   species (prefer one array per species: the kernels stay simple).
2. Per-state consumption and production: a small table `rate[state, species]` so that hypoxic
   (glycolytic) cells consume more glucose and produce lactate while oxygenated cells
   consume oxygen — the rule-based stand-in for MicroC's metabolic network, labelled as such.
3. Lifecycle: MicroC's necrosis rule (oxygen **and** glucose below their thresholds) as an
   option next to the current oxygen-only death; glucose-limited division factor.
4. Units: glucose and lactate in mM; document the conversion of MicroC's mol/cell/s rates.
5. Tests: two-species steady state against the dense exact solution, conservation of the
   produced lactate flux, necrosis rule, and the spheroid with glucose (viable rim thinner
   when glucose is limiting).
6. Results note: glucose-limited vs oxygen-limited spheroid, compared qualitatively with
   MicroC's Fig. 6 (glucose and lactate profiles).

Milestone 6 (performance) runs as soon as the CUDA machine is available; Milestone 10
(OpenCellComms) and 11 (gene network) stay last.
