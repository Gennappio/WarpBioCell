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
* 2026-09-22 — Milestone 11 (gene network per cell): `network/` (expression parser, MaBoSS
  `.bnd/.cfg` and BoolNet `.bnet` readers, `NetworkRuntime`), `kernels/network_kernels.py`
  (bit-packed states, postfix evaluator, asynchronous / synchronous / MaBoSS-Gillespie
  updates, input clamps, fate flags, inheritance), lifecycle `phenotype_model: network`,
  `network:` config section, fate metrics columns, checkpoints, 34 network tests + CUDA
  comparison, `configs/networks/microc_jaya.{bnd,cfg}` (from OpenCellComms),
  `configs/tumor_spheroid_network.yaml` + `docs/results/spheroid_network.md`.

* 2026-09-24 — **warpfvm**, a FiPy-style finite-volume package for MicroC (separate subproject in
  `warpfvm/`, own pyproject): FiPy's grids, numbering, CellVariable/constrain, Diffusion /
  ImplicitSource / Transient terms and explicit sources, FiPy's term algebra, solve/sweep, FiPy's
  solver names; Warp stencil assembly and a device-side preconditioned CG. Assembled systems
  equal FiPy's to 1e-12 (85 cases), MicroC's `update` flow (steady, transient, gradient
  boundaries, Picard coupling) agrees with FiPy to round-off; 2nd-order convergence checked;
  CPU benchmark and validation in `warpfvm/docs/validation.md`; GPU tests written, not run.

## Decisions taken

* Validation path: reproduce a MicroC oxygen-only configuration first, then a published
  spheroid dataset (2026-09-22).
* CUDA: tested at the end, on the user's GPU machine, by uploading the repository there
  (`README.md`, "On a CUDA machine"). No GPU numbers before that.
* Gene regulatory network: last milestone (11), design notes in `docs/vision.md`; done with
  the MicroC network taken from the OpenCellComms `MicroC` adapter (MaBoSS files, unit
  rates). MicroC's NetLogo graph-walk propagation is not reproduced (2026-09-22).
* Patient geometry: synthetic shapes first, masks as a thin loader; the tissue boundary is
  rigid until the mechanobiology extension. A real NIfTI segmentation is not needed now.
* Milestone 8 stops at the OpenUSD export; the Isaac demonstrator waits for a machine with
  Isaac (2026-09-22).
* Milestone 9: glucose and lactate as species with per-state rates; MicroC's file value for
  glucose consumption (3e-15 mol/cell/s) is not used, the stoichiometric rates are
  (2026-09-22). The remaining Milestone 9 items (MCT1 / lactate uptake, pH, multiple cell
  types) waited for the gene network; now that it exists they are open work items, not
  blocked (2026-09-22).
* Milestone 10 is the OpenCellComms adapter; no LLM call inside WarpBioCell (2026-09-22).

* warpfvm (2026-09-24): lives in this repository as a separate package, not inside
  `warpbiocell`; every solver name runs preconditioned CG (the systems are symmetric);
  `LinearLUSolver` reproduces FiPy's early return when the start already meets the tolerance;
  its own CG loop instead of `warp.optim.linear` (1000x slower dot products on the CPU);
  CUDA-graph replay off by default until validated on hardware.

## Open items that need a human decision

* The published spheroid dataset for quantitative validation (cell line, medium O₂, growth
  curve, viable-rim thickness, necrosis onset diameter).
* The `MicroC_warp` adapter is created but **not committed** in the OpenCellComms repository
  (that repository has unrelated uncommitted changes of yours); commit it there when ready.
  Rebuilt on 2026-09-22 as atomic nodes in the MicroC adapter's ABM structure (the engine's
  scheduler owns the loop, one iteration = one cell step, full per-node observability);
  `docs/integrations/opencellcomms.md` lists the library calls it depends on.
* In network mode the division rate is the rate *while Proliferation is ON* (~20 % of the
  cells at stationarity with the MicroC network); `configs/tumor_spheroid_network.yaml` uses
  0.1/h for an effective ~0.02/h — an illustrative choice to confirm or replace.

* Switching MicroCpy to warpfvm (two import lines in `multi_substance_simulator.py`,
  one in `core/domain.py`) is a change in the OpenCellComms repository: not done.
* FiPy's LU early return can leave a MicroC field unchanged when its residual is below
  1e-6 ||b|| (seen in 2-D test problems); check MicroC's real 2-D source magnitudes before
  relying on either library's LU there.
* Upstream candidate: `warp.optim.linear` CPU slowness (docs/upstream.md), not filed.

## Next

* Milestone 6 — performance on the CUDA machine (README "On a CUDA machine"): run the gpu
  tests, the three benchmarks at 10³–10⁶ cells, commit `benchmarks/results/*_cuda0.json`,
  then profile the contact kernel and the SOR sweeps and decide whether anything merits an
  upstream Warp issue (docs/upstream.md).
* Network follow-ups: MCT1 lactate uptake and pH as species the network can read
  (`MCT1_stimulus` already reads lactate); a comparison of the `maboss` mode against the
  MaBoSS binary on the MicroC network (the closed-form checks stand in for now).
* warpfvm on the CUDA machine: `pytest warpfvm/tests -m gpu -v`, `warpfvm/benchmarks/bench_fvm.py`
  in float64 and float32, commit `warpfvm/benchmarks/results/fvm_cuda0_*.json`; then decide the
  `use_cuda_graph` default and whether a multigrid preconditioner is worth it (CG iterations
  grow as ~3n: 280 at 96^3).
