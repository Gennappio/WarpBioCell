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
* 2026-09-22 — Milestone 5, experiment runner: YAML config with unit-named keys and strict
  schema (`simulation/config.py`), `Experiment(config).run(dir)`, `python -m warpbiocell.run`
  with `--set` overrides, run directory (config, metadata, metrics.csv, profiles.csv,
  checkpoints, figures), 73 CPU tests, CUDA-vs-CPU test set ready for the GPU machine.
  First quantitative analysis in `docs/results/spheroid_baseline.md`: necrotic core with a
  stable 125 µm viable rim at 38 mmHg; hypoxia onset at 580 µm diameter at 150 mmHg.

## Decisions taken

* Validation path: reproduce a MicroC oxygen-only configuration first, then a published
  spheroid dataset (agreed 2026-09-22).
* CUDA: tested at the end, on the user's GPU machine, by uploading the repository there
  (`README.md`, "On a CUDA machine"). No GPU numbers before that.

## Open items that need a human decision

* MicroC's exact oxygen-only parameter set (Table C of their S1 Text is not in the repository):
  needed for the like-for-like 2-D slab reproduction.

## Next: Milestone 5.1 — first scientific experiment (oxygen-boundary sensitivity)

docs/roadmap.md "First sensitivity study", runnable on the CPU in minutes:

1. `python -m warpbiocell.sweep --config configs/tumor_spheroid.yaml --sweep configs/sweeps/oxygen_boundary.yaml`:
   a sweep file is a list of override sets plus a seed list; every run is an ordinary run
   directory under `runs/<sweep>/`, and `summary.csv` collects the final metrics per run.
2. Study: `oxygen.boundary_mmHg` in {20, 38, 60, 100, 150} × 3 seeds, 7 days (at 150 mmHg
   12 days). Measure final viable count, necrotic fraction, spheroid radius, hypoxia-onset
   radius, radial oxygen gradient, plus the seed-to-seed spread the results doc flags as
   unmeasured.
3. `docs/results/oxygen_boundary_sweep.md` with a figure of onset radius vs boundary oxygen
   against the zero-order estimate.
4. `docs/model.md`: equations, operator splitting, every assumption and parameter provenance
   in one place (the MVP-completion item "documentation explains equations and assumptions").
5. GitHub Actions: `pytest` on CPU on every push (Milestone 0 leftover).
6. MicroC 2-D slab configuration (`oxygen.grid` with Neumann faces in z and a one-cell-thick
   layer) once the parameter set is available.

Then Milestone 7 (patient geometry) — Milestone 6 (performance) waits for the CUDA machine.
