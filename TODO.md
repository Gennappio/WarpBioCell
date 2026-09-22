# Current task

Replace this file's content when the task is done.

## Done

* 2026-09-22 — architecture spike (Milestones 0–2): `docs/architecture_spike.md`, cells +
  HashGrid + overdamped repulsion, CPU benchmarks.
* 2026-09-22 — Milestone 3, cell lifecycle: per-cell RNG streams, stochastic division with
  prefix-sum slot allocation, contact inhibition, `cell_step` operator splitting.
* 2026-09-22 — Milestone 4, oxygen: regular 3-D grid with per-axis Dirichlet/Neumann
  boundaries, deterministic fixed-point trilinear deposit, red-black SOR steady-state solver
  with linearised Michaelis–Menten uptake, FTCS reference, trilinear sampling, hypoxia /
  oxygen-dependent division / anoxic death in the lifecycle, oxygen metrics and radial
  profile. 58 CPU tests (numerical, computational, biological kept separate),
  `benchmarks/results/field_cpu.json`, `examples/spheroid_oxygen.py` shows growth → gradient
  → hypoxia → necrotic core with a viable rim, all emergent.

## Open items that need a human decision

* Which CUDA machine provides GPU numbers (AGENTS.md, Device policy). Nothing has run on
  CUDA yet; `tests/test_gpu.py` covers mechanics only and should gain field and lifecycle
  cases once a device exists.
* Validation data for Milestone 5: which published spheroid dataset (growth curve, viable-rim
  thickness, necrosis onset diameter, cell line, medium O2) the quantitative comparison
  targets. Candidate: reproduce a MicroC oxygen-only configuration first.

## Next: Milestone 5 — tumor spheroid experiment

The coupled model exists; this milestone makes it a reproducible experiment.

1. `simulation/config.py`: one YAML → dataclasses (simulation, cells, mechanics, lifecycle,
   oxygen, grid, output). Units in every key name or comment. Validate: grid contains the
   initial cluster with margin; `rate * dt_mechanics`; capacity vs expected growth.
2. `python -m warpbiocell.run --config configs/tumor_spheroid.yaml` producing
   `runs/<timestamp>/{config.yaml, metadata.json, metrics.csv, checkpoint/, figures/}`;
   metadata = config, seed, package version + git commit, Warp version, device, timestamp.
3. Metrics per step to CSV (AGENTS.md "Outputs" list) plus periodic radial profiles;
   checkpoint = `.npz` of the population arrays and the field (restart not required yet).
4. Figures (matplotlib, optional dependency): population curves, radial oxygen and state
   profiles, a mid-plane oxygen slice with cell positions.
5. First quantitative analysis on the default configuration: growth curve, viable-rim
   thickness, hypoxic and necrotic fractions vs time; compare qualitatively with MicroC and
   with the zero-order diffusion estimate of the critical radius.
6. `python -m warpbiocell.sweep` can wait; but design the config so a sweep is a list of
   overrides.

Do not start patient geometry or USD export before the experiment runner exists.
