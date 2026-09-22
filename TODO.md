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
* 2026-09-22 — Milestone 5.1: sweep runner (`python -m warpbiocell.sweep`), oxygen-boundary
  sensitivity study (`docs/results/oxygen_boundary_sweep.md`), `docs/model.md`, GitHub
  Actions CI.

## Decisions taken

* Validation path: reproduce a MicroC oxygen-only configuration first, then a published
  spheroid dataset (2026-09-22).
* CUDA: tested at the end, on the user's GPU machine, by uploading the repository there
  (`README.md`, "On a CUDA machine"). No GPU numbers before that.
* Gene regulatory network: last milestone (11), design notes in `docs/vision.md`; no network
  hooks before then (2026-09-22).

## Open items that need a human decision

* The published spheroid dataset for quantitative validation (cell line, medium O₂, growth
  curve, viable-rim thickness, necrosis onset diameter).
* Whether Milestone 7 should start with NIfTI masks (needs `nibabel`, a new dependency) or
  with synthetic geometries (a mesh or an SDF) to build the seeding and confinement
  machinery first. Default proposal: synthetic first, NIfTI as a thin loader afterwards.

## Next: Milestone 7 — patient geometry (Milestone 6 waits for the CUDA machine)

docs/roadmap.md; the scale-gap question in docs/vision.md must be answered as part of it.

1. Geometry representation: a signed-distance field on the same kind of regular grid as the
   oxygen field (`fields/scalar_field.py`), built from a synthetic shape first (sphere,
   ellipsoid, union of spheres) and later from a segmentation mask.
2. Seeding: fill the region `sdf < 0` with cells at a target packing (jittered lattice, as
   `spherical_cluster` does), with an option to seed only a sub-volume.
3. Confinement: a boundary force in the mechanics kernel from the sampled SDF gradient
   (cells pushed back inside), documented as a contact law like the cell–cell one.
4. Field domain: Dirichlet oxygen on the tissue boundary voxels rather than on the box faces
   (a "vessel at the tissue surface" first approximation), configurable.
5. Tests: seeding density and containment, SDF sampling exactness for a sphere, a cell
   pushed out of the region returns, spheroid results unchanged when the region is a large
   sphere.
6. Scale gap: measure cells per mm³ at the current packing and state in `docs/vision.md`
   what sub-volume a 10⁶-cell budget covers; decide coarse-graining later, with data.
