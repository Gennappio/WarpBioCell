# Tumour filling an ellipsoidal tissue region (Milestone 7 demonstrator)

Run: `configs/tumor_in_ellipsoid.yaml` (seed 0, CPU, 2026-09-22). Ellipsoid with semi-axes
250 × 200 × 150 µm inside an 800 µm box, filled at initialisation with 6 713 cells (jittered
lattice at one diameter spacing, whole cells inside the surface), confined by the surface with
the wall contact law, oxygen held at 38 mmHg on every grid node outside the tissue. 4 simulated
days, 18.6 s wall time. Figures in the run directory (`oxygen_slice.png` shows the surface
contour, `cells_3d.png` the cut-away).

| day | cells | proliferative | hypoxic | dead | O₂ min at cells [mmHg] | cells outside tissue |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 6 713 | 6 713 | 0 | 0 | 9.9 | 0 |
| 1 | 9 353 | 2 071 | 0 | 0 | 8.1 | 0 |
| 2 | 10 388 | 1 034 | 190 | 0 | 6.2 | 0 |
| 3 | 10 973 | 623 | 405 | 0 | 5.2 | 0 |
| 4 | 11 355 | 456 | 526 | 0 | 4.7 | 0 |

What it shows:

* **Seeding, confinement and the tissue-surface oxygen source work together.** No cell centre
  ever leaves the region; the oxygen decreases from 38 mmHg at the surface to 4.7 mmHg in the
  core, i.e. the core of a 150 µm semi-minor axis becomes hypoxic, consistent with the
  ~145 µm hypoxia-onset radius of the free spheroid at the same boundary oxygen
  (`docs/results/oxygen_boundary_sweep.md`).
* **Mechanically constrained growth emerges.** With a rigid boundary the population can only
  grow by compressing: 6 713 → 11 355 cells in a fixed volume (1.7× the seeding density), the
  neighbour counts rise and contact inhibition shuts proliferation down (6 713 → 456
  proliferative cells). Growth saturates without any explicit carrying capacity.

Caveats specific to this configuration:

* The boundary is rigid and the cell–cell law is a soft linear repulsion, so the compressed
  state has overlapping cells: a physically loose regime. A real tissue would deform; the
  two-way coupling with tissue mechanics is the mechanobiology extension in docs/vision.md.
* "Oxygen at the tissue surface" means every voxel outside the region is a perfect source.
  It is the simplest configurable choice, not a vascular model.
* The ellipsoid stands in for a segmented lesion; `geometry.shape: mask` with a NIfTI file
  goes through the same path (`geometry/masks.py`, optional `scipy` + `nibabel`) but has only
  been exercised on synthetic voxel masks in the tests.
