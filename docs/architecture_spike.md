# Architecture spike

Investigation of NVIDIA Warp 1.17.0 for WarpBioCell, plus the results of the first technical
spike (10k cells + HashGrid + overlap repulsion). Written 2026-09-22 against Warp 1.17.0 on
macOS/arm64 (CPU only). Anything marked *unverified* was read from source or docs but not
executed on CUDA.

## 1. Data representation for 10^5–10^6 dynamic cells

Structure-of-arrays of flat `wp.array`s with preallocated capacity, active cells in the
prefix `[0, count)`. Implemented in `warpbiocell.cells.state.CellPopulation`.

| Field | Type | Bytes/cell | Milestone |
|---|---|---|---|
| position | vec3f | 12 | 1 |
| radius | float32 | 4 | 1 |
| velocity (overdamped, scratch) | vec3f | 12 | 2 |
| contact_count | int32 | 4 | 2 |
| cell_state, cell_type | int32 ×2 | 8 | 3 |
| age, oxygen_local | float32 ×2 | 8 | 3–4 |
| id (birth order) | int32 | 4 | 3 |
| rng state (per-cell counter) | uint32 | 4 | 3 |

≈ 56 bytes per cell → 56 MB for 10^6 cells. Trivial next to any field grid.

Reasons for this layout over alternatives:

* `wp.struct` arrays (AoS) would halve memory coalescing for kernels that touch one field
  and complicate slicing views; Warp's own particle examples use SoA.
* Slicing a 1-D contiguous array (`position[:count]`) is a view, not a copy, so
  `HashGrid.build` can index only the active prefix and the indices it returns coincide with
  global slot indices. Verified: `tests/test_neighbors.py::test_grid_only_indexes_active_prefix`.
* float32 everywhere. Positions in micrometers stay below 10^4, where float32 resolves ~10^-3
  µm; overlaps of 10^-3 µm are already biologically irrelevant. `HashGrid` supports float64 if
  a patient-scale domain (cm) ever needs it.

## 2. Neighbor queries

`wp.HashGrid(dim, dim, dim)` + `wp.hash_grid_query(grid, x, radius)` in kernels, wrapped in
`warpbiocell.spatial.neighbors.NeighborGrid`.

* `dim` is the number of hash buckets per axis, not a spatial extent; points anywhere in space
  are hashed. Bucket bookkeeping costs `2 × 4 bytes × dim³` (16 MB at 128³). Native struct:
  `point_cells`, `point_ids` (per point) and `cell_starts`, `cell_ends` (per bucket).
* Build = compute cell id per point + radix sort + range detection. Measured on CPU:
  0.15 ms at 10k points, 0.68 ms at 100k, i.e. negligible against the contact pass.
* The query visits the 27 buckets around the point, so bucket edge must equal the query
  radius (`2 × r_max + margin`). Candidates per cell ≈ 27 × (query_radius / spacing)³ ≈ 40 at
  10% compression; actual contacts ≈ 6.
* `wp.hash_grid_point_id(grid, tid)` maps thread → point in sorted order so neighbouring
  threads read neighbouring memory. Used in the contact kernel as Warp's DEM example does.
* Gather pattern: each thread sums the forces on its own cell. No atomics, bitwise
  reproducible on a given device. Cost is 2× the pair evaluations of a scatter pattern, which
  is the right trade for reproducibility at this scale.
* Grouped hash grids (`groups=` argument) exist for multi-environment batching; potentially
  useful for parameter sweeps running several replicates in one grid. Unverified.

## 3. Dynamic creation and removal of cells

Chosen strategy: **prefix layout + deterministic append**. Implemented as planned in
`cells/lifecycle.py` and `kernels/cell_kernels.py` (Milestone 3); measured cost on CPU at
100k cells: decision kernel 0.55 ms, scan + host read 0.03 ms, placement with every cell
dividing 3.1 ms — under 1% of a cell step with 10 mechanics substeps
(`benchmarks/results/lifecycle_cpu.json`).

1. A lifecycle kernel writes `divide_flag[i] ∈ {0,1}` for every active cell (per-cell RNG,
   see §6 of AGENTS.md on reproducibility).
2. `warp.utils.array_scan(divide_flag, offsets, inclusive=False)` gives each dividing cell a
   unique daughter slot `count + offsets[i]`; the total is the last inclusive value.
3. A placement kernel writes the daughter into its slot (position = parent + jittered
   direction from the parent's RNG stream, radius, state).
4. `count += n_daughters`. Reading `n_daughters` is one host↔device sync per cell step.
5. Capacity overflow is checked on the host at that same sync; the run stops with a clear
   error rather than reallocating silently.

Daughter slot order depends only on parent index, never on thread scheduling, so the result
is reproducible across runs and devices (up to float summation order in the physics).

Dead cells are not removed in the MVP (AGENTS.md: they remain as physical entities), so no
compaction is needed. When removal arrives: mark `state == REMOVED`, `array_scan` over the keep
mask, scatter the survivors into a second buffer set, swap. That is one extra full copy per
cell step and only when something was removed.

Rejected: per-cell Python objects, free-slot lists (need a device-side allocator and break the
prefix invariant), `wp.array` reallocation per division.

## 4. Options for 3-D scalar diffusion

The oxygen field is quasi-steady on the cell time scale, so the real requirement is a fast
**steady-state solve** `D ∇²O − U(O) ρ = 0` per cell step, not time integration.

| Option | What Warp gives | Assessment |
|---|---|---|
| A. Explicit Jacobi/FTCS, hand-written 3-D stencil kernel | nothing specific; ~30 lines | Reference solver only. Stable for `dt ≤ dx²/(6D)`: with D = 2000 µm²/s and dx = 20 µm, dt ≤ 0.03 s → ~10⁴ substeps per 0.25 h cell step. Use to validate B/C, not in production. |
| B. Iterative steady-state: Jacobi or red-black Gauss–Seidel on the same stencil, iterate to tolerance | same kernel, plus `warp.utils.array_sum` for the residual | Simple, GPU-friendly, no dependencies. Convergence needs O(N_x²) sweeps; multigrid V-cycles would fix it later. Michaelis–Menten uptake handled by lagging U(O) one iteration (Picard). **Recommended first implementation.** |
| C. Implicit sparse solve | `warp.sparse` (BSR), `warp.optim.linear.cg/bicgstab/gmres` with preconditioners | Assemble the 7-point Laplacian once (geometry fixed), rebuild only the diagonal from cell density each step. Converges in tens of CG iterations. Best long-term option; also the natural place for an implicit adjoint if gradients are ever wanted. |
| D. `warp.fem` | `fem.Grid3D`, `Nanogrid`, examples `example_diffusion_3d.py`, `example_convection_diffusion.py` | Full FEM machinery; more than a regular-grid finite-difference problem needs. Worth revisiting for patient geometry (Milestone 7), where the domain is not a box. |

Cells ↔ grid coupling: trilinear scatter of consumption (`wp.atomic_add` on 8 nodes; order
non-deterministic but the sum is, up to float round-off) and trilinear gather for sampling.
Alternative without atomics: bin cells into voxels with `array_scan`, then reduce per voxel.

*Implemented (Milestone 4):* option B as red-black SOR with per-axis Dirichlet/Neumann
faces, warm-started every cell step; the scatter uses **int64 fixed-point** atomics, which
makes the deposit exactly order-independent. CPU cost (`benchmarks/results/field_cpu.json`):
one sweep 0.42 ms at 41³ nodes and 3.2 ms at 81³; a warm-started update on the spheroid
needs 10–60 sweeps, i.e. the field costs less than the mechanics substeps of the same cell
step. Cold solves need 70–400 sweeps at ω = 1.8 — multigrid (option B+) or CG (option C)
become relevant only for grids beyond ~100³ or for cold starts.

## 5. GPU memory estimates

| Item | Size |
|---|---|
| Cells, 10^6 × 56 B | 56 MB |
| HashGrid, 10^6 points (2 int per point) + 128³ buckets (2 int each) | 8 MB + 16 MB |
| Oxygen grid 200³ float32, 2 buffers (ping-pong or in/out) | 64 MB |
| Sparse Laplacian for 200³ (option C), 7 non-zeros/row, float32 + int32 index | ≈ 450 MB |
| Cell density grid 200³ | 32 MB |
| Tape for gradients (not MVP): every intermediate array per substep | ≫ GB without checkpointing |

MVP (10^5 cells, 100³ grid) fits in well under 200 MB on any CUDA device. 10^6 cells with a
sparse implicit solve on 200³ is the first configuration where memory needs attention.

## 6. Expected CPU ↔ GPU synchronization points

Per cell step (target: exactly one unavoidable sync):

| Point | Avoidable? |
|---|---|
| `n_daughters` after the division scan (needed to launch placement with the right `dim` and to bump `count`) | No. Could be deferred by launching over capacity with a device-side count, at the cost of idle threads. |
| Steady-state solver residual check | Yes: run a fixed iteration budget, check the residual every k iterations or once per cell step. |
| Metrics (`array_sum`, min/max) | Yes: accumulate on device, copy every N steps (AGENTS.md). |
| `HashGrid.build` | Warp allocates sort scratch on first build; `reserve(num_points)` up front avoids later reallocation. Build itself is asynchronous on CUDA (unverified). |
| Kernel launches | None; and the whole mechanics substep loop is CUDA-graph capturable (`wp.ScopedCapture`), as Warp's DEM example does, once `reserve` has been called. |

The mechanics substep loop (grid build + 2 kernels, ×N substeps) contains no host sync at all.

## 7. Relevant upstream Warp material

Bundled with the package (`warp/examples`, `warp/tests`):

* `core/example_dem.py` — HashGrid + contact forces + graph capture. The spike's kernel
  structure follows it.
* `optim/example_particle_repulsion.py` — pair potential via `wp.grad`, periodic box, HashGrid.
* `fem/example_diffusion.py`, `fem/example_diffusion_3d.py`, `fem/example_convection_diffusion.py`
  — steady and transient diffusion with `warp.fem` and `warp.sparse`.
* `tests/geometry/test_hash_grid.py` — brute-force validation pattern reused in
  `tests/test_neighbors.py`.
* `warp.utils.array_scan`, `radix_sort_pairs`, `runlength_encode`, `array_sum` — primitives
  for §3.
* `wp.render.UsdRenderer.render_points` — USD point export for Milestone 8 (needs `usd-core`).

## 8. Technical risks

1. **CPU-only development.** Warp's CPU backend is single-threaded; 100k cells cost 26 ms per
   mechanics substep here. Enough to develop and test, useless for performance claims. A CUDA
   machine is required from Milestone 6 (still TODO in AGENTS.md).
2. **Explicit mechanics stability.** `rate × dt ≤ 0.5` is enforced (`ContactParams.check_substep`);
   crowded cells pushed by many aligned neighbours contract faster than an isolated pair.
   Observed stable at `rate × dt = 0.05` with ~6 contacts. Needs re-checking when adhesion or
   larger stiffness is introduced.
3. **Time-scale gap.** Mechanics relaxation of a 10k-cell cluster is diffusive (hours of
   simulated time for full relaxation, see §10); with `dt_mechanics = 0.005 h` a 240 h
   experiment is 48k substeps. Fine on GPU, slow on CPU. `dt_mechanics` becomes a third
   configurable time step next to `dt_field` and `dt_cells`; AGENTS.md should list it.
4. **Diffusion solver cost.** Steady-state per cell step on 100³–200³ is the dominant field
   cost. Option B may need multigrid; option C needs a preconditioner. Decide with a benchmark
   in Milestone 4, not now.
5. **Float32 coordinate range.** Fine for spheroids (< 1 mm). Patient geometry in cm would
   push float32 resolution to ~10^-3 relative of the cell radius; switch to float64 arrays and
   `HashGrid(dtype=float64)` or use a local origin per tissue region.
6. **Determinism across devices.** Same-device runs are bitwise reproducible (tested on CPU).
   CPU vs CUDA differ by float summation order; `tests/test_gpu.py` asserts agreement within
   tolerance only.
7. **`warp.sim` is migrating to Newton.** Do not depend on it; the spike uses only core Warp.

## 9. Minimal architecture for Milestones 1–2

Implemented in this spike:

```text
src/warpbiocell/
    device.py                     resolve_device("cpu"|"cuda:0"|"auto"), synchronize
    cells/state.py                CellPopulation: SoA + capacity + active prefix views
    cells/initialization.py       spherical_cluster(n, radius, spacing_factor, jitter, seed)
    cells/mechanics.py            ContactParams (k, gamma, r_max, margin; stability rule),
                                  contact_substep, relax_contacts
    spatial/neighbors.py          NeighborGrid over the active prefix
    kernels/mechanics_kernels.py  contact_velocities (gather), integrate_positions
    reference/mechanics.py        O(N²) numpy reference of one substep
tests/                            state, neighbours vs brute force, analytical pair decay,
                                  centre-of-mass conservation, Warp vs reference, cluster
                                  relaxation, bitwise reproducibility, CUDA (skipped here)
benchmarks/bench_mechanics.py     per-kernel timings, JSON with device metadata
examples/spike_repulsion.py       the 10k-cell relaxation run
```

Model as implemented (units µm, h):

```text
overlap_ij = r_i + r_j − |x_i − x_j|
v_i        = (k/γ) Σ_j overlap_ij · (x_i − x_j)/|x_i − x_j|      for overlap_ij > 0
x_i       += v_i · dt
```

Positions are integrated in place. If gradients are ever wanted, switch to `position_in` →
`position_out` double buffering; nothing else in the design changes.

## 10. Benchmark plan and spike results

### Spike results (CPU, Apple arm64, Warp 1.17.0, single-threaded)

`benchmarks/results/mechanics_cpu.json`, 20 timed repetitions, median:

| cells | contacts/cell | grid build | contact kernel | integrate | substep | cells/s |
|---:|---:|---:|---:|---:|---:|---:|
| 1 000 | 5.2 | 0.11 ms | 0.11 ms | 0.005 ms | 0.35 ms | 2.9 × 10⁶ |
| 10 000 | 5.7 | 0.15 ms | 2.73 ms | 0.015 ms | 2.96 ms | 3.4 × 10⁶ |
| 100 000 | 5.8 | 0.68 ms | 24.1 ms | 0.11 ms | 26.0 ms | 3.8 × 10⁶ |

Linear scaling; the contact kernel is 93% of the substep. These are **CPU** numbers and say
nothing about GPU performance.

Physics check (`examples/spike_repulsion.py`, 10k cells, 10% initial compression,
k/γ = 10 h⁻¹, dt = 0.005 h): overlaps decay monotonically (mean 0.79 → 0.17 → 0.02 →
0.003 µm at 0, 2, 12, 24 h; max 1.6 → 0.065 µm), the cluster radius grows from 193 to 209 µm,
centre of mass is conserved, no cell is lost. Interior relaxation is diffusive, hence slow.

### Plan for the CUDA machine (Milestone 6)

1. Same script, `--device cuda:0`, sizes 10³–10⁶. Expect the contact kernel to drop below
   1 ms at 100k; report GPU memory via `wp.get_mem_usage` or `nvidia-smi`.
2. Add CUDA-graph capture of the substep loop and measure launch-overhead savings at small N.
3. Separate scaling of `HashGrid.build` vs query with candidates/cell as a parameter (vary
   `margin` and compression).
4. Once Milestones 3–4 exist: lifecycle kernel + scan, field solve, host↔device transfer,
   total cell step — each timed separately, as AGENTS.md requires.
5. Record every run's device, Warp version, commit hash and parameters in the JSON.

Upstream candidates to watch for while profiling: HashGrid build cost for dynamic point
counts, query performance when the query radius is much smaller than the bucket edge, and
scatter/gather patterns between particles and grids.
