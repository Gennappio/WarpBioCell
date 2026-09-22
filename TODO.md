# Current task

Replace this file's content when the task is done.

## Done: architecture spike (2026-09-22)

`docs/architecture_spike.md`, `src/warpbiocell/` (cells, spatial, kernels, reference), 17 CPU
tests, `benchmarks/results/mechanics_cpu.json`, `examples/spike_repulsion.py`. Verified on CPU
only; CUDA tests exist (`tests/test_gpu.py`) but have not run yet.

## Open items that need a human decision

* Which CUDA machine provides GPU numbers (AGENTS.md, Device policy).
* Whether to run `tests/test_gpu.py` and `benchmarks/bench_mechanics.py --device cuda:0`
  before or after Milestone 3.

## Next: Milestone 3 — cell lifecycle

Following `docs/architecture_spike.md` §3 (prefix layout + deterministic append):

1. Add `cell_state` (DEAD/HYPOXIC/QUIESCENT/PROLIFERATIVE), `cell_type`, `age`, `id` and a
   per-cell RNG counter to `CellPopulation`. Keep the SoA layout.
2. Per-cell RNG with `wp.rand_init(seed, cell_id, step)`; no global random state.
3. Division kernel: `P = 1 - exp(-lambda * dt_cells)`; contact inhibition from `contact_count`
   (explicit, configurable threshold). Daughter slots via `warp.utils.array_scan`; daughter
   placed at parent + random unit vector * (r_parent * placement_factor), never coincident.
4. Death kernel: placeholder oxygen-independent rate until Milestone 4; dead cells stay.
5. Capacity check at the single host sync per cell step; clear error on overflow.
6. Tests: division produces exactly one daughter; dead cell cannot divide; population never
   exceeds capacity; fixed seed reproduces the run bitwise; growth curve without inhibition
   matches `N0 * exp(lambda t)` statistically.
7. Bench: lifecycle kernel + scan timing, appended to `bench_mechanics.py` or a new script.

Do not start oxygen (Milestone 4) before the lifecycle tests pass.
