# Current task

Replace this file's content when the task is done.

## Done

* 2026-09-22 — architecture spike (Milestones 0–2): `docs/architecture_spike.md`, cells +
  HashGrid + overdamped repulsion, CPU benchmarks.
* 2026-09-22 — Milestone 3, cell lifecycle: per-cell RNG streams, stochastic division with
  prefix-sum slot allocation, placeholder death rate, contact inhibition from neighbour
  counts, `cell_step` operator splitting, population metrics, 31 CPU tests,
  `benchmarks/results/lifecycle_cpu.json`, `examples/growth_contact_inhibition.py`.

## Open items that need a human decision

* Which CUDA machine provides GPU numbers (AGENTS.md, Device policy). `tests/test_gpu.py`
  and both benchmarks accept `--device cuda:0` but have never run on CUDA.
* Oxygen unit convention for Milestone 4: mmHg (PhysiCell convention) or % O2 (MicroC
  convention, 3–9% at the boundary). Default proposal: mmHg, with the % conversion written
  next to the parameters.

## Next: Milestone 4 — oxygen

Following `docs/architecture_spike.md` §4, option B first (iterative steady state), with the
explicit scheme kept as the numerical reference:

1. `fields/scalar_field.py`: regular 3-D grid (origin, spacing `dx`, shape), float32 values,
   Dirichlet boundary value; trilinear sampling and deposit helpers in
   `kernels/field_kernels.py`.
2. Cell → grid coupling: deposit per-cell uptake capacity to the 8 surrounding nodes
   (`wp.atomic_add`; float sum order is not deterministic — document, and keep the
   residual tolerance well above round-off).
3. Steady-state solver: Jacobi (or red-black Gauss–Seidel) sweeps of
   `D ∇²O = rho * u_max * O / (K + O)` with the uptake linearised at the previous iterate
   (`O_new = sum_nb O / (6 + dx² c / D)`, `c = rho u_max / (K + O_old)`), which keeps O ≥ 0.
   Fixed iteration budget plus a residual check every k sweeps (one sync).
4. Explicit FTCS time stepping as reference only, with the `dt <= dx²/(6D)` guard.
5. Gather: sample O at cell positions into `oxygen_local`.
6. Lifecycle coupling: `HYPOXIC` if `oxygen_local < hypoxia_threshold` (precedence above
   contact inhibition); `lambda = base_rate * oxygen_factor(O)` with a documented, simple
   form (e.g. clamp((O - O_death)/(O_hyp - O_death), 0, 1)); death rate increases below
   `death_threshold`. Every parameter labelled illustrative/estimated/literature-derived.
7. Tests (numerical / computational / biological kept separate): uniform field stays
   uniform; Gaussian pulse vs analytical solution (explicit reference); symmetry
   preservation; zero consumption converges to the boundary value; 1-D linear-uptake steady
   state vs the cosh profile; trilinear sampling exact for linear fields; Warp vs numpy
   reference on a tiny grid; spheroid run develops a radial oxygen gradient with hypoxic
   core (biological, qualitative).
8. Bench: steady-state solve at 50³, 100³, 200³ nodes; deposit + sample at 10k–100k cells.

Do not start the tumor spheroid experiment runner (Milestone 5) before the field tests pass.
