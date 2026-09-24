# warpfvm validation

All numbers from the CPU of the development Mac (Apple M-series, Warp's CPU backend runs
kernels on one thread), warp-lang 1.17.0, FiPy 4.0.3 (MicroCpy runs 4.0.2), float64,
2026-09-24. Nothing has run on CUDA yet; `tests/test_fvm_gpu.py` and
`benchmarks/bench_fvm.py --device cuda:0` produce those numbers on a GPU machine.

The three kinds of validation are kept apart, as in WarpBioCell (AGENTS.md).

## 1. Computational: warpfvm computes what FiPy computes

| check | test | result |
|---|---|---|
| mesh: cell and face centres, volumes, face areas, cell distances, the eight face masks, numbering, translation | `test_fvm_mesh.py` (1-D, 2-D, 3-D, anisotropic, one-cell axes) | identical to FiPy |
| assembled matrix `L` and right-hand side `b`, entry by entry, before any solve | `test_fvm_assembly.py::test_assembly_matches_fipy`: 4 geometries x 7 equations x 3 boundary patterns = 84 cases | max difference <= 1e-12 of max entry |
| the 7 equations | steady with explicit sources; with uniform decay; with a per-cell sink; with sinks and "wrong-sign" implicit terms (FiPy lags them); transient with sink and source; transient with mixed-sign implicit terms and `TransientTerm(coeff=2)`; scaled terms | |
| the 3 boundary patterns | every face fixed; some faces fixed and the rest zero-flux; one value per face | |
| FiPy's under-relaxation (`sweep(underRelaxation=0.7)`) | `test_under_relaxation_matches_fipy` | <= 1e-12 |
| an independent loop-by-loop reference of FiPy's discretization | `test_assembly_matches_loop_reference` | <= 1e-12 |
| `sweep` residual and `justResidualVector` | `test_fvm_solvers.py` | FiPy's values to 1e-10 |

## 2. Numerical: the scheme solves the equations

| check | result |
|---|---|
| 1-D Dirichlet-Dirichlet diffusion: linear profile | exact (1e-12) |
| uniform boundary value, no source | uniform field (1e-14) |
| 1-D diffusion-decay, `cosh` solution, n = 20..160 | max errors 4.3e-3, 1.2e-3, 3.0e-4, 7.7e-5: orders 1.90, 1.95, 1.98 |
| 3-D manufactured `sin sin sin` solution, n = 8, 16, 32 | max errors 1.2e-2, 3.2e-3, 8.0e-4: orders 1.95, 1.99 |
| mirror symmetry of a symmetric 3-D problem | 1e-12 |
| transient, zero-flux boundaries: sum of `phi V` grows by exactly `sum(s V) dt` | 1e-11 relative, 5 steps |
| steady flux balance: production = outflow through the fixed faces | 1e-10 relative |
| 1-D heat equation, `exp(-D pi^2 t) sin(pi x)`, 400 backward-Euler steps | within 0.2 % |
| long transient reaches the steady solution | 1e-10 |

## 3. Solutions against FiPy on MicroC's problems

`test_fvm_vs_fipy.py` runs MicroC's `MultiSubstanceSimulator.update` line for line (solver
choice included: `LinearLUSolver(iterations=10, tolerance=1e-6)`, or GMRES for non-"fixed"
boundaries) with `lib = fipy` and `lib = warpfvm`. Four substances with MicroC's diffusion
coefficients and boundary values (oxygen 1e-9 m^2/s and 0.07 mM, glucose 6.7e-10 and 5 mM,
lactate 6.7e-10 and 1 mM, TGFA 5.18e-11, 0 mM and decay 1e-4 1/s plus a per-cell uptake),
cells in a central spheroid, SI units.

| run | largest difference over the four substances |
|---|---|
| steady, 15^3 over 750 um (MicroC's 3-D workflow) | 4.2e-12 mM (glucose, range 4.4e-2 mM) |
| steady, 12 x 10 x 8 over 600 um | 2.0e-12 mM |
| steady, 30^2 and 50^2 over 1500 um (MicroC's 2-D workflows) | 3.3e-13 mM |
| transient, 5 steps of 60 s, same geometries | 1.4e-11 mM |
| MicroC's "linear_gradient" boundaries, routed to GMRES with tolerance 1e-6 by MicroC | 2.1e-5 mM on a 0.95 mM range: both answers are 1e-6-residual approximations |
| MicroC's Picard coupling (Michaelis-Menten oxygen uptake, under-relaxation 0.7, stop at 1e-10) | 8 iterations in both, 2.0e-14 mM |

`examples/microc_drop_in.py` (4 substances, 2000 cells): 15^3 in 16.7 ms against 135 ms, 40^3
in 0.49 s against 30.6 s, largest difference 6e-10 of a field's range.

## Findings

* **FiPy's LU solver can skip the solve.** `LinearLUSolver` first compares the residual of the
  starting values with `tolerance * ||b||` and returns them untouched when it is smaller;
  `||b||` includes the boundary terms. In the 2-D test problems (MicroC's 30^2 and 50^2 grids,
  sources per unit area without MicroC's 2-D adjustment factors) FiPy with MicroC's
  `tolerance=1e-6` leaves glucose and lactate exactly at their initial values, while the exact
  discrete solution has a glucose dip of 1.5e-5 and 4.1e-5 mM. On the 15^3 3-D grid the solve
  runs normally (dip 0.044 mM). warpfvm reproduces the behaviour for parity; whether MicroC's
  real 2-D runs trigger it depends on their source magnitudes and has not been checked.
* **FiPy's `var.value` is the live array.** A copy taken with `np.asarray(var.value)` changes
  under a later solve. MicroC copies (`np.array`, `.copy()`) where it matters; warpfvm returns
  a copy.
* **MicroC's cost in 3-D is mostly the sparse LU.** FiPy's own conjugate gradient is 3x (15^3)
  to 110x (48^3) faster than its LU on the same problem.
* **Warp's `warp.optim.linear` solvers are unusably slow on the CPU.** Its tiled dot product
  costs about 1000x a native reduction (53 ms against 0.05 ms for 32^3 values), so one CG
  iteration on a 32^3 grid takes ~160 ms. warpfvm uses its own loop on native reductions. A
  candidate for an upstream report (docs/upstream.md in WarpBioCell), not filed.

## CPU benchmark

`benchmarks/bench_fvm.py --device cpu --sizes 15 32 48 64 96 --fipy-max 96`
(`benchmarks/results/fvm_cpu_float64.json`): MicroC's steady oxygen problem, 20 um voxels,
cold start from the boundary value, `LinearLUSolver` in both libraries, median of 2.

| n | cells | warpfvm | CG iterations | ms / iteration | warm re-solve | FiPy LU | FiPy PCG | max diff / dip |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 15 | 3 375 | 2.9 ms | 50 | 0.06 | 2.5 ms | 30 ms | 9.5 ms | 2e-14 |
| 32 | 32 768 | 32 ms | 90 | 0.35 | 28 ms | 1.9 s | 62 ms | 3e-12 |
| 48 | 110 592 | 171 ms | 140 | 1.2 | 149 ms | 27 s | 0.25 s | 7e-13 |
| 64 | 262 144 | 0.65 s | 190 | 3.4 | 0.56 s | - | 0.66 s | 2e-12 |
| 96 | 884 736 | 3.3 s | 280 | 11.7 | 2.8 s | - | 2.8 s | 1e-12 |

Iterations grow about as 3n (Jacobi-preconditioned CG on a Laplacian); a multigrid
preconditioner would make them nearly constant. The warm re-solve (sources changed by 1 %,
one Picard step) saves little because the LU name solves to 1e-12; `LinearPCGSolver` with the
tolerance the coupling needs is the cheaper choice inside Picard loops.
