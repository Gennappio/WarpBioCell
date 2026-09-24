# warpfvm

FiPy-style finite volumes on uniform grids, run by [NVIDIA Warp](https://github.com/NVIDIA/warp)
on the CPU or on a CUDA GPU. It implements the part of [FiPy](https://www.ctcms.nist.gov/fipy/)
that MicroC uses, with FiPy's names, cell and face numbering and discretization, so that
MicroC's diffusion code runs by changing two imports:

```python
from fipy import Grid2D, Grid3D, CellVariable, DiffusionTerm, ImplicitSourceTerm, TransientTerm
from fipy.solvers.scipy import LinearGMRESSolver, LinearLUSolver
# becomes
from warpfvm import Grid2D, Grid3D, CellVariable, DiffusionTerm, ImplicitSourceTerm, TransientTerm
from warpfvm.solvers import LinearGMRESSolver, LinearLUSolver
```

For the same equation it assembles the same matrix and right-hand side as FiPy (to 1e-12
relative in 85 combinations of equation, geometry and boundaries) and returns the same
solution (to round-off). It is not a general FiPy
replacement: see "Supported" below.

## Install

```bash
pip install -e warpfvm                 # from this repository (warp-lang and numpy only)
pip install -e "warpfvm[dev]"          # plus pytest, FiPy and scipy for the parity tests
pip install "warpfvm @ git+https://github.com/Gennappio/WarpBioCell.git#subdirectory=warpfvm"
```

## Device and precision

A mesh carries its device and precision; everything built on it follows.

```python
mesh = Grid3D(dx=20e-6, dy=20e-6, dz=20e-6, nx=128, ny=128, nz=128, device="cuda:0", dtype="float64")
```

Without arguments: `WARPFVM_DEVICE` (else Warp's preferred device: the first GPU, or the CPU)
and `WARPFVM_PRECISION` (else `float64`, like FiPy). `float32` halves memory traffic and gives
about 1e-6 relative accuracy.

## Example

```python
import warpfvm as fv

mesh = fv.Grid3D(dx=50e-6, dy=50e-6, dz=50e-6, nx=15, ny=15, nz=15)       # MicroC's 3-D grid
oxygen = fv.CellVariable(mesh=mesh, value=0.07, name="Oxygen")              # mM
oxygen.constrain(0.07, mesh.exteriorFaces)                                  # fixed boundary
rates = fv.bin_rates(mesh, cell_positions_m, -3e-17)  # (n, 3) positions in m; mol/s per cell -> mM/s
source = fv.CellVariable(mesh=mesh, value=rates)

eq = fv.DiffusionTerm(coeff=1e-9) - fv.ImplicitSourceTerm(coeff=1e-4) == -source
solver = fv.LinearPCGSolver(tolerance=1e-8)
eq.solve(var=oxygen, solver=solver)
print(solver.convergence.iterations, solver.convergence.relative_residual, oxygen.value.min())

step = fv.TransientTerm() == fv.DiffusionTerm(coeff=1e-9) + source          # backward Euler
step.solve(var=oxygen, dt=60.0)
```

`examples/microc_drop_in.py` runs MicroC's `MultiSubstanceSimulator.update` unchanged with
both libraries and prints times and differences.

## Supported

| FiPy | warpfvm |
|---|---|
| `Grid1D`, `Grid2D`, `Grid3D` (uniform `dx`, `dy`, `dz`, `nx..`, `Lx..`) | same numbering of cells and faces, same `cellCenters`, `faceCenters`, `cellVolumes`, face masks (`facesLeft`, `facesRight`, `facesBottom`/`Down`, `facesTop`/`Up`, `facesFront`, `facesBack`, `exteriorFaces`, `interiorFaces`), `shape`, `numberOfCells`, `numberOfFaces`, translation `mesh + offset` |
| `CellVariable(mesh, value, name, hasOld)` | `value`, `setValue(value, where=)`, `old`, `updateOld`, `constrain(value, where=faces)` with a number or one value per face (later constraints win), `release`, `-var`, `number * var` |
| `DiffusionTerm(coeff=D)` | D a number |
| `ImplicitSourceTerm(coeff=S)` | S a number or one value per cell; FiPy's rule decides per cell whether it goes on the diagonal or is lagged to the right-hand side |
| `TransientTerm(coeff=c)` | c a number; backward Euler over `dt` |
| explicit sources | a number, a numpy or Warp array, a `CellVariable` |
| `+ - * / ==`, `solve`, `sweep` (FiPy's residual), `justResidualVector` | same |
| `LinearPCGSolver`, `LinearCGSolver`, `LinearBicgstabSolver`, `LinearGMRESSolver`, `LinearLUSolver`, `DefaultSolver` | see "Solvers" |
| criteria `default`/`RHS`, `unscaled`, `initial`, `legacy` | same meaning |

Not supported, with an explicit error: per-cell or tensor diffusion coefficients, convection
terms, coupled equations, higher-order diffusion, fixed-gradient constraints
(`var.faceGrad.constrain`; unconstrained boundary faces are zero-flux as in FiPy),
constraints on interior faces or cell values, non-uniform grids, physical units, vector
variables, FiPy's lazy variable algebra (`var + 1`), old-style `boundaryConditions=`.

## Solvers

Every system warpfvm assembles is symmetric, and positive definite as soon as a boundary value
is fixed or a sink or transient term is present, so every solver name runs Jacobi-preconditioned
conjugate gradient on the device. The names exist so FiPy code runs unchanged.

* The tolerance means what it means in FiPy >= 4: stop when `||L x - b|| <= tolerance * ||b||`
  (criterion "default"), with the true residual checked at the end.
* `LinearLUSolver` reproduces FiPy's direct solver: FiPy first compares the residual of the
  starting values with `tolerance` and returns them untouched when it is small enough, then
  solves exactly. warpfvm does the same check, then runs CG to near machine precision
  (relative residual 1e-12 in float64, 1e-6 in float32).
* A steady problem with no fixed value, sink or transient term is singular. Its right-hand
  side is projected onto the range, with a `SingularSystemWarning` if the sources do not
  balance, and the solution keeps the mean of the starting values.
* `solver.convergence` holds iterations, residual, threshold and whether it converged; a
  `ConvergenceWarning` is raised at the iteration limit.
* `eq.solve(var)` without a solver uses `LinearLUSolver`, FiPy's default with the scipy suite.

## Differences from FiPy worth knowing

* `var.value` returns a copy; FiPy returns its live internal array.
* FiPy's `LinearLUSolver` (and warpfvm's, which reproduces it) returns the starting values
  unchanged when their residual is below `tolerance * ||b||`. `||b||` includes the boundary
  terms, so a weak interior source can leave a field exactly unchanged: with MicroC's
  `tolerance=1e-6` this happens in the 2-D test problems of `tests/test_fvm_vs_fipy.py`
  (glucose dip of 1.5e-5 mM computed as 0). A `LinearPCGSolver` with a tolerance
  appropriate to the source strength, or a tighter LU tolerance, avoids it.
* A 1-cell-thick 2-D grid has unit thickness, as in FiPy: sources per "volume" are per area.

## How it works

* `mesh.py` numbers cells x-fastest and faces family by family exactly as FiPy does; face masks
  are numpy booleans. Boundary constraints are packed into compact per-side arrays.
* `system.py` + `kernels.py` assemble a 7-point stencil on the device: a diagonal, a
  right-hand side and one off-diagonal per axis (with a scalar D every interior face of an
  axis has the same transmissibility, so no matrix is stored). The system is multiplied by
  FiPy's diagonal sign and divided by an interior diagonal: positive definite, of order one,
  and safe from float32 underflow in SI units. `LinearSystem.to_scipy()` returns FiPy's
  `(L, b)` for inspection.
* `krylov.py` runs CG with alpha and beta kept on the device and native reductions
  (`wp.utils.array_inner`): one stencil product, two inner products and two fused updates per
  iteration, one host synchronization every `check_every` iterations. Optional CUDA-graph
  replay (`use_cuda_graph=True`) is implemented but not yet validated on hardware.
* Why not `warp.optim.linear.cg`: with warp-lang 1.17 on the CPU its tiled dot product costs
  about a thousand times a native reduction (53 ms against 0.05 ms for 32^3 values), which
  makes one CG iteration ~160 ms on a 32^3 grid.

## Validation and performance

[docs/validation.md](docs/validation.md): mesh, assembly and solution parity with FiPy,
analytic solutions and orders of convergence, conservation, CPU benchmarks. Summary on
MicroC's steady oxygen problem, CPU (one thread), float64:

| grid | warpfvm | FiPy LU (MicroC's solver) | FiPy PCG | max difference / oxygen dip |
|---:|---:|---:|---:|---:|
| 15^3 | 2.9 ms | 30 ms | 9.5 ms | 2e-14 |
| 32^3 | 32 ms | 1.9 s | 62 ms | 3e-12 |
| 48^3 | 0.17 s | 27 s | 0.25 s | 7e-13 |
| 96^3 | 3.3 s | not run | 2.8 s | 1e-12 |

On the CPU warpfvm matches FiPy's own CG; the large gain against MicroC today comes from not
using a sparse LU in 3-D. GPU numbers have not been measured yet.

## On a CUDA machine

```bash
pip install -e "warpfvm[dev]"
pytest warpfvm/tests                          # CPU suite plus the gpu-marked CUDA checks
pytest warpfvm/tests -m gpu -v                # CUDA vs CPU, graph vs eager, bitwise repeatability
python warpfvm/benchmarks/bench_fvm.py --device cuda:0 --sizes 15 32 64 128 192 256 --fipy-max 96
python warpfvm/benchmarks/bench_fvm.py --device cuda:0 --precision float32 --sizes 64 128 256 --fipy-max 0
python warpfvm/examples/microc_drop_in.py --n 100 --device cuda:0
```

The benchmark writes `benchmarks/results/fvm_cuda0_<precision>.json`.

## Tests

`pytest warpfvm/tests` (about 150 tests, a few seconds on a laptop CPU). FiPy parity tests skip
when FiPy is not installed; CUDA tests skip without a GPU.
