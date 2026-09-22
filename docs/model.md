# The model: equations, discretisation, assumptions

One place for everything the simulator computes, as implemented on 2026-09-22 (Milestones
0–9). Code references point at the kernels that implement each equation. Units: µm, h, mmHg.

## 1. State

A cell `i` is a sphere with centre `x_i` [µm] and radius `r_i` [µm], a state
`s_i ∈ {PROLIFERATIVE, QUIESCENT, HYPOXIC, DEAD}`, a type (one phenotype in the MVP), an age
[h], the oxygen sampled at its centre `O_i` [mmHg], a persistent random-number stream, and
per-step scratch values (velocity, neighbour count). Storage is structure-of-arrays with
preallocated capacity `max_cells`; live cells occupy the prefix `[0, count)` and nothing is
ever removed in the MVP (`cells/state.py`).

The oxygen field `O(x)` [mmHg] lives on a regular cubic grid of spacing `dx` with per-axis
Dirichlet (fixed `O_b`) or Neumann (zero-flux) faces (`fields/scalar_field.py`).

## 2. Time stepping and operator splitting

Three time steps (`simulation/simulator.py`):

```text
dt_cells       lifecycle update                     0.25 h (default)
dt_mechanics   contact relaxation substep           0.025 h; rate·dt ≤ 0.5 enforced
(field)        quasi-steady: solved to convergence at every cell step, no time step
```

One cell step, in this order:

1. deposit living cells on the grid, solve the steady-state oxygen field (warm start from the
   previous solution), sample it at the cells;
2. lifecycle decisions for every cell (death, state, division), daughters appended;
3. `dt_cells / dt_mechanics` mechanics substeps, each rebuilding the neighbour grid;
4. metrics on demand.

Consequences: the lifecycle at step `n` sees the oxygen of the configuration at the start of
step `n` and the crowding left by the mechanics of step `n−1`; daughters inherit the parent's
sampled oxygen until the next field update.

## 3. Mechanics (`kernels/mechanics_kernels.py`)

Overdamped linear repulsion between overlapping spheres, no inertia, no adhesion:

```text
overlap_ij = r_i + r_j − |x_i − x_j|
F_ij       = k · overlap_ij · (x_i − x_j)/|x_i − x_j|      if overlap_ij > 0, else 0
dx_i/dt    = (1/γ) Σ_j F_ij
```

Explicit Euler with step `dt_mechanics`; only `rate = k/γ` [1/h] matters (default 10 /h: an
isolated overlapping pair relaxes as `exp(−2·rate·t)`). Neighbours come from a hash grid with
bucket edge equal to the query radius `2 r_max + margin`; the same pass counts
`neighbor_count_i` = cells within that radius (the crowding measure). Pairs closer than
1e-6 µm are skipped (undefined direction; never produced by initialisation or division).

Validation: neighbour counts against brute force, the analytical pair decay
`(1 − 2·rate·dt)^n`, centre-of-mass conservation, O(N²) numpy reference, relaxation of a
compressed cluster (`tests/test_mechanics.py`, `tests/test_neighbors.py`).

## 4. Lifecycle (`kernels/cell_kernels.py`)

Evaluated once per `dt_cells` for every non-dead cell, in this order, with all random draws
from the cell's own stream:

```text
death rate    = anoxic_death_rate          if O_i < death_threshold    (max with death_rate)
              = death_rate                 otherwise
P(die)        = 1 − exp(−death rate · dt)                              → DEAD, stop

crowded       = neighbor_count_i ≥ inhibition_threshold
state         = HYPOXIC        if O_i < hypoxia_threshold
              = QUIESCENT      else if crowded
              = PROLIFERATIVE  otherwise

oxygen_factor = 1                                            if O_i ≥ hypoxia_threshold
              = (O_i − death_threshold)/(hypoxia_threshold − death_threshold)   in between
              = 0                                            if O_i ≤ death_threshold
P(divide)     = 1 − exp(−division_rate · oxygen_factor · dt)          unless crowded
```

Division: the daughter takes slot `count + k` where `k` is the rank of the parent among the
dividing cells (prefix sum), inherits radius, type, state and sampled oxygen, gets age 0, and
the pair is placed symmetrically about the parent's centre, `placement_factor · r` apart, in a
random direction from the parent's stream. The overlap this creates is relaxed by the
mechanics within the same step.

Stated simplifications (all revisable, none hidden):

* memoryless cell cycle — no refractory period, no minimum age; a cell divides at most once
  per step and a daughter not in its birth step, so free growth per step is `(1 + p)` with
  `p = 1 − exp(−λ dt)`: an effective rate below `λ` by O(λ dt) (0.5 % at the defaults);
* no volume conservation — the daughter has the parent's radius, growth is implicit in the
  division rate, and the packing density inside the spheroid is set by the mechanics;
* linear ramp for `oxygen_factor` and a step for the death rate (illustrative forms);
* dead cells stay in place, keep their volume, do not consume oxygen and do not age;
* with both thresholds at 0 the rules ignore oxygen (Milestone 3 behaviour).

Validation: exact division counts, dead cells inert, contact inhibition, the discrete
branching law, death statistics, threshold behaviour with prescribed oxygen
(`tests/test_lifecycle.py`, `tests/test_lifecycle_oxygen.py`).

## 5. Oxygen field (`kernels/field_kernels.py`, `fields/diffusion.py`)

Reaction–diffusion with Michaelis–Menten uptake by living cells of number density `n(x)`
[cells/µm³]:

```text
∂O/∂t = D ∇²O − n(x) · q(O),        q(O) = q_max · O / (K + O)
```

Quasi-steady on the cell time scale (diffusion relaxes in seconds over the spheroid;
`dx²/(6D)` ≈ 0.03 s), so at every cell step the steady state `D ∇²O = n q(O)` is solved.

Discretisation: 7-point Laplacian on cubic voxels; Dirichlet faces fixed at `O_b`, Neumann
faces mirrored. Solver: red-black successive over-relaxation with the uptake linearised at
the current iterate,

```text
c      = n q_max / (K + O)
O_gs   = Σ_nb O / (6 + dx² c / D)
O_new  = max((1 − ω) O + ω O_gs, 0)
```

which preserves `O ≥ 0` (exactly for ω = 1; the clamp only acts on transients otherwise).
Warm start from the previous step; stop when the max-norm residual relative to the local
magnitude, `|Σ_nb O + dx²P g/D − (6 + dx²c/D) O| / max(6 O, O_b)`, falls below `tolerance`
(default 1e-5; the float32 floor is ≈ 1e-7·ω/(2−ω)). An explicit FTCS scheme with the guard `dt ≤ dx²/(6D)` exists as the
numerical reference only.

Cell ↔ grid transfer: trilinear weights. Deposit accumulates int64 fixed-point weights
(2⁻²⁴ resolution), so the density is exactly independent of thread order; sampling is
trilinear interpolation, clamped to the grid for cells outside it (which also do not deposit).

Validation: uniform field invariant, Gaussian pulse against the analytical solution, 1-D
`cosh` profile for linear uptake with Neumann side faces, symmetry, positivity, exact discrete
solution (dense solve + Picard) and numpy references (`tests/test_fields_*.py`).

## 5a. Metabolic species (`fields/species.py`, `fields/metabolism.py`)

Every species `C` obeys the same reaction–diffusion equation with per-state coefficients,
`n_k(x)` being the number density of cells in state `k` (deposited once per step, int64
fixed point):

```text
D ∇²C − Σ_k n_k q_k(C) + Σ_k n_k p_k · g(x) = 0        (quasi-steady)
q_k(C) = q_max[k] · C/(K + C)                           uptake, Michaelis–Menten
g(x)   = S/(K_S + S)  for a source species S, else 1     production scaled by the source's uptake factor
```

Oxygen is the first species (uniform uptake over living states unless `uptake_state_factors`
says otherwise); glucose and lactate follow in configuration order, so lactate can use the
glucose solution of the same step as its source. The same red-black SOR handles all of them:
the effective consuming density `Σ_k w_k n_k` (weights `q_max[k]/max q_max`) and the
production capacity `Σ_k n_k p_k` are combined on the grid before each solve. Convergence is
judged by the residual relative to the local magnitude, `|r| / max(6C, C_boundary)`.

Lifecycle coupling with glucose `G`: `P(divide) ∝ oxygen_factor(O) · glucose_factor(G)` with
the same linear ramp (0 at `glucose_death_threshold`, 1 at `glucose_threshold`), and with
`necrosis_requires_glucose` the anoxic death rate applies only when `O < death_threshold`
and `G < glucose_death_threshold` (MicroC's necrosis rule). HYPOXIC cells are glycolytic:
the phenotype is a rule until the gene network (Milestone 11).

Validation: per-state densities against the reference deposit, the glucose → lactate chain
against the exact discrete solution (dense solve with production and source), no lactate
without glycolytic cells or without glucose, the lifecycle rules with prescribed values
(`tests/test_species.py`).

## 5b. Tissue region (`geometry/`, `kernels/geometry_kernels.py`)

A region is a signed-distance field `φ(x)` [µm], negative inside, stored on the oxygen grid
with its node gradient (central differences). Synthetic shapes are evaluated analytically
(sphere and union exact; ellipsoid by the first-order `f/|∇f|` approximation); masks go
through two Euclidean distance transforms, `φ = d_outside − d_inside`, with the zero level
placed half a voxel from the boundary voxels, then nearest-voxel resampling.

Seeding: lattice points with `φ(x) < −(r + margin)` (whole cell inside), optionally also
inside a sub-volume; the number of cells is reported and capped by `max_cells`.

Confinement, overdamped like the cell–cell contact:

```text
p_i    = φ(x_i) + r_i                           penetration of the cell surface through the tissue surface
dx_i/dt += − wall_rate · p_i · ∇φ/|∇φ|          if p_i > 0;   wall_rate · dt_mechanics ≤ 0.5 enforced
```

Oxygen source: with `oxygen_source: tissue_surface` every node with `φ ≥ 0` is pinned to
`O_b`; the solver, the residual and the reference scheme all skip pinned nodes exactly like
Dirichlet faces (`tests/test_geometry.py` checks the pinned solution against the dense exact
solution).

## 6. Parameters and provenance

Defaults in `configs/tumor_spheroid.yaml`; labels follow AGENTS.md (illustrative / estimated
/ literature-derived / fitted). None is fitted yet.

| Parameter | Value | Label | Note |
|---|---|---|---|
| cell radius | 8 µm | illustrative | typical tumour cell 6–10 µm |
| `k/γ` | 10 /h | illustrative | pair relaxation time 3 min |
| query margin | 2 µm | illustrative | crowding counts neighbours within 18 µm |
| division rate λ | 0.0289 /h | illustrative | 24 h doubling |
| inhibition threshold | 8 neighbours | illustrative | cubic-lattice interior has 6, close packing 12 |
| anoxic death rate | 0.5 /h | illustrative | |
| hypoxia threshold | 8 mmHg (≈ 1 % O₂) | estimated | HIF activation range; MicroC uses 2.2 % = 15.7 mmHg |
| death threshold | 2 mmHg | illustrative | |
| D | 7.2e6 µm²/h (2000 µm²/s) | literature-derived | order of magnitude for O₂ in tissue; MicroC 1000 µm²/s |
| q_max | 1.4e8 mmHg µm³/h | estimated | 5e-17 mol/cell/s over α = 1.3 µM/mmHg; MicroC 3e-17 |
| K | 3.4 mmHg | estimated | MicroC 0.45 % O₂ = 3.2 mmHg |
| O_b | 38 mmHg (≈ 5 % O₂) | illustrative | 150 mmHg ≈ air-saturated medium; MicroC 7 % = 50 mmHg |

Conversions: 7.13 mmHg per % O₂ (37 °C, 1 atm, humidified); α = 1.3 µM/mmHg =
1.3e-21 mol µm⁻³ mmHg⁻¹. MicroC's full parameter files: `docs/reference/microc_parameters.md`.

## 7. Randomness and reproducibility

Every cell owns a counter-based stream `rand_init(seed, slot)` (`rng_state`), advanced only
by its own draws; daughters use the stream of their slot. Daughter slots are assigned by a
prefix sum. Same configuration + seed + device ⇒ bitwise identical results, independent of
`max_cells` (tested). CPU vs CUDA: float round-off in mechanics and field, statistical
agreement in the biology.

## 8. What the model does not contain

Lactate uptake (MCT1), ATP and pH; adhesion and motility; cell growth and volume changes;
removal of dead cells; multiple cell types; gene networks (Milestone 11); vasculature;
deformable tissue boundaries. Each is listed in docs/roadmap.md with its milestone.
