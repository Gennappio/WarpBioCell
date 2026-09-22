# AGENTS.md

## Project

**Working name:** Cellular Digital Twin / WarpBioCell

A **GPU-accelerated agent-based biological tissue simulator** built on NVIDIA Warp.

The first scientific demonstrator is a tumor spheroid whose cells proliferate, die, interact mechanically, consume oxygen and respond to the local microenvironment.

The long-term goal is a dynamic biological layer for patient digital twins (imaging, spatial omics, OpenUSD, Isaac for Healthcare, scientific agents).

This is a scientific computing project. Do not turn it into a generic software framework prematurely.

Related documents:

* [docs/vision.md](docs/vision.md) — long-term architecture, future integrations, open questions;
* [docs/roadmap.md](docs/roadmap.md) — milestones, definition of MVP complete, experiment runner, sweeps;
* [docs/upstream.md](docs/upstream.md) — upstream contribution strategy and PR discipline;
* [TODO.md](TODO.md) — the current task.

Read `TODO.md` first. Read the other documents only when the task touches them.

---

## Scientific objective

> Can simple local cellular rules plus a diffusive oxygen field generate plausible tumor spheroid growth and necrotic-core formation?

Expected qualitative phenomena: initial proliferation, spatial competition, reduced proliferation under hypoxia, death under severe oxygen deprivation, a viable outer rim, a possible hypoxic or necrotic core.

Behavior must emerge from explicit rules. Do not hard-code outcomes (for example a necrotic core) and do not add hidden rules to force a result.

Calibrating documented parameters against data is legitimate, provided the parameter is labeled `fitted` and the fitting procedure is recorded.

---

## MVP scope

The MVP contains ONLY: cells as agents, 3D positions, radius, cell type/state, local neighbor detection, proliferation, death, simple contact mechanics, a 3D oxygen field, oxygen consumption, oxygen-dependent behavior, metrics, reproducible experiments, visualization/export.

Do NOT implement yet: glucose, lactate, ATP metabolism, MCT1, intracellular or gene regulatory networks, immune cells, drug pharmacokinetics, vascularization, mutations, spatial transcriptomics, LLM agents, Isaac integration, clinical prediction.

---

## Environment

```text
language:      Python >= 3.10 (developed on 3.12, virtualenv managed with uv in .venv/)
compute:       NVIDIA Warp (warp-lang >= 1.17)
supporting:    numpy, pytest; matplotlib and pandas when metrics/plots arrive
optional later: OpenUSD, MONAI, AnnData, Zarr
```

Avoid heavy dependencies. Justify every new one.

### Device policy

Warp has no CUDA backend on macOS. The development machine is a Mac, so local runs are **CPU-only**.

* All code must run with `device="cpu"`. Never hard-code `"cuda"`; take the device from configuration.
* All tests must pass on CPU. Tests that need CUDA are marked `gpu` and skipped when no CUDA device exists.
* GPU-vs-reference tests and all performance numbers come from a CUDA machine (TODO: specify which one).
* Never report a CPU timing as GPU performance. Every benchmark records the device.

### Commands

```bash
uv pip install --python .venv/bin/python -e ".[dev]"   # or: pip install -e ".[dev]"
.venv/bin/python -m pytest                             # CPU-safe suite; gpu-marked tests auto-skip
.venv/bin/python -m pytest -m gpu                      # CUDA-only tests
.venv/bin/python examples/spike_repulsion.py           # 10k cells + HashGrid + repulsion
.venv/bin/python examples/growth_contact_inhibition.py # growth by division under contact inhibition
.venv/bin/python benchmarks/bench_mechanics.py         # per-kernel mechanics timings -> benchmarks/results/
.venv/bin/python benchmarks/bench_lifecycle.py         # lifecycle and full cell-step timings
```

Not yet existing: `python -m warpbiocell.run --config configs/tumor_spheroid.yaml` (experiment runner, see docs/roadmap.md).

---

## Design principles

Separate BIOLOGY from COMPUTATION from VISUALIZATION from PATIENT DATA.

* A biological rule must not depend on Isaac, USD or rendering.
* A simulation kernel must not contain visualization logic.
* Patient-specific initialization must not be embedded in the generic simulator.

Priorities, in approximately this order: correctness, clarity, scientific interpretability, reproducibility, profiling, performance.

* Do not optimize a scientifically incorrect model.
* Do not build abstractions before at least two real use cases need them.
* No large inheritance hierarchies. Prefer explicit data flows.

---

## Architecture (approximate)

```text
src/warpbiocell/
    simulation/     simulator.py (cell_step, TimeStepping)  config.py (later)
    cells/          state.py  model.py  initialization.py  lifecycle.py  mechanics.py
    spatial/        neighbors.py
    fields/         scalar_field.py  diffusion.py  oxygen.py
    kernels/        cell_kernels.py  field_kernels.py  mechanics_kernels.py
    reference/      slow numpy versions of kernels, used only by tests
    metrics/        population.py  spatial.py
    io/             checkpoints.py  export.py
    visualization/  simple_3d.py
examples/tumor_spheroid/
tests/  benchmarks/  docs/  configs/
```

This may change if a better design emerges. Prefer clarity over abstraction.

---

## Model specification

### Cell representation

Structure-of-arrays on the device. No per-cell Python objects.

```text
id, position xyz, radius, cell_type, cell_state, age, oxygen_local
```

`cell_state` is a single enum and the only source of truth (no separate `alive` flag). States are mutually exclusive, evaluated in this precedence:

```text
DEAD            irreversible
HYPOXIC         alive, local oxygen below hypoxia threshold
QUIESCENT       alive, adequately oxygenated, contact-inhibited
PROLIFERATIVE   alive, adequately oxygenated, free to divide
```

### Population capacity

Preallocate `max_cells` and track `active_cell_count` (or an equivalent strategy). Never reallocate device arrays on division, and never use Python append-style semantics.

Dynamic population is an explicit architectural challenge. Investigate preallocated arrays, free-slot lists, active masks, compaction and prefix sums, and benchmark the alternatives before choosing.

Targets: 10^4 cells comfortably in the MVP; architecture able to reach 10^5–10^6 without a rewrite. Do not claim million-cell performance without benchmarking it.

### Neighbor detection

Use Warp `HashGrid`, not O(N²) comparison. Query radius ≈ cell diameter + interaction margin. Benchmark neighbor-query scaling independently.

### Mechanics

Overdamped (inertia-free) dynamics:

```text
overlap = r_i + r_j - distance(i,j)
F_ij    = k * overlap          if overlap > 0   (or another documented contact law)
dx_i/dt = (1/gamma) * sum_j F_ij
```

No velocities or momentum are stored. Purpose: avoid unrealistic overlap, allow spheroid expansion, generate local mechanical constraints. No full biomechanical solver in the MVP.

### Proliferation

Rate-based and stochastic:

```text
P(divide during dt) = 1 - exp(-lambda * dt)
lambda = base_proliferation_rate * oxygen_factor
```

Implemented simplifications (kernels/cell_kernels.py), each to be revisited explicitly if changed:

* a cell divides at most once per `dt_cells` and a daughter cannot divide in the step of its birth, so free growth per step is `(1 + p)` with `p = 1 - exp(-lambda dt)`, an effective rate below `lambda` by O(lambda dt) (0.5% at dt = 0.25 h, lambda = 0.0289/h);
* the cell cycle is memoryless: no refractory period, no minimum age;
* the daughter inherits the parent's radius (no volume conservation, no growth model); the pair is placed symmetrically about the parent's centre, `placement_factor * r` apart (default 1.0), and the resulting overlap is relaxed by the mechanics within the same step;
* daughter slots are allocated by a prefix sum over dividing cells, so results never depend on thread scheduling or on `max_cells`.

### Contact inhibition

An explicit, configurable local crowding criterion reduces or prevents division. Implemented as `neighbor_count >= inhibition_threshold`, where `neighbor_count` is the number of cells within the mechanics query radius (`2 r_max + margin`); crowded cells are `QUIESCENT`. The threshold is an illustrative parameter.

### Death

Oxygen-dependent: below the severe-hypoxia threshold, death probability increases. Until the oxygen field exists, a constant oxygen-independent `death_rate` (`P = 1 - exp(-death_rate dt)`, default 0) is the placeholder. Dead cells are not deleted immediately; they remain as physical entities, never divide and stop ageing. Removal, shrinkage and necrotic material are later work.

### Randomness

Every cell owns a counter-based stream `rand_init(seed, slot)` kept in `rng_state` and advanced only by its own draws. A daughter uses the stream of the slot it is born into. No global random state anywhere.

### Oxygen field

Concentration `O(x,y,z,t)` on a regular 3D grid:

```text
∂O/∂t = D ∇²O - U(O) * rho_cells
```

* Uptake `U(O)` must depend on concentration (Michaelis–Menten, or linear with a floor) so that oxygen can never become negative. Document the chosen form.
* Boundary conditions are explicit and configurable. Start with constant oxygen at the domain boundaries (Dirichlet).
* No vasculature.

### Time scales and field solver

Three processes, three time steps:

```text
dt_cells       biological update (state, division, death)      ~ 0.1–1 h
dt_mechanics   explicit contact relaxation substep             ~ 0.005 h; rate * dt_mechanics <= 0.5 enforced
dt_field       oxygen diffusion (explicit reference only)      ~ 0.01 s
```

Oxygen diffusion is orders of magnitude faster than cellular dynamics. Mechanics sits in between: several mechanics substeps per cell step, all without host synchronization.

An explicit diffusion scheme is stable only for:

```text
dt_field <= dx² / (6 D)
```

With realistic tissue values this is a fraction of a second, against a cell step of minutes to hours, which means tens of thousands of explicit substeps per cell step. Do not assume explicit sub-stepping is viable.

Preferred approach: treat oxygen as **quasi-steady-state** and solve it to convergence at each cell step, using an implicit or iterative solver (Jacobi, red-black Gauss–Seidel, multigrid, or implicit LOD). Keep an explicit solver as a reference for numerical validation. Check and document stability and convergence criteria.

### Simulation cycle

```text
1. update spatial index
2. map cells to field
3. compute oxygen consumption
4. solve oxygen field
5. sample oxygen at cell positions
6. update cell states
7. proliferation/death decisions
8. mechanical interactions
9. update positions
10. collect metrics
```

The order may change for numerical reasons. Document the chosen operator splitting.

---

## Configuration and units

* Scientific parameters live in explicit configuration (YAML), never scattered through code.
* One consistent convention: length in micrometers, time in hours or seconds (pick one), concentration in a documented unit.
* Every parameter states its unit. Never mix unit systems implicitly.

Example shape:

```yaml
simulation: { duration_hours: 240, dt_cell_hours: 0.25, device: cpu }
cells:      { initial_count: 500, max_cells: 100000, radius_um: 8, proliferation_rate: ... }
oxygen:     { diffusion_coefficient: ..., boundary_concentration: ..., consumption_rate: ...,
              hypoxia_threshold: ..., death_threshold: ... }
```

---

## Reproducibility

Every run records: configuration, random seed, simulation version, Warp version, device information, timestamp.

* No global random state: per-cell persistent streams (see "Randomness" above), so results do not depend on thread scheduling.
* Daughter-slot allocation is deterministic (prefix sum over dividing cells), not first-come-first-served between threads.
* Same configuration + seed + device gives bitwise identical results, independent of `max_cells` (tested). Across devices (CPU vs CUDA) expect agreement to float round-off in the mechanics and **statistical** agreement of the biology only, and test it as such.

---

## Outputs

Per-step metrics, at minimum:

```text
time, living_cells, dead_cells, proliferative_cells, hypoxic_cells,
mean_radius, spheroid_radius, oxygen_mean, oxygen_min
```

Useful spatial metrics: radial oxygen profile, radial cell-state distribution, necrotic fraction, packing density.

Metrics export to CSV or Parquet; checkpoints to an efficient binary format. Outputs must be structured and machine-readable. Avoid GPU → CPU → GPU transfers inside the timestep loop; copy metrics periodically rather than every step.

---

## Validation and tests

Three kinds of validation, with separate tests. Do not confuse them.

* **Numerical** — does the code solve the equations correctly? (Gaussian pulse diffusion against the analytical solution.)
* **Computational** — does Warp execution match a simple CPU/numpy reference on tiny problems? Reference code may be slow; its purpose is correctness.
* **Biological** — does the behavior resemble known biology? (For example viable-rim thickness and spheroid growth curves against published data.)

Required categories: unit, numerical, integration, reproducibility, GPU-vs-reference.

Examples: a single cell does not spontaneously disappear; division produces exactly one daughter; a dead cell cannot divide; neighbor query detects known neighbors; uniform oxygen stays uniform; diffusion preserves symmetry; the zero-consumption field converges correctly; a fixed seed reproduces the simulation.

Never rely only on visual inspection.

### Benchmarks

1k, 10k, 100k cells when hardware permits. Measure separately: neighbor search, mechanics, lifecycle, field solve, host↔device transfer, total timestep. Record runtime, device memory, cells/sec, steps/sec, device. Profile first; do not optimize blindly.

---

## Rules for coding agents

Before modifying unfamiliar code: inspect the repository structure, read the relevant documentation, identify conventions, locate tests, state the intended change, implement the smallest coherent solution, run tests, benchmark when performance-relevant, summarize what changed.

* Do not refactor unrelated code.
* Do not rename public interfaces without reason.
* Do not add dependencies without justification.

### When changing biological logic

Always answer:

```text
What assumption changed?
Why?
What equation/rule represents it?
What units are used?
What observable behavior should change?
How can we test it?
```

If any answer is unclear, do not silently invent biology. Mark the assumption explicitly.

### Parameters and sources

Prefer peer-reviewed literature, original model papers, official documentation and validated datasets. Keep references next to the configuration or documentation.

Label every biological parameter as one of:

```text
illustrative | estimated | literature-derived | fitted
```

Never present an arbitrary parameter as biologically established.

### Agent-generated experiments

An AI agent must never silently modify biological assumptions. Any agent-generated experiment preserves: config, parameter provenance, model version, seed, hypothesis, expected measurement. Agents propose experiments; they do not rewrite simulation laws invisibly.

### Visualization

Visualization exists to understand the simulation: 3D cell positions, state coloring, oxygen slices, population curves, radial profiles. Do not spend substantial time on UI or photorealism before scientific validation.

### Commits and upstream

Commits are coherent and descriptive (`feat: implement hash-grid neighbor search`, `test: validate oxygen diffusion against reference`), never `updates` or `fix stuff`.

Never open an upstream issue or PR without explicit human approval. See [docs/upstream.md](docs/upstream.md).
