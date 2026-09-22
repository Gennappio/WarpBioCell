# Vision

Long-term direction of WarpBioCell. Nothing in this document is an MVP requirement. The MVP scope is defined in [AGENTS.md](../AGENTS.md) and the milestones in [roadmap.md](roadmap.md).

## Core idea

Existing healthcare digital twins often represent anatomy, geometry, instruments, tissue mechanics, imaging and physical interactions.

This project adds a complementary layer:

> **biological entities that evolve over time.**

```text
Patient data
CT / MRI / segmentation / spatial omics
                    │
                    ▼
             Tissue geometry
                    │
                    ▼
          Cellular Digital Twin
              NVIDIA Warp
                    │
          ┌─────────┼─────────┐
          ▼         ▼         ▼
       cells      fields    geometry
          │         │         │
          └─────────┼─────────┘
                    ▼
             evolving tissue
                    │
                    ▼
                 OpenUSD
                    │
                    ▼
          Isaac for Healthcare
```

A future scientific-agent layer may sit above this:

```text
Scientific question → OpenCellComms → experimental design → Cellular Digital Twin
        → simulation results → analysis / validation → next experiment
```

## What success looks like

Success is NOT a large codebase, many abstractions, many features or a complex GUI.

Success is:

```text
simple scientific model
+ GPU-native execution
+ clear biological assumptions
+ quantitative validation
+ real performance data
+ compelling visual demonstration
+ one or more naturally discovered upstream contributions
```

The final project should demonstrate that agent-based biological simulation can become a first-class dynamic component of patient digital twins.

## Future integrations

### Isaac for Healthcare

Isaac is NOT part of the core MVP. Once the simulator works independently, create an adapter, conceptually a `CellularStateProvider` exposing positions, radii, cell types, cell states, scalar fields and simulation time.

```text
Isaac for Healthcare
        │
        ▼
CellularStateProvider
        │
   ┌────┼─────┐
   ▼    ▼     ▼
Warp  MicroC PhysiCell
```

The Warp implementation becomes one provider. Isaac must not depend on internal simulator details.

### OpenUSD

The preferred bridge toward Isaac visualization. Implemented (Milestone 8, 2026-09-22): `io/export.py` writes a `UsdGeomPointInstancer` with one time sample per checkpoint, colour by state or oxygen, the tissue surface as points and the grid box as a guide; micrometre units, one time code per hour. A `CellularStateProvider` for Isaac would read the same checkpoints or the live arrays; it is not built until a machine with Isaac for Healthcare is available. Photorealism is not a priority during the scientific MVP.

### Patient-specific geometry

After the generic spheroid works, support initialization inside arbitrary tissue geometry, for example from a NIfTI segmentation mask:

```text
CT/MRI → segmentation → 3D tumor region → cell seeding → ABM
```

MONAI may later provide segmentation. The simulator must not require MONAI. No clinical claims.

### Spatial omics

```text
AnnData / H5AD / Zarr → obsm["spatial"] → initial cell positions + phenotypes / expression
        → patient-specific cellular twin
```

Do not implement before MVP completion.

### MicroC

MicroC is a scientific reference and a source of biological modeling experience. Do NOT blindly port it.

Use it to understand cell lifecycle, metabolism, diffusion, cell-cell interactions and experimental outputs, then redesign the appropriate components for GPU execution, modularity, patient digital twins and large-scale experiments.

Where useful, reproduce MicroC experiments to validate the new implementation.

### Metabolic model

After oxygen-only validation, candidates are glucose, lactate, ATP, OXPHOS, glycolysis, MCT1 and metabolic symbiosis. They enter only after the simpler system is numerically and biologically understood. Do not add metabolic complexity merely because MicroC contains it.

### Agentic layer (OpenCellComms)

A future orchestration layer, not part of the simulator kernel:

```text
scientific question → hypothesis → experiment design → simulation matrix
        → WarpBioCell runs → analysis → comparison → next experiment
```

The simulator must therefore expose a clean programmatic API and structured machine-readable outputs:

```python
experiment = Experiment(config)
result = experiment.run()
metrics = result.metrics
```

Constraints on agent-generated experiments are in [AGENTS.md](../AGENTS.md).

### Mechanobiology

A later high-value extension is two-way coupling between cell biology (proliferation, death, migration) and tissue mechanics (stress, deformation, pressure). Potential tools: Warp, Newton, Isaac physics. Examples: pressure-dependent proliferation, mechanically constrained growth, tumor-induced tissue deformation.

Do not implement until the uncoupled ABM is validated.

## Potential publication / demo

> **GPU-accelerated patient-specific cellular digital twins coupling agent-based tumor biology with physical simulation.**

```text
MRI → segmented tumor → initial cellular state → GPU ABM
    → oxygen / hypoxia / necrosis → predicted evolution → Isaac visualization
```

A useful north star, not an MVP requirement.

### Gene network per cell (Milestone 11, design notes — implemented 2026-09-22, see model.md §4b)

Decided 2026-09-22: a Boolean gene regulatory network inside every agent is worth building
— it is what makes MicroC's gene-perturbation experiments (knockouts, overexpression, p53,
MCT1) reproducible on the GPU, and it fits the architecture: ~70 nodes are 128 bits per
cell, updates are embarrassingly parallel, and the per-cell RNG streams keep runs
reproducible. It is scheduled last so that imaging and visualization come first.

Design to follow when the time comes:

* input format BoolNet `.bnet` (`A, B & !C`), optionally MaBoSS `.bnd/.cfg` rates; the MicroC
  network is exported from GINsim in one of these formats (no `.zginml` parser);
* rules compiled to a postfix program in integer arrays, evaluated by one interpreter kernel
  with a boolean stack held in a `uint64`; code generation only if profiling asks for it;
* three update semantics on the same core: synchronous (tests), asynchronous random
  (MicroC), continuous-time Markov with `rate_up`/`rate_down` (MaBoSS; primary);
* environment → input nodes clamped every cell step (`Oxygen_supply = O > threshold`,
  `Glucose_supply`, ...; the thresholds are MicroC's `input-parameters.txt`); output nodes
  `Proliferation / Apoptosis / Growth_Arrest / Necrosis` read by the lifecycle, which keeps the
  stochastic division rate and the space constraint;
* the built-in rules remain the `rules` phenotype model; the network is the `network` model,
  chosen in the configuration — the two use cases that justify the abstraction;
* validation: toy networks with known attractors (exact, synchronous), a single MaBoSS node
  against the analytical two-state Markov chain (statistical), a small network against MaBoSS
  itself when its Python package is available, then the MicroC network with oxygen inputs only.

What was built follows this design, with two differences: the MicroC network came as MaBoSS
`.bnd/.cfg` from the OpenCellComms `MicroC` adapter (unit rates, so its `maboss` mode and the
asynchronous mode are the same process); and the comparison against the MaBoSS binary is
still open — the closed-form checks (single-node kinetics, flip probabilities, agreement of
the two modes on the 106-node network) stand in. Measured on the CPU: the 106-node network
at 200 asynchronous picks per step costs ~10 µs per cell per step (2000 cells: 22 ms), the
Gillespie mode ~80 µs per cell for 5 time units; both are per-cell loops that map directly
onto GPU threads.

Related prior art: PhysiBoSS (PhysiCell + MaBoSS, CPU); a GPU MaBoSS for networks alone has
been published (to verify); a per-agent network inside a spatial GPU ABM with coupled fields
appears to be new.

## Open questions

These are unresolved and should be answered before the corresponding milestone becomes primary work.

### The scale gap between spheroid and patient

Measured, not estimated (Milestone 7, 2026-09-22): the mechanics settles the packing at
0.94 cells per (16 µm)³, i.e. 2.3×10⁵ cells per mm³ for 8 µm cells.

```text
500 µm spheroid        ≈ 1.5×10^4 cells     one agent per cell; runs on the CPU in minutes
10^5 cells             ≈ 0.44 mm³            a lesion of ~0.9 mm diameter
10^6 cells             ≈ 4.4 mm³             a lesion of ~2 mm diameter (the CUDA target)
1 cm³ imaged tumour    ≈ 2.3×10^8 cells      beyond one agent per cell on a single GPU
```

`geometry/seeding.py` makes this explicit: filling a region above `cells.max_cells` raises
`CellBudgetError` with the count, and `geometry.seed_within_radius_um` seeds a sub-volume.
So today a segmented lesion up to ~2 mm can be simulated cell by cell (once the GPU numbers
confirm the 10⁶ budget), and anything larger needs a strategy that is still open:
coarse-grained agents (one agent per cell cluster), a hybrid continuum/ABM model with agents
only in a region of interest, a representative sub-volume with periodic or mirror faces
(Neumann faces exist), or multi-GPU. The choice changes what "patient-specific cellular
twin" means scientifically and should be made with data, not before.

### Positioning against existing simulators

PhysiCell, BioDynaMo, Chaste, CompuCell3D and FLAME GPU 2 already cover agent-based tissue simulation, some of them on GPU. The project needs a clear statement of what it offers that they do not. Candidates:

* native integration with the Warp / OpenUSD / Isaac ecosystem;
* differentiability — Warp kernels support automatic differentiation, which could enable gradient-based calibration of the continuous parts of the model (field solve, mechanics), although stochastic discrete events (division, death) do not differentiate directly;
* a like-for-like benchmark against PhysiCell or FLAME GPU 2 on the same spheroid experiment.

### Which data validates the spheroid

Pick the quantitative targets early: growth curve, viable-rim thickness, necrotic-core onset diameter, and the dataset or paper each one comes from.
