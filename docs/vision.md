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

The preferred bridge toward Isaac visualization. `UsdGeomPointInstancer` is a candidate for large cell populations. Appearance may encode cell type, cell state or oxygen level. Photorealism is not a priority during the scientific MVP.

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

## Open questions

These are unresolved and should be answered before the corresponding milestone becomes primary work.

### The scale gap between spheroid and patient

With an 8 µm cell radius and random close packing (~0.64), tissue holds roughly 3×10^5 cells per mm³.

```text
500 µm spheroid        ≈ 2×10^4 cells     → MVP target range, one agent per cell is fine
10^6 cells             ≈ 3 mm³
1 cm³ imaged tumor     ≈ 3×10^8 cells     → beyond one-agent-per-cell on a single GPU
```

An MRI-visible tumor cannot be simulated cell by cell with the MVP architecture. Milestone 7 needs an explicit strategy: coarse-grained agents (one agent per cell cluster), a hybrid continuum/ABM model with agents only in regions of interest, a representative sub-volume, or multi-GPU. The choice changes what "patient-specific cellular twin" means scientifically.

### Positioning against existing simulators

PhysiCell, BioDynaMo, Chaste, CompuCell3D and FLAME GPU 2 already cover agent-based tissue simulation, some of them on GPU. The project needs a clear statement of what it offers that they do not. Candidates:

* native integration with the Warp / OpenUSD / Isaac ecosystem;
* differentiability — Warp kernels support automatic differentiation, which could enable gradient-based calibration of the continuous parts of the model (field solve, mechanics), although stochastic discrete events (division, death) do not differentiate directly;
* a like-for-like benchmark against PhysiCell or FLAME GPU 2 on the same spheroid experiment.

### Which data validates the spheroid

Pick the quantitative targets early: growth curve, viable-rim thickness, necrotic-core onset diameter, and the dataset or paper each one comes from.
