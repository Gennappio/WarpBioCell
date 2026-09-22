# AGENTS.md

## Project

**Working name:** Cellular Digital Twin / WarpCell

This project develops a **GPU-accelerated agent-based biological tissue simulator** built on NVIDIA Warp.

The first scientific demonstrator is a tumor spheroid whose cells proliferate, die, interact mechanically, consume oxygen and respond to the local microenvironment.

The long-term goal is a **dynamic biological layer for patient digital twins**, capable of connecting:

* patient medical imaging;
* cell-level agent-based simulation;
* spatial and molecular data;
* GPU physics;
* OpenUSD;
* NVIDIA Isaac for Healthcare;
* scientific agents such as OpenCellComms.

This is a scientific computing project.

Do not turn it into a generic software framework prematurely.

---

# 1. Core idea

Existing healthcare digital twins often represent:

* anatomy;
* geometry;
* instruments;
* tissue mechanics;
* imaging;
* physical interactions.

This project adds a complementary layer:

> **biological entities that evolve over time.**

The conceptual architecture is:

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
Scientific question
       │
       ▼
OpenCellComms
       │
       ▼
experimental design
       │
       ▼
Cellular Digital Twin
       │
       ▼
simulation results
       │
       ▼
analysis / validation
       │
       ▼
next experiment
```

---

# 2. Scientific objective

The first scientific question is deliberately simple:

> Can simple local cellular rules plus a diffusive oxygen field generate plausible tumor spheroid growth and necrotic-core formation?

The simulator should reproduce qualitatively expected phenomena such as:

* initial proliferation;
* spatial competition;
* reduced proliferation under hypoxia;
* cell death under severe oxygen deprivation;
* emergence of a viable outer rim;
* possible formation of a hypoxic or necrotic core.

Do not tune the model to reproduce a predetermined result.

Simulation behavior must emerge from explicit rules.

---

# 3. Scope of MVP

The first version MUST contain only:

1. cells represented as agents;
2. 3D positions;
3. cell radius;
4. cell type/state;
5. local cell-cell neighborhood detection;
6. proliferation;
7. death;
8. simple mechanical/contact interaction;
9. a 3D oxygen field;
10. oxygen consumption;
11. oxygen-dependent cellular behavior;
12. simulation metrics;
13. reproducible experiments;
14. visualization/export.

Do NOT initially implement:

* glucose;
* lactate;
* ATP metabolism;
* MCT1;
* intracellular networks;
* gene regulatory networks;
* immune cells;
* drug pharmacokinetics;
* vascularization;
* mutations;
* spatial transcriptomics;
* LLM agents;
* Isaac integration;
* clinical prediction.

These belong to later milestones.

---

# 4. Technology

Primary language:

```text
Python
```

Core compute framework:

```text
NVIDIA Warp
```

Expected dependencies should remain minimal.

Likely supporting libraries:

```text
numpy
warp-lang
pytest
matplotlib
pandas
```

Optional later:

```text
OpenUSD
MONAI
AnnData
Zarr
```

Avoid introducing heavy dependencies unless necessary.

---

# 5. Design principle

The architecture must separate:

```text
BIOLOGY
from
COMPUTATION
from
VISUALIZATION
from
PATIENT DATA
```

A biological rule must not depend directly on Isaac, USD or rendering.

A simulation kernel must not contain visualization logic.

Patient-specific initialization must not be embedded into the generic simulator.

---

# 6. Proposed architecture

Use approximately:

```text
src/
    warpcell/

        simulation/
            simulator.py
            state.py
            config.py

        cells/
            model.py
            lifecycle.py
            mechanics.py

        spatial/
            neighbors.py

        fields/
            scalar_field.py
            diffusion.py
            oxygen.py

        kernels/
            cell_kernels.py
            field_kernels.py
            mechanics_kernels.py

        metrics/
            population.py
            spatial.py

        io/
            checkpoints.py
            export.py

        visualization/
            simple_3d.py

examples/
    tumor_spheroid/

tests/

benchmarks/

docs/
```

This structure may change if repository exploration reveals a better design.

Prefer clarity over abstraction.

---

# 7. Cell representation

Cells should initially use a **structure-of-arrays** representation suitable for GPU computation.

Conceptually each cell contains:

```text
id
position xyz
radius
cell_type
cell_state
age
oxygen_local
alive
```

Possible states:

```text
PROLIFERATIVE
QUIESCENT
HYPOXIC
DEAD
```

Do NOT represent every cell as a Python object if that prevents efficient GPU execution.

The biological abstraction may expose cell-like concepts while storage remains GPU-oriented.

---

# 8. Population capacity

Avoid reallocating GPU arrays every time a cell divides.

Prefer a preallocated capacity:

```text
max_cells
```

with:

```text
active_cell_count
```

or an equivalent efficient strategy.

The MVP should support at least:

```text
10^4 cells comfortably
```

and be architecturally capable of scaling toward:

```text
10^5–10^6 cells
```

without requiring a complete rewrite.

Do not claim million-cell performance without benchmarking it.

---

# 9. Neighbor detection

Cell-cell interactions require local neighborhood queries.

Prefer NVIDIA Warp spatial structures such as:

```text
HashGrid
```

rather than O(N²) pairwise comparisons.

Neighborhood radius should normally be related to:

```text
cell diameter + interaction margin
```

Benchmark neighborhood-query scaling independently.

This component is strategically important because it may expose useful upstream Warp improvements.

---

# 10. Mechanical model

The initial mechanics should remain simple.

For overlapping cells:

```text
overlap = r_i + r_j - distance(i,j)
```

If:

```text
overlap > 0
```

apply a repulsive displacement or force.

A minimal model may use:

```text
F_repulsive = k * overlap
```

or another documented simple contact law.

Avoid building a full biomechanical solver in the MVP.

Primary purpose:

* avoid unrealistic overlap;
* allow spheroid expansion;
* generate local mechanical constraints.

---

# 11. Cell proliferation

Each living cell has a proliferation rule.

Prefer a rate-based stochastic model.

Example:

```text
P(divide during dt) = 1 - exp(-lambda * dt)
```

where:

```text
lambda
```

depends on biological conditions.

Initially:

```text
lambda = base_proliferation_rate * oxygen_factor
```

Division should create a daughter cell near the parent.

Ensure that:

* daughter placement does not create extreme overlap;
* total state is updated consistently;
* random behavior is reproducible with a seed.

---

# 12. Contact inhibition

A cell should not divide indefinitely in highly crowded regions.

Implement a simple local crowding criterion such as:

```text
number_of_neighbors
```

or:

```text
local packing / overlap
```

which reduces or prevents division.

Keep the rule explicit and configurable.

---

# 13. Cell death

The MVP should support oxygen-dependent death.

Conceptually:

```text
if oxygen < severe_hypoxia_threshold:
    death_probability increases
```

Do not immediately delete dead cells.

Dead cells may remain temporarily as physical entities.

Later versions may support:

```text
cell removal
cell shrinkage
necrotic material
```

---

# 14. Oxygen field

This is the first continuous environmental field.

Represent oxygen concentration:

```text
O(x,y,z,t)
```

on a regular 3D grid.

Initial model:

```text
∂O/∂t = D ∇²O - consumption
```

where:

```text
D = diffusion coefficient
```

and cells act as local sinks.

Boundary conditions must be explicit.

Start with a simple configurable boundary condition such as:

```text
constant oxygen at domain boundaries
```

Avoid adding vasculature initially.

---

# 15. Coupling cells ↔ field

Each simulation cycle should conceptually perform:

```text
1. update spatial index
2. map cells to field
3. compute oxygen consumption
4. solve oxygen diffusion
5. sample oxygen at cell positions
6. update cell states
7. proliferation/death decisions
8. mechanical interactions
9. update positions
10. collect metrics
```

Exact order may be modified if numerical considerations require it.

Document the chosen operator splitting.

---

# 16. Time scales

Cellular dynamics and field dynamics occur at very different time scales.

Do NOT assume they must use the same timestep.

Support conceptually:

```text
dt_field
dt_cells
```

with multiple field updates per biological update if necessary.

Document assumptions.

Numerical stability must be checked for explicit diffusion schemes.

---

# 17. Baseline experiment

The first required experiment is:

## Tumor spheroid growth

Initial condition:

```text
small approximately spherical cluster
```

Environment:

```text
3D domain
constant oxygen boundary
no glucose model
no vasculature
```

Cells:

```text
single tumor-cell phenotype
```

Expected qualitative evolution:

```text
small spheroid
    ↓
growth
    ↓
oxygen gradient
    ↓
inner hypoxia
    ↓
reduced proliferation
    ↓
possible necrotic center
```

Do not hard-code a necrotic core.

It must emerge from the model.

---

# 18. Outputs

Every simulation should generate structured outputs.

At minimum:

```text
time
living_cells
dead_cells
proliferative_cells
hypoxic_cells
mean_radius
spheroid_radius
oxygen_mean
oxygen_min
```

Useful spatial metrics may include:

```text
radial oxygen profile
radial cell-state distribution
necrotic fraction
packing density
```

Results must be exportable to:

```text
CSV or Parquet
```

and checkpoints to an efficient binary format.

---

# 19. Reproducibility

Every experiment must record:

```text
configuration
random seed
simulation version
Warp version
GPU/device information
timestamp
```

Identical configuration + seed should reproduce results as far as reasonably possible.

Do not silently use global random state.

---

# 20. Configuration

Scientific parameters belong in explicit configuration.

Example:

```yaml
simulation:
  duration_hours: 240
  dt_cell_hours: 0.25

cells:
  initial_count: 500
  max_cells: 100000
  radius_um: 8
  proliferation_rate: ...

oxygen:
  diffusion_coefficient: ...
  boundary_concentration: ...
  consumption_rate: ...
  hypoxia_threshold: ...
  death_threshold: ...
```

Do not scatter scientific constants throughout code.

Units must be documented.

---

# 21. Units

Use one consistent physical-unit convention.

Recommended biological units:

```text
length: micrometers
time: hours or seconds
concentration: documented model-specific unit
```

Every parameter must clearly state its unit.

Never mix unit systems implicitly.

---

# 22. Validation philosophy

There are three different types of validation.

Do not confuse them.

### Numerical validation

Does the implementation solve the equations correctly?

Example:

```text
diffusion of a Gaussian pulse
```

should match expected analytical/numerical behavior.

### Computational validation

Does GPU execution produce results consistent with a simple CPU reference implementation?

### Biological validation

Does the resulting behavior resemble known biological behavior?

These must have separate tests.

---

# 23. Reference implementations

For important kernels, maintain small CPU/reference versions where useful.

Example:

```text
CPU diffusion reference
GPU Warp diffusion
```

Compare them on tiny problems.

Reference code may be slow.

Its purpose is correctness.

---

# 24. Tests

Required test categories:

```text
unit tests
numerical tests
integration tests
reproducibility tests
GPU vs reference tests
```

Examples:

```text
single cell does not spontaneously disappear
division produces exactly one daughter
dead cell cannot divide
neighbor query detects known neighbors
uniform oxygen remains uniform
diffusion preserves symmetry
zero-consumption field converges correctly
fixed seed reproduces simulation
```

Never rely only on visual inspection.

---

# 25. Benchmarks

Benchmark at least:

```text
1k cells
10k cells
100k cells
```

when hardware permits.

Measure separately:

```text
neighbor search
mechanics
cell lifecycle
field diffusion
host ↔ device transfer
total timestep
```

Record:

```text
runtime
GPU memory
cells/sec
steps/sec
```

Do not optimize blindly.

Profile first.

---

# 26. Strategic Warp contribution rule

One purpose of this project is to identify genuine opportunities for upstream contribution to NVIDIA Warp.

Never invent an artificial Warp feature request merely to create a PR.

Instead:

```text
build real scientific workload
        ↓
profile it
        ↓
find recurring limitation
        ↓
create minimal reproduction
        ↓
determine whether limitation is generic
        ↓
consider upstream Warp issue/PR
```

Potential areas include:

```text
HashGrid queries
particle interaction patterns
dynamic particle populations
field sampling
scatter operations
USD export
performance bottlenecks
```

Any upstream contribution must be broadly useful beyond this project.

---

# 27. Isaac for Healthcare integration

Isaac is NOT part of the core MVP.

Once the biological simulator works independently, create an adapter.

Conceptually:

```text
CellularStateProvider
```

exposing:

```text
positions
radii
cell types
cell states
scalar fields
simulation time
```

Possible architecture:

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

Our Warp implementation becomes one provider.

Do not make Isaac depend on internal simulator details.

---

# 28. OpenUSD

OpenUSD should be the preferred bridge toward Isaac visualization.

Possible representation:

```text
UsdGeomPointInstancer
```

for large cell populations.

Cell appearance may encode:

```text
cell type
cell state
oxygen level
```

Examples:

```text
proliferative
hypoxic
dead
```

Do not prioritize photorealistic visualization during the scientific MVP.

---

# 29. Patient-specific extension

After generic spheroid simulation works, add support for initialization inside arbitrary tissue geometry.

Potential input:

```text
NIfTI segmentation mask
```

Pipeline:

```text
CT/MRI
  ↓
segmentation
  ↓
3D tumor region
  ↓
cell seeding
  ↓
ABM
```

MONAI may later provide segmentation.

The simulator must not require MONAI.

---

# 30. Spatial omics extension

Future extension:

```text
AnnData / H5AD / Zarr
             │
             ▼
        obsm["spatial"]
             │
             ▼
initialize cell positions
             +
phenotypes / expression
             │
             ▼
patient-specific cellular twin
```

This connects naturally to single-cell and spatial transcriptomics.

Do not implement before MVP completion.

---

# 31. MicroC relationship

MicroC is a scientific reference and source of biological modeling experience.

Do NOT blindly port MicroC.

Use it to understand:

```text
cell lifecycle
metabolism
diffusion
cell-cell interactions
experimental outputs
```

Then redesign appropriate components for:

```text
GPU execution
modularity
patient digital twins
large-scale experiments
```

Where useful, reproduce MicroC experiments to validate the new implementation.

---

# 32. Future metabolic model

After oxygen-only validation, possible additions are:

```text
glucose
lactate
ATP
OXPHOS
glycolysis
MCT1
metabolic symbiosis
```

These should enter only after the simpler system is numerically and biologically understood.

Avoid introducing metabolic complexity merely because MicroC contains it.

---

# 33. Agentic layer

OpenCellComms is a future orchestration layer, not part of the simulator kernel.

Eventually:

```text
scientific question
      ↓
hypothesis
      ↓
experiment design
      ↓
simulation matrix
      ↓
WarpCell runs
      ↓
analysis
      ↓
comparison
      ↓
next experiment
```

The simulator must therefore expose a clean programmatic API.

Example future API:

```python
experiment = Experiment(config)
result = experiment.run()
metrics = result.metrics
```

Structured machine-readable outputs are essential.

---

# 34. Scientific agent constraints

An AI agent must never silently modify biological assumptions.

Any agent-generated experiment must preserve:

```text
config
parameter provenance
model version
seed
hypothesis
expected measurement
```

The agent should propose experiments, not rewrite simulation laws invisibly.

---

# 35. Development philosophy

Prioritize:

```text
correctness
clarity
scientific interpretability
reproducibility
profiling
performance
```

in approximately that order.

Do not optimize a scientifically incorrect model.

Do not build abstractions before at least two real use cases need them.

Do not create large inheritance hierarchies.

Prefer explicit data flows.

---

# 36. Coding-agent behavior

Before modifying unfamiliar code:

1. inspect repository structure;
2. read relevant documentation;
3. identify existing conventions;
4. locate tests;
5. state the intended change;
6. implement the smallest coherent solution;
7. run tests;
8. benchmark when performance-relevant;
9. summarize what changed.

Do not refactor unrelated code.

Do not rename public interfaces without reason.

Do not add dependencies without justification.

---

# 37. Scientific-agent behavior

When changing biological logic:

Always answer:

```text
What assumption changed?
Why?
What equation/rule represents it?
What units are used?
What observable behavior should change?
How can we test it?
```

If any of these are unclear, do not silently invent biology.

Mark the assumption explicitly.

---

# 38. Literature and sources

When implementing biological parameters or models:

Prefer:

```text
peer-reviewed literature
original model papers
official documentation
validated datasets
```

Keep references near configuration or documentation.

Never present an arbitrary parameter as biologically established.

Distinguish:

```text
illustrative parameter
estimated parameter
literature-derived parameter
fitted parameter
```

---

# 39. Visualization

Visualization exists to understand the simulation.

Minimum useful views:

```text
3D cell positions
cell-state coloring
oxygen slices
population curves
radial profiles
```

Do not spend substantial development time on UI before scientific validation.

---

# 40. Performance objective

The architecture should exploit GPU parallelism for:

```text
cell updates
neighbor queries
mechanical interactions
field operations
sampling
```

Avoid repeated:

```text
GPU → CPU → GPU
```

transfers inside the timestep loop.

Metrics may be copied periodically rather than every step.

---

# 41. Dynamic population challenge

Cell division creates a changing population.

Treat this as an explicit architectural challenge.

Investigate:

```text
preallocated arrays
free-slot lists
active masks
compaction
prefix sums
```

before choosing a solution.

Benchmark alternatives.

Do not automatically adopt Python append-style semantics.

---

# 42. Experiment runner

Provide a simple command-line experiment runner.

Example:

```bash
python -m warpcell.run \
    --config configs/tumor_spheroid.yaml
```

Desired output:

```text
runs/
  2026-.../
      config.yaml
      metadata.json
      metrics.csv
      checkpoint/
      figures/
```

This structure should later be usable by OpenCellComms.

---

# 43. Parameter sweeps

After MVP:

```text
python -m warpcell.sweep \
    --config experiments/oxygen_sensitivity.yaml
```

Support varying:

```text
oxygen boundary level
consumption rate
proliferation rate
initial spheroid size
```

Store each run independently.

Do not build a distributed scheduler yet.

---

# 44. First sensitivity study

Recommended first study:

Vary:

```text
oxygen boundary concentration
```

Measure:

```text
final viable-cell count
necrotic fraction
spheroid radius
radial oxygen gradient
```

This provides a clear first scientific experiment.

---

# 45. Future mechanobiology

A later high-value extension is two-way coupling:

```text
CELL BIOLOGY
     │
 proliferation
 death
 migration
     │
     ▼
TISSUE MECHANICS
     │
 stress
 deformation
 pressure
     │
     ▼
CELL BIOLOGY
```

Potential future tools:

```text
Warp
Newton
Isaac physics
```

Examples:

```text
pressure-dependent proliferation
mechanically constrained growth
tumor-induced tissue deformation
```

Do not implement until the uncoupled ABM is validated.

---

# 46. Potential publication/demo

The project should eventually support a scientific demonstration such as:

> **GPU-accelerated patient-specific cellular digital twins coupling agent-based tumor biology with physical simulation.**

Possible figure:

```text
MRI
 ↓
segmented tumor
 ↓
initial cellular state
 ↓
GPU ABM
 ↓
oxygen / hypoxia / necrosis
 ↓
predicted evolution
 ↓
Isaac visualization
```

This is a useful north star.

It is not an MVP requirement.

---

# 47. Upstream contribution strategy

Potential upstream destinations are different:

### NVIDIA Warp

Contribute:

```text
generic computation capability
performance improvement
new simulation primitive
scientific example
```

### Isaac for Healthcare

Contribute:

```text
dynamic biological-state interface
OpenUSD cellular representation
patient-twin adapter
```

### MONAI

Only contribute if actual imaging/segmentation needs expose a generic missing feature.

### BioNeMo / single-cell ecosystem

Only contribute if spatial-omics integration exposes a generic data-layer need.

Never force a contribution where the project does not naturally belong.

---

# 48. Pull-request discipline

Before proposing anything upstream:

1. verify repository CONTRIBUTING instructions;
2. search existing issues;
3. search open and merged PRs;
4. build a minimal reproducible use case;
5. verify the problem still exists on current main;
6. discuss significant architectural changes first.

Do not open speculative mega-PRs.

Prefer:

```text
small
general
tested
benchmarked
well-motivated
```

contributions.

Never open an upstream PR without explicit human approval.

---

# 49. Commit discipline

Commits should represent coherent changes.

Prefer:

```text
feat: add GPU cell-state representation
feat: implement hash-grid neighbor search
test: validate oxygen diffusion against reference
bench: add population-scaling benchmark
feat: add oxygen-dependent cell lifecycle
```

Avoid meaningless commits such as:

```text
updates
fix stuff
changes
```

---

# 50. Definition of MVP complete

The MVP is complete when all of the following are true:

* GPU cells can be initialized;
* neighborhood queries work;
* cells mechanically separate after overlap;
* cells divide;
* cells die;
* oxygen diffuses in 3D;
* cells consume oxygen;
* oxygen influences cell state;
* simulation produces reproducible results;
* automated tests pass;
* CPU/reference checks exist where appropriate;
* performance is measured at multiple population sizes;
* tumor spheroid experiment can be run from configuration;
* results contain quantitative metrics;
* simulation can be visualized;
* documentation explains equations and assumptions.

Only after this point should patient-specific and Isaac integrations become primary work.

---

# 51. Milestones

## Milestone 0 — Repository foundation

Deliver:

```text
project skeleton
configuration
tests
benchmark infrastructure
CI
```

---

## Milestone 1 — GPU cellular particles

Deliver:

```text
cell arrays
cell initialization
Warp kernels
cell movement
basic visualization
```

Target:

```text
10k+ cells
```

---

## Milestone 2 — Spatial interaction

Deliver:

```text
HashGrid
neighbor search
contact detection
repulsive mechanics
```

Validate against controlled configurations.

---

## Milestone 3 — Cell lifecycle

Deliver:

```text
division
death
contact inhibition
stochastic reproducibility
```

---

## Milestone 4 — Oxygen

Deliver:

```text
3D field
diffusion
boundary conditions
cell consumption
cell sampling
```

Validate diffusion independently.

---

## Milestone 5 — Tumor spheroid

Couple:

```text
cells + mechanics + oxygen
```

Produce:

```text
growth
oxygen gradient
hypoxia
necrosis
```

Analyze quantitatively.

---

## Milestone 6 — Performance

Profile:

```text
1k
10k
100k
```

Identify actual bottlenecks.

Investigate whether any limitation merits a Warp upstream contribution.

---

## Milestone 7 — Patient geometry

Support:

```text
segmentation mask
        ↓
cell initialization inside tissue
```

No clinical claims.

---

## Milestone 8 — OpenUSD / Isaac

Deliver:

```text
cellular state → USD
```

Then build a small Isaac for Healthcare demonstrator.

---

## Milestone 9 — Advanced biology

Candidate additions:

```text
glucose
lactate
metabolic phenotypes
drug response
spatial omics
multiple cell types
```

Implement selectively.

---

## Milestone 10 — OpenCellComms

Expose simulation as a scientific tool.

Example:

```text
"Test how oxygen availability affects necrotic-core formation."
```

Agent generates:

```text
parameter sweep
simulations
analysis
comparison
next experiment
```

---

# 52. What success looks like

Success is NOT:

```text
large codebase
many abstractions
many features
complex GUI
```

Success is:

```text
simple scientific model
+
GPU-native execution
+
clear biological assumptions
+
quantitative validation
+
real performance data
+
compelling visual demonstration
+
one or more naturally discovered upstream contributions
```

The final project should demonstrate that agent-based biological simulation can become a first-class dynamic component of patient digital twins.

---

# 53. Immediate first task for coding agent

Start by investigating the current NVIDIA Warp API and repository.

Do NOT implement the full simulator immediately.

Produce:

```text
docs/architecture_spike.md
```

containing:

1. recommended Warp data representation for 100k–1M dynamic cells;
2. recommended neighbor-query implementation;
3. strategy for dynamic creation/removal of cells;
4. options for 3D scalar diffusion;
5. GPU-memory estimates;
6. expected CPU↔GPU synchronization points;
7. relevant Warp examples/tests already present upstream;
8. potential technical risks;
9. minimal architecture for Milestones 1–2;
10. a benchmark plan.

Then implement a technical spike containing only:

```text
10k spherical cells
+
Warp HashGrid neighbor search
+
simple overlap repulsion
```

Benchmark it.

Do not implement oxygen or proliferation until this foundation is verified.
