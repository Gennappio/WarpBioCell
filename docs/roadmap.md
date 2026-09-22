# Roadmap

Model rules and constraints are in [AGENTS.md](../AGENTS.md). Long-term direction is in [vision.md](vision.md). The task in progress is in [TODO.md](../TODO.md).

## Definition of MVP complete

The MVP is complete when all of the following are true:

* cells can be initialized on the device;
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
* performance is measured at multiple population sizes on a CUDA device;
* the tumor spheroid experiment can be run from configuration;
* results contain quantitative metrics;
* the simulation can be visualized;
* documentation explains equations and assumptions.

Only after this point do patient-specific and Isaac integrations become primary work.

## Baseline experiment: tumor spheroid growth

```text
initial condition   small approximately spherical cluster
environment         3D domain, constant oxygen boundary, no glucose, no vasculature
cells               single tumor-cell phenotype
```

Expected qualitative evolution:

```text
small spheroid → growth → oxygen gradient → inner hypoxia
        → reduced proliferation → possible necrotic center
```

The necrotic core must emerge from the model. Do not hard-code it.

## Milestones

### Milestone 0 — Repository foundation (done 2026-09-22; CI still missing)

Project skeleton, configuration, tests, benchmark infrastructure, CI. Decide the Python version and packaging. Update the Commands section of AGENTS.md.

### Milestone 1 — GPU cellular particles (done 2026-09-22; visualization pending)

Cell arrays, cell initialization, Warp kernels, cell movement, basic visualization. Target: 10k+ cells.

### Milestone 2 — Spatial interaction (done 2026-09-22, CPU-validated)

HashGrid, neighbor search, contact detection, repulsive mechanics. Validate against controlled configurations.

### Milestone 3 — Cell lifecycle (done 2026-09-22, CPU-validated)

Division, death, contact inhibition, stochastic reproducibility (deterministic slot allocation, per-cell RNG).

### Milestone 4 — Oxygen (done 2026-09-22, CPU-validated)

3D field, diffusion solver, boundary conditions, cell consumption, cell sampling. Validate diffusion independently. Settle the quasi-steady-state solver choice.

### Milestone 5 — Tumor spheroid (done 2026-09-22, CPU; quantitative validation against data still open)

Couple cells + mechanics + oxygen. Produce growth, oxygen gradient, hypoxia, necrosis. Analyze quantitatively against the chosen validation data.

### Milestone 5.1 — First scientific experiment (done 2026-09-22)

Sweep runner (`python -m warpbiocell.sweep`), oxygen-boundary sensitivity study with seed
replicates, `docs/model.md` (equations and assumptions in one place), continuous integration.

### Milestone 6 — Performance (waits for the CUDA machine)

Profile at 1k, 10k, 100k, 10^6 cells on CUDA. Identify actual bottlenecks. Investigate whether any limitation merits a Warp upstream contribution. The user runs this by uploading the repository to the GPU machine (README, "On a CUDA machine").

### Milestone 7 — Patient geometry (done 2026-09-22 for synthetic shapes and voxel masks; a real segmentation not yet exercised)

Segmentation mask → cell initialization inside tissue. Requires an answer to the scale-gap question in [vision.md](vision.md). No clinical claims.

### Milestone 8 — OpenUSD / Isaac (USD export done 2026-09-22; Isaac demonstrator deferred to a machine that has it)

Cellular state → USD, then a small Isaac for Healthcare demonstrator.

### Milestone 9 — Advanced biology (glucose and lactate done 2026-09-22; pH, drug response, spatial omics, multiple cell types open)

Candidates: glucose, lactate, pH, metabolic phenotypes, drug response, spatial omics, multiple cell types. Implement selectively; MicroC's `diffusion-parameters.txt` (docs/reference/microc_parameters.md) is the reference set for the metabolic fields.

### Milestone 10 — OpenCellComms (done 2026-09-22 as the MicroC_warp adapter in the OpenCellComms repository; docs/integrations/opencellcomms.md)

Expose the simulation as a scientific tool. Example: "Test how oxygen availability affects necrotic-core formation." The agent generates the parameter sweep, simulations, analysis, comparison and next experiment.

### Milestone 11 — Gene regulatory network per cell (last, decided 2026-09-22)

A Boolean network inside every agent, MaBoSS-style, as an alternative phenotype model to the built-in rules. Design notes in [vision.md](vision.md) ("Gene network"). Needs the MicroC network exported from GINsim (MaBoSS `.bnd/.cfg` or BoolNet `.bnet`). This is what makes MicroC's gene-perturbation experiments reproducible on the GPU; it is deliberately scheduled after the imaging and visualization milestones.

## Experiment runner

A simple command-line runner:

```bash
python -m warpbiocell.run --config configs/tumor_spheroid.yaml
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

## Parameter sweeps (after MVP)

```bash
python -m warpbiocell.sweep --config experiments/oxygen_sensitivity.yaml
```

Support varying oxygen boundary level, consumption rate, proliferation rate and initial spheroid size. Store each run independently. Do not build a distributed scheduler yet.

### First sensitivity study

Vary the oxygen boundary concentration. Measure final viable-cell count, necrotic fraction, spheroid radius and radial oxygen gradient.
