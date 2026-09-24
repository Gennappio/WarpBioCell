# Upstream contribution strategy

One purpose of this project is to identify genuine opportunities for upstream contribution, primarily to NVIDIA Warp.

Never invent an artificial feature request merely to create a PR. Never open an upstream issue or PR without explicit human approval.

## How a contribution is found

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
consider upstream issue/PR
```

Any upstream contribution must be broadly useful beyond this project.

## Destinations

### NVIDIA Warp

Generic computation capability, performance improvement, new simulation primitive, scientific example.

Areas where this workload may expose something: HashGrid queries, particle interaction patterns, dynamic particle populations, field sampling, scatter operations, USD export, performance bottlenecks.

### Isaac for Healthcare

Dynamic biological-state interface, OpenUSD cellular representation, patient-twin adapter.

### MONAI

Only if actual imaging/segmentation needs expose a generic missing feature.

### BioNeMo / single-cell ecosystem

Only if spatial-omics integration exposes a generic data-layer need.

Never force a contribution where the project does not naturally belong.

## Candidates found so far (not filed)

* **warp.optim.linear on the CPU** (2026-09-24, warp-lang 1.17.0, Apple M-series CPU). The tiled
  dot product the iterative solvers use (`TiledDot.compute`) costs ~1000x a native
  `wp.utils.array_inner` on the same data (28.6 ms against 0.028 ms for 32^3 values), so
  `warp.optim.linear.cg` takes ~160 ms per iteration on a 32^3 Poisson problem on the CPU.
  Minimal reproduction: `warpfvm/benchmarks/repro_warp_cpu_tiled_dot.py`. warpfvm works around
  it with its own CG loop on `array_inner`. Before proposing: check current Warp main and open
  issues, and check whether CUDA is affected (probably not: the tiles target GPUs).

## Before proposing anything

1. verify the repository CONTRIBUTING instructions;
2. search existing issues;
3. search open and merged PRs;
4. build a minimal reproducible use case;
5. verify the problem still exists on current main;
6. discuss significant architectural changes first.

Do not open speculative mega-PRs. Prefer contributions that are small, general, tested, benchmarked and well-motivated.
