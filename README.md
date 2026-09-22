# WarpBioCell

GPU-accelerated agent-based biological tissue simulator built on [NVIDIA Warp](https://github.com/NVIDIA/warp).

First scientific target: a tumor spheroid whose cells proliferate, die, interact mechanically, consume oxygen and respond to the local microenvironment.

Project rules and model specification: [AGENTS.md](AGENTS.md). Current task: [TODO.md](TODO.md).

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
```

Or with plain pip: `pip install -e ".[dev]"`.

Warp has no CUDA backend on macOS, so local runs are CPU-only. All code takes the device from configuration; CUDA-only tests are marked `gpu` and skipped when no CUDA device exists.

## Commands

```bash
python -m warpbiocell.run --config configs/tumor_spheroid.yaml        # baseline experiment -> runs/<timestamp>_tumor_spheroid/
python -m warpbiocell.run --config configs/tumor_spheroid.yaml \
    --set oxygen.boundary_mmHg=150 --set oxygen.grid.box_um=1200       # culture-like variant via overrides
python -m warpbiocell.sweep --config configs/tumor_spheroid.yaml \
    --sweep configs/sweeps/oxygen_boundary.yaml                        # 5 boundary values x 3 seeds -> summary.csv
python examples/analyze_sweep.py runs/<timestamp>_oxygen_boundary --zero-order-oxygen --figure sweep.png
pytest                                         # CPU-safe suite
pytest -m gpu                                  # CUDA-only tests (skipped without CUDA)
python examples/spike_repulsion.py             # 10k cells + HashGrid + overlap repulsion
python examples/growth_contact_inhibition.py   # growth by division under contact inhibition
python examples/spheroid_oxygen.py             # coupled spheroid: growth + oxygen + hypoxia + necrosis
python benchmarks/bench_mechanics.py           # per-kernel mechanics timings at 1k / 10k / 100k cells
python benchmarks/bench_lifecycle.py           # lifecycle and full cell-step timings
python benchmarks/bench_field.py               # oxygen deposit / SOR sweep / sample / update timings
```

A run directory holds `config.yaml` (verbatim), `metadata.json` (resolved config, seed,
versions, git commit, device, status), `metrics.csv`, `profiles.csv`, `checkpoint/*.npz` and
`figures/*.png` (matplotlib, optional: `pip install -e ".[plots]"`).

## On a CUDA machine

Nothing has been run on CUDA yet; everything takes the device from configuration. Checklist:

```bash
pip install -e ".[dev]"
python -c "import warp as wp; wp.init(); print(wp.get_cuda_devices())"
pytest                                          # CPU suite plus the gpu-marked CUDA-vs-CPU checks
pytest -m gpu -v                                # only the CUDA checks (mechanics, field, lifecycle, experiment)
python benchmarks/bench_mechanics.py --device cuda:0 --sizes 1000 10000 100000 1000000
python benchmarks/bench_lifecycle.py --device cuda:0 --sizes 1000 10000 100000
python benchmarks/bench_field.py     --device cuda:0 --nodes 41 81 161 --cells 10000 100000
python -m warpbiocell.run --config configs/tumor_spheroid.yaml --device cuda:0
```

The benchmarks write `benchmarks/results/*_cuda0.json` next to the CPU files; commit them.
Same-device runs are bitwise reproducible on CUDA too (per-cell RNG streams, prefix-sum slot
allocation, integer deposit); CPU and CUDA agree to float round-off in the physics.

## Status

Milestones 0–5 done and validated on CPU: cell arrays with preallocated capacity, HashGrid
neighbour search, overdamped overlap repulsion, stochastic division/death with deterministic
daughter allocation, contact inhibition, a quasi-steady oxygen field (red-black SOR with
Michaelis–Menten uptake) coupled to hypoxia, reduced division and anoxic death, and a
reproducible experiment runner. The baseline spheroid produces growth → oxygen gradient →
hypoxia → necrotic core with a stable viable rim, all emergent from the rules; numbers in
[docs/results/spheroid_baseline.md](docs/results/spheroid_baseline.md). Next: see
[TODO.md](TODO.md). Design notes: [docs/architecture_spike.md](docs/architecture_spike.md).
