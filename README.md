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
pytest                                         # CPU-safe suite
pytest -m gpu                                  # CUDA-only tests (skipped without CUDA)
python examples/spike_repulsion.py             # 10k cells + HashGrid + overlap repulsion
python examples/growth_contact_inhibition.py   # growth by division under contact inhibition
python examples/spheroid_oxygen.py             # coupled spheroid: growth + oxygen + hypoxia + necrosis
python benchmarks/bench_mechanics.py           # per-kernel mechanics timings at 1k / 10k / 100k cells
python benchmarks/bench_lifecycle.py           # lifecycle and full cell-step timings
python benchmarks/bench_field.py               # oxygen deposit / SOR sweep / sample / update timings
```

## Status

Milestones 0–4 done and validated on CPU: cell arrays with preallocated capacity, HashGrid
neighbour search, overdamped overlap repulsion, stochastic division/death with deterministic
daughter allocation, contact inhibition, a quasi-steady oxygen field (red-black SOR with
Michaelis–Menten uptake) coupled to hypoxia, reduced division and anoxic death. The coupled
run in `examples/spheroid_oxygen.py` produces the expected growth → oxygen gradient →
hypoxia → necrotic core with a viable rim, all emergent from the rules. Next: the
reproducible experiment runner (see [TODO.md](TODO.md)).
Design notes: [docs/architecture_spike.md](docs/architecture_spike.md).
