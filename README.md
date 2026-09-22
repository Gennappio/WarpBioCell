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
pytest                                   # CPU-safe suite
pytest -m gpu                            # CUDA-only tests (skipped without CUDA)
python examples/spike_repulsion.py       # 10k cells + HashGrid + overlap repulsion
python benchmarks/bench_mechanics.py     # per-kernel timings at 1k / 10k / 100k cells
```

## Status

Milestone 0–2 technical spike: cell arrays, HashGrid neighbor search, overdamped overlap repulsion. See [docs/architecture_spike.md](docs/architecture_spike.md).
