"""Minimal reproduction: warp.optim.linear's tiled dot product on the CPU.

With warp-lang 1.17.0 on an Apple M-series CPU, ``TiledDot.compute`` (used by every solver in
warp.optim.linear) takes ~1000x the time of the native ``wp.utils.array_inner`` on the same
data, so ``warp.optim.linear.cg`` on the CPU spends ~160 ms per iteration on a 32^3 problem.
Kept as the reproduction for a possible upstream report (see docs/upstream.md); not filed.

    python warpfvm/benchmarks/repro_warp_cpu_tiled_dot.py
"""

import time

import numpy as np
import warp as wp
from warp._src.optim.linear import TiledDot

wp.config.log_level = wp.LOG_WARNING
wp.init()
device = wp.get_device("cpu")
print(f"warp {wp.__version__} on {device.alias} ({device.name})")
for n in (4096, 32768, 262144):
    a = wp.array(np.random.default_rng(0).random(n), dtype=wp.float64, device=device)
    tiled = TiledDot(max_length=n, device=device, scalar_type=wp.float64, max_column_count=2)
    tiled.compute(a, a)
    out = wp.empty(1, dtype=wp.float64, device=device)
    wp.utils.array_inner(a, a, out=out)
    wp.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(3):
        tiled.compute(a, a)
    wp.synchronize_device(device)
    t1 = time.perf_counter()
    for _ in range(3):
        wp.utils.array_inner(a, a, out=out)
    wp.synchronize_device(device)
    t2 = time.perf_counter()
    same = np.isclose(tiled.col(0).numpy()[0], out.numpy()[0])
    print(f"n={n:>7}: TiledDot {1e3 * (t1 - t0) / 3:8.2f} ms   array_inner {1e3 * (t2 - t1) / 3:7.3f} ms   ratio {(t1 - t0) / (t2 - t1):6.0f}x   same value: {same}")
