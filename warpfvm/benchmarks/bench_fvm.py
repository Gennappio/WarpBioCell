"""warpfvm against FiPy on MicroC's steady oxygen problem, at growing grid sizes.

Problem (the shape of MicroC's metabolic diffusion node): cubic grid of n^3 voxels of 20 um,
D = 1e-9 m^2/s, 0.07 mM fixed on every face, a spheroid of consuming cells (3e-17 mol/s per
cell, one cell per voxel) filling the middle 60 % of the box, SI units as in MicroC.

For each n it measures:

    warpfvm   assembly; cold solve from the uniform boundary value with LinearLUSolver
              (MicroC's call); warm re-solve after a 1 % change of the uptake (one step of
              MicroC's Picard coupling); iterations and time per CG iteration
    FiPy      the same cold solve with FiPy's LinearLUSolver (SuperLU on the CPU, what MicroC
              runs) and with FiPy's LinearPCGSolver (scipy CG) at the same tolerance, up to
              --fipy-max voxels per axis
    agreement max |warpfvm - FiPy| relative to the oxygen dip (FiPy's LU answer when it ran,
              else its PCG answer at the same 1e-12 tolerance)

Every record carries the device and the precision. Warp's CPU kernels run on one thread; GPU
numbers come only from a CUDA device.

    python warpfvm/benchmarks/bench_fvm.py --device cpu --sizes 15 32 48 64 96
    python warpfvm/benchmarks/bench_fvm.py --device cuda:0 --sizes 15 32 64 128 192 256 --fipy-max 96
    python warpfvm/benchmarks/bench_fvm.py --device cuda:0 --precision float32 --sizes 64 128 256
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import time
import warnings
from pathlib import Path

import numpy as np
import warp as wp

import warpfvm as wf

DX = 20e-6  # m
D = 1.0e-9  # m^2/s
BOUNDARY = 0.07  # mM
UPTAKE = 3.0e-17  # mol/s per cell


def _problem(lib, n, **mesh_kwargs):
    mesh = lib.Grid3D(dx=DX, dy=DX, dz=DX, nx=n, ny=n, nz=n, **mesh_kwargs)
    x, y, z = (np.asarray(c) for c in mesh.cellCenters)
    centre = n * DX / 2
    inside = (x - centre) ** 2 + (y - centre) ** 2 + (z - centre) ** 2 < (0.3 * n * DX) ** 2
    rate = np.where(inside, -UPTAKE / DX**3, 0.0)  # mol/(m^3 s) = mM/s, negative = consumption
    var = lib.CellVariable(mesh=mesh, value=BOUNDARY)
    var.constrain(BOUNDARY, mesh.exteriorFaces)
    return mesh, var, rate


def _equation(lib, mesh, rate):
    source = lib.CellVariable(mesh=mesh, value=rate)
    return lib.DiffusionTerm(coeff=D) == -source


def _sync(device):
    if device is not None:
        wp.synchronize_device(device)


def bench_warpfvm(n, device, precision, repeats):
    mesh, var, rate = _problem(wf, n, device=device, dtype=precision)
    eq = _equation(wf, mesh, rate)
    eq.solve(var=var, solver=wf.LinearLUSolver())  # compile and load every kernel once
    assemble, cold, warm, iterations, warm_iterations = [], [], [], 0, 0
    for _ in range(repeats):
        var.setValue(BOUNDARY)
        _sync(device)
        t0 = time.perf_counter()
        system = eq.assemble(var=var)
        _sync(device)
        t1 = time.perf_counter()
        solver = wf.LinearLUSolver()
        solver._solve(system, var)
        _sync(device)
        t2 = time.perf_counter()
        assemble.append(t1 - t0)
        cold.append(t2 - t1)
        iterations = solver.convergence.iterations
        converged = solver.convergence.converged
        solution = var.value.astype(np.float64)
        warm_eq = _equation(wf, mesh, 1.01 * rate)
        _sync(device)
        t3 = time.perf_counter()
        warm_solver = wf.LinearLUSolver()
        warm_eq.solve(var=var, solver=warm_solver)
        _sync(device)
        warm.append(time.perf_counter() - t3)
        warm_iterations = warm_solver.convergence.iterations
    cold_s = float(np.median(cold))
    itemsize = 8 if precision == "float64" else 4
    return solution, {
        "assemble_ms": 1e3 * float(np.median(assemble)),
        "cold_solve_ms": 1e3 * cold_s,
        "cold_iterations": int(iterations),
        "ms_per_iteration": 1e3 * cold_s / max(iterations, 1),
        "warm_solve_ms": 1e3 * float(np.median(warm)),
        "warm_iterations": int(warm_iterations),
        "converged": bool(converged),
        "device_bytes_estimate": int(9 * n**3 * itemsize),
    }


def bench_fipy(n, repeats, with_lu):
    import fipy as fp
    from fipy.solvers.scipy import LinearLUSolver, LinearPCGSolver

    out = {}
    solution = None
    solvers = [("fipy_pcg", lambda: LinearPCGSolver(tolerance=1e-12, iterations=100000))]
    if with_lu:
        solvers.insert(0, ("fipy_lu", lambda: LinearLUSolver(iterations=10, tolerance=1e-6)))
    for name, make in solvers:
        times = []
        for _ in range(repeats):
            mesh, var, rate = _problem(fp, n)
            eq = _equation(fp, mesh, rate)
            t0 = time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                eq.solve(var=var, solver=make())
            times.append(time.perf_counter() - t0)
        out[f"{name}_solve_ms"] = 1e3 * float(np.median(times))
        if solution is None:  # FiPy's LU answer when it ran, else its PCG answer at 1e-12
            solution = np.array(var.value, dtype=np.float64)
            out["fipy_reference"] = name
    return solution, out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--precision", default="float64", choices=["float64", "float32"])
    parser.add_argument("--sizes", type=int, nargs="+", default=[15, 32, 48, 64])
    parser.add_argument("--fipy-max", type=int, default=64, help="largest n solved with FiPy's PCG (0 disables FiPy)")
    parser.add_argument("--fipy-lu-max", type=int, default=48, help="largest n solved with FiPy's LU (SuperLU fill-in grows fast in 3-D)")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    wp.config.log_level = wp.LOG_WARNING
    wp.init()
    device = wp.get_device(args.device)
    try:
        import fipy

        fipy_version = fipy.__version__
    except ImportError:
        fipy_version = None
        args.fipy_max = 0

    records = []
    print(f"device={device.alias} ({device.name})  precision={args.precision}  warp={wp.__version__}  fipy={fipy_version}")
    print(f"{'n':>4} {'cells':>9} {'assemble':>9} {'cold':>9} {'iters':>6} {'ms/it':>7} {'warm':>9} {'w.it':>5} {'FiPy LU':>10} {'FiPy PCG':>10} {'|diff|/dip':>11}")
    for n in args.sizes:
        solution, record = bench_warpfvm(n, device, args.precision, args.repeats)
        record = {"n": n, "cells": n**3, **record}
        if args.fipy_max and n <= args.fipy_max:
            reference, fipy_record = bench_fipy(n, max(1, min(args.repeats, 2)), with_lu=n <= args.fipy_lu_max)
            record.update(fipy_record)
            dip = BOUNDARY - reference.min()
            record["oxygen_dip_mM"] = float(dip)
            record["max_diff_over_dip"] = float(np.abs(solution - reference).max() / dip)
        records.append(record)
        r = record
        print(
            f"{n:>4} {n**3:>9} {r['assemble_ms']:>7.2f}ms {r['cold_solve_ms']:>7.1f}ms {r['cold_iterations']:>6} "
            f"{r['ms_per_iteration']:>7.3f} {r['warm_solve_ms']:>7.1f}ms {r['warm_iterations']:>5} "
            + (f"{r['fipy_lu_solve_ms']:>8.1f}ms" if "fipy_lu_solve_ms" in r else f"{'-':>10}")
            + (f" {r['fipy_pcg_solve_ms']:>8.1f}ms {r['max_diff_over_dip']:>11.2e}" if "fipy_pcg_solve_ms" in r else f" {'-':>10} {'-':>11}")
        )

    result = {
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "warpfvm_version": wf.__version__,
        "warp_version": wp.__version__,
        "fipy_version": fipy_version,
        "device": {"alias": device.alias, "name": device.name, "is_cuda": device.is_cuda},
        "precision": args.precision,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "problem": {"dx_m": DX, "D_m2_per_s": D, "boundary_mM": BOUNDARY, "uptake_mol_per_s_per_cell": UPTAKE, "solver": "LinearLUSolver"},
        "records": records,
    }
    out = args.out or Path(__file__).parent / "results" / f"fvm_{device.alias.replace(':', '')}_{args.precision}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
