"""MicroC's diffusion update written once, run with FiPy and with warpfvm.

``update`` below is MultiSubstanceSimulator.update from MicroCpy with the FiPy calls unchanged:
the only difference between the two runs is the module passed as ``lib`` (and the solver
classes). Cells sit in a spheroid; each has MicroC-like rates in mol/s, binned per voxel.

    python warpfvm/examples/microc_drop_in.py                       # MicroC's 3-D grid, 15^3
    python warpfvm/examples/microc_drop_in.py --n 40 --device cpu   # a finer grid
    python warpfvm/examples/microc_drop_in.py --n 100 --device cuda:0 --no-fipy
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import warp as wp

import warpfvm

SUBSTANCES = {  # MicroC adapter values: D [m^2/s], boundary and initial [mM], decay [1/s]
    "Oxygen": dict(D=1.0e-9, boundary=0.07, decay=0.0, rate=-3.0e-17),
    "Glucose": dict(D=6.7e-10, boundary=5.0, decay=0.0, rate=-5.0e-17),
    "Lactate": dict(D=6.7e-10, boundary=1.0, decay=0.0, rate=1.0e-16),
    "TGFA": dict(D=5.18e-11, boundary=0.0, decay=1.0e-4, rate=2.0e-20),
}


def setup(lib, n, size_m, **mesh_kwargs):
    """MeshManager + _create_fipy_variables_for_substances ("fixed" boundaries)."""
    d = size_m / n
    mesh = lib.Grid3D(dx=d, dy=d, dz=d, nx=n, ny=n, nz=n, **mesh_kwargs)
    variables = {}
    for name, cfg in SUBSTANCES.items():
        var = lib.CellVariable(name=name, mesh=mesh, value=cfg["boundary"])
        var.constrain(cfg["boundary"], mesh.facesTop | mesh.facesBottom | mesh.facesLeft | mesh.facesRight | mesh.facesFront | mesh.facesBack)
        variables[name] = var
    return mesh, variables


def update(lib, LinearLUSolver, LinearGMRESSolver, mesh, variables, sources):
    """MultiSubstanceSimulator.update (steady state), FiPy calls unchanged."""
    for name, var in variables.items():
        config = SUBSTANCES[name]
        source_var = lib.CellVariable(mesh=mesh, value=sources[name])
        decay_rate = config["decay"]
        if decay_rate > 0.0:
            steady_terms = lib.DiffusionTerm(coeff=config["D"]) - lib.ImplicitSourceTerm(coeff=decay_rate)
        else:
            steady_terms = lib.DiffusionTerm(coeff=config["D"])
        equation = steady_terms == -source_var
        solver = LinearLUSolver(iterations=10, tolerance=1e-6)
        equation.solve(var=var, solver=solver)


def spheroid_cells(n_cells, size_m, rng):
    """Cell centres in a ball of radius 0.3 * size around the centre of the box [m]."""
    direction = rng.normal(size=(n_cells, 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    radius = 0.3 * size_m * rng.random(n_cells) ** (1.0 / 3.0)
    return size_m / 2 + direction * radius[:, None]


def run(label, lib, lu, gmres, n, size_m, cells, mesh_kwargs):
    mesh, variables = setup(lib, n, size_m, **mesh_kwargs)
    # warpfvm.bin_rates is FiPy-independent: the same per-voxel field for both libraries
    grid = warpfvm.Grid3D(dx=size_m / n, dy=size_m / n, dz=size_m / n, nx=n, ny=n, nz=n, device="cpu")
    sources = {name: warpfvm.bin_rates(grid, cells, cfg["rate"]) for name, cfg in SUBSTANCES.items()}
    update(lib, lu, gmres, mesh, variables, sources)  # warm-up (kernel compilation, caches)
    mesh, variables = setup(lib, n, size_m, **mesh_kwargs)
    t0 = time.perf_counter()
    update(lib, lu, gmres, mesh, variables, sources)
    fields = {name: np.array(var.value, dtype=np.float64) for name, var in variables.items()}
    elapsed = time.perf_counter() - t0
    print(f"{label:>8}: {1e3 * elapsed:9.1f} ms for {len(SUBSTANCES)} substances on {n}^3 = {n**3} voxels")
    return fields


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=15, help="voxels per axis (MicroC's 3-D workflow: 15)")
    parser.add_argument("--size-um", type=float, default=750.0, help="box edge [um] (MicroC's 3-D workflow: 750)")
    parser.add_argument("--cells", type=int, default=2000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--precision", default="float64", choices=["float64", "float32"])
    parser.add_argument("--no-fipy", action="store_true")
    args = parser.parse_args()
    wp.config.log_level = wp.LOG_WARNING

    size_m = args.size_um * 1e-6
    cells = spheroid_cells(args.cells, size_m, np.random.default_rng(0))
    ours = run("warpfvm", warpfvm, warpfvm.LinearLUSolver, warpfvm.LinearGMRESSolver, args.n, size_m, cells, dict(device=args.device, dtype=args.precision))
    if args.no_fipy:
        return
    import fipy
    from fipy.solvers.scipy import LinearGMRESSolver, LinearLUSolver

    theirs = run("FiPy", fipy, LinearLUSolver, LinearGMRESSolver, args.n, size_m, cells, {})
    for name in SUBSTANCES:
        change = np.ptp(theirs[name])
        diff = np.abs(ours[name] - theirs[name]).max()
        print(f"{name:>8}: range {theirs[name].min():.6g} .. {theirs[name].max():.6g} mM   max |warpfvm - FiPy| = {diff:.2e} mM ({diff / max(change, 1e-300):.1e} of the range)")


if __name__ == "__main__":
    main()
