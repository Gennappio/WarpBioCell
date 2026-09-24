"""CUDA against the CPU (skipped without a CUDA device).

Expectations: float64 on CUDA agrees with float64 on the CPU to the solver tolerance (the
reductions add in a different order, so CG iterates differ at round-off); the CUDA-graph loop
gives the same numbers as the eager loop; two runs on the same GPU agree bitwise when the
native reductions are deterministic; float32 stays within 1e-4 of the consumption dip.
"""

import numpy as np
import pytest

import warpfvm as wf

pytestmark = pytest.mark.gpu


def _spheroid(device, dtype="float64", n=24, dx=20e-6):
    mesh = wf.Grid3D(dx=dx, dy=dx, dz=dx, nx=n, ny=n, nz=n, device=device, dtype=dtype)
    x, y, z = mesh.cellCenters
    c = n * dx / 2
    r2 = (x - c) ** 2 + (y - c) ** 2 + (z - c) ** 2
    consumption = np.where(r2 < (0.3 * n * dx) ** 2, 3.0e-17 / dx**3, 0.0)
    phi = wf.CellVariable(mesh=mesh, value=0.07)
    phi.constrain(0.07, mesh.exteriorFaces)
    sink = wf.CellVariable(mesh=mesh, value=np.where(r2 < (0.2 * n * dx) ** 2, 1e-3, 0.0))
    eq = wf.DiffusionTerm(coeff=1e-9) - wf.ImplicitSourceTerm(coeff=sink) == consumption
    return mesh, phi, eq


def _solve(device, dtype="float64", **solver_kwargs):
    _, phi, eq = _spheroid(device, dtype)
    solver = wf.LinearLUSolver(**solver_kwargs)
    eq.solve(var=phi, solver=solver)
    assert solver.convergence.converged
    return phi.value.astype(np.float64), solver.convergence


def test_cuda_float64_matches_cpu(cuda):
    cpu, _ = _solve("cpu")
    gpu, _ = _solve(cuda)
    assert np.abs(gpu - cpu).max() <= 1e-10 * np.abs(cpu).max()


def test_cuda_graph_loop_matches_eager_loop(cuda):
    eager, c1 = _solve(cuda, use_cuda_graph=False)
    graph, c2 = _solve(cuda, use_cuda_graph=True)
    assert c1.iterations == c2.iterations
    assert np.abs(graph - eager).max() <= 1e-13 * np.abs(eager).max()


def test_same_gpu_runs_are_bitwise_identical(cuda):
    first, _ = _solve(cuda)
    second, _ = _solve(cuda)
    assert np.array_equal(first, second)


def test_cuda_float32_close_to_float64(cuda):
    reference, _ = _solve("cpu")
    single, _ = _solve(cuda, dtype="float32")
    dip = 0.07 - reference.min()
    assert np.abs(single - reference).max() < 1e-4 * dip


def test_cuda_transient_steps_match_cpu(cuda):
    fields = []
    for device in ("cpu", cuda):
        mesh, phi, _ = _spheroid(device, n=16)
        eq = wf.TransientTerm() == wf.DiffusionTerm(coeff=1e-9) - 2e-3
        for _ in range(4):
            eq.solve(var=phi, dt=5.0, solver=wf.LinearLUSolver())
        fields.append(phi.value)
    assert np.abs(fields[1] - fields[0]).max() <= 1e-10 * np.abs(fields[0]).max()


def test_cuda_singular_problem(cuda, rng):
    mesh = wf.Grid2D(dx=1.0, dy=1.0, nx=32, ny=24, device=cuda)
    source = rng.normal(size=mesh.numberOfCells)
    source -= source.mean()
    phi = wf.CellVariable(mesh=mesh, value=0.5)
    solver = wf.LinearGMRESSolver(tolerance=1e-10)
    (wf.DiffusionTerm(coeff=1.0) == -source).solve(var=phi, solver=solver)
    assert solver.convergence.singular and solver.convergence.converged
    assert phi.value.mean() == pytest.approx(0.5, abs=1e-12)


def test_cuda_against_fipy(cuda):
    fp = pytest.importorskip("fipy")
    from fipy.solvers.scipy import LinearLUSolver

    n, dx = 16, 20e-6
    mesh = fp.Grid3D(dx=dx, dy=dx, dz=dx, nx=n, ny=n, nz=n)
    phi = fp.CellVariable(mesh=mesh, value=0.07)
    phi.constrain(0.07, mesh.exteriorFaces)
    source = fp.CellVariable(mesh=mesh, value=-np.linspace(0, 1e-2, n**3))
    (fp.DiffusionTerm(coeff=1e-9) - fp.ImplicitSourceTerm(coeff=1e-4) == -source).solve(var=phi, solver=LinearLUSolver())
    ours_mesh = wf.Grid3D(dx=dx, dy=dx, dz=dx, nx=n, ny=n, nz=n, device=cuda)
    ours = wf.CellVariable(mesh=ours_mesh, value=0.07)
    ours.constrain(0.07, ours_mesh.exteriorFaces)
    ours_source = wf.CellVariable(mesh=ours_mesh, value=-np.linspace(0, 1e-2, n**3))
    (wf.DiffusionTerm(coeff=1e-9) - wf.ImplicitSourceTerm(coeff=1e-4) == -ours_source).solve(var=ours, solver=wf.LinearLUSolver())
    assert np.abs(ours.value - np.asarray(phi.value)).max() <= 1e-10 * 0.07
