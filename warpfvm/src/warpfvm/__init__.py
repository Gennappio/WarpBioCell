"""warpfvm: FiPy-style finite volumes on uniform grids, run by NVIDIA Warp on the CPU or CUDA.

The subset of FiPy that MicroC uses, with FiPy's names, cell and face numbering and
discretization, so that MicroC's diffusion code runs by changing its imports:

    from fipy import Grid3D, CellVariable, DiffusionTerm, ImplicitSourceTerm, TransientTerm
    from fipy.solvers.scipy import LinearGMRESSolver, LinearLUSolver
becomes
    from warpfvm import Grid3D, CellVariable, DiffusionTerm, ImplicitSourceTerm, TransientTerm
    from warpfvm.solvers import LinearGMRESSolver, LinearLUSolver

Meshes carry the device and the precision (``Grid3D(..., device="cuda:0", dtype="float32")``,
or the WARPFVM_DEVICE / WARPFVM_PRECISION environment variables). See README.md for what is
supported and what is not.
"""

from warpfvm.mesh import Grid1D, Grid2D, Grid3D, UniformGrid
from warpfvm.solvers import (
    Convergence,
    ConvergenceWarning,
    DefaultSolver,
    LinearBicgstabSolver,
    LinearCGSolver,
    LinearGMRESSolver,
    LinearLUSolver,
    LinearPCGSolver,
)
from warpfvm.sources import bin_rates, cell_ids
from warpfvm.system import LinearSystem
from warpfvm.terms import DiffusionTerm, ImplicitDiffusionTerm, ImplicitSourceTerm, TransientTerm
from warpfvm.variables import CellVariable

__version__ = "0.1.0"

__all__ = [
    "CellVariable",
    "Convergence",
    "ConvergenceWarning",
    "DefaultSolver",
    "DiffusionTerm",
    "Grid1D",
    "Grid2D",
    "Grid3D",
    "ImplicitDiffusionTerm",
    "ImplicitSourceTerm",
    "LinearBicgstabSolver",
    "LinearCGSolver",
    "LinearGMRESSolver",
    "LinearLUSolver",
    "LinearPCGSolver",
    "LinearSystem",
    "TransientTerm",
    "UniformGrid",
    "bin_rates",
    "cell_ids",
]
