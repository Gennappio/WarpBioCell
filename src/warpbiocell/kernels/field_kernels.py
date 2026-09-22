"""Warp kernels for a regular 3-D scalar field: cell <-> grid transfer and reaction-diffusion.

Equation (AGENTS.md, "Oxygen field"), with ``n`` the number density of consuming cells:

    dO/dt = D lap(O) - n(x) * q(O),      q(O) = q_max * O / (K + O)      (Michaelis-Menten)

Discretisation: 7-point Laplacian on cubic voxels of edge ``dx``. Dirichlet faces are never
written; Neumann faces use mirrored neighbours (``f[-1] = f[1]``). Helper ``neighbor_sum``
always mirrors at the edges, which is only ever reached on Neumann faces because Dirichlet
face nodes return early.

Steady state (the production path): red-black successive over-relaxation with the uptake
linearised at the current iterate, ``c = n q_max / (K + O)``:

    O_gs  = sum_nb(O) / (6 + dx^2 c / D)
    O_new = max((1 - omega) O + omega O_gs, 0)

Nodes of one colour read only nodes of the other colour, so each colour pass is race-free
and deterministic. With ``omega = 1`` the update is positivity-preserving without the clamp;
for ``omega > 1`` the clamp only acts on transients, never at the fixed point.

Cell -> grid deposit uses trilinear weights accumulated as int64 fixed point: integer
atomics are associative, so the density does not depend on the order in which threads run.
"""

import warp as wp

from warpbiocell.cells.model import STATE_DEAD

FIXED_ONE = wp.constant(16777216.0)  # 2^24: fixed-point unit for trilinear weights
FIXED_ONE_INT = 16777216


@wp.func
def neighbor_sum(f: wp.array3d(dtype=wp.float32), i: int, j: int, k: int) -> float:
    im = i - 1
    if im < 0:
        im = 1
    ip = i + 1
    if ip > f.shape[0] - 1:
        ip = f.shape[0] - 2
    jm = j - 1
    if jm < 0:
        jm = 1
    jp = j + 1
    if jp > f.shape[1] - 1:
        jp = f.shape[1] - 2
    km = k - 1
    if km < 0:
        km = 1
    kp = k + 1
    if kp > f.shape[2] - 1:
        kp = f.shape[2] - 2
    return f[im, j, k] + f[ip, j, k] + f[i, jm, k] + f[i, jp, k] + f[i, j, km] + f[i, j, kp]


@wp.func
def is_fixed(i: int, j: int, k: int, nx: int, ny: int, nz: int, bc_x: int, bc_y: int, bc_z: int) -> bool:
    """True on a Dirichlet face (bc == 0)."""
    if bc_x == 0 and (i == 0 or i == nx - 1):
        return True
    if bc_y == 0 and (j == 0 or j == ny - 1):
        return True
    if bc_z == 0 and (k == 0 or k == nz - 1):
        return True
    return False


@wp.kernel
def deposit_trilinear(
    position: wp.array(dtype=wp.vec3),
    cell_state: wp.array(dtype=wp.int32),
    origin: wp.vec3,
    inv_dx: wp.float32,
    accumulator: wp.array3d(dtype=wp.int64),
):
    """Add each living cell's unit weight to the 8 nodes around it (fixed point)."""
    c = wp.tid()
    if cell_state[c] == STATE_DEAD:
        return
    u = (position[c] - origin) * inv_dx
    i0 = int(wp.floor(u[0]))
    j0 = int(wp.floor(u[1]))
    k0 = int(wp.floor(u[2]))
    # Cells outside the grid's trilinear support are ignored (the domain must contain them).
    if i0 < 0 or j0 < 0 or k0 < 0:
        return
    if i0 + 1 >= accumulator.shape[0] or j0 + 1 >= accumulator.shape[1] or k0 + 1 >= accumulator.shape[2]:
        return
    fx = u[0] - float(i0)
    fy = u[1] - float(j0)
    fz = u[2] - float(k0)
    for di in range(2):
        wx = fx
        if di == 0:
            wx = 1.0 - fx
        for dj in range(2):
            wy = fy
            if dj == 0:
                wy = 1.0 - fy
            for dk in range(2):
                wz = fz
                if dk == 0:
                    wz = 1.0 - fz
                w = wx * wy * wz
                wp.atomic_add(accumulator, i0 + di, j0 + dj, k0 + dk, wp.int64(int(w * FIXED_ONE + 0.5)))


@wp.kernel
def fixed_point_to_density(
    accumulator: wp.array3d(dtype=wp.int64),
    inv_voxel_volume: wp.float32,
    density: wp.array3d(dtype=wp.float32),
):
    i, j, k = wp.tid()
    density[i, j, k] = float(accumulator[i, j, k]) / FIXED_ONE * inv_voxel_volume


@wp.kernel
def sample_trilinear(
    field: wp.array3d(dtype=wp.float32),
    position: wp.array(dtype=wp.vec3),
    origin: wp.vec3,
    inv_dx: wp.float32,
    out: wp.array(dtype=wp.float32),
):
    """Trilinear interpolation at each cell; positions outside the grid are clamped to it."""
    c = wp.tid()
    u = (position[c] - origin) * inv_dx
    nx = field.shape[0]
    ny = field.shape[1]
    nz = field.shape[2]
    ux = wp.clamp(u[0], 0.0, float(nx - 1))
    uy = wp.clamp(u[1], 0.0, float(ny - 1))
    uz = wp.clamp(u[2], 0.0, float(nz - 1))
    i0 = wp.min(int(wp.floor(ux)), nx - 2)
    j0 = wp.min(int(wp.floor(uy)), ny - 2)
    k0 = wp.min(int(wp.floor(uz)), nz - 2)
    fx = ux - float(i0)
    fy = uy - float(j0)
    fz = uz - float(k0)
    c00 = field[i0, j0, k0] * (1.0 - fx) + field[i0 + 1, j0, k0] * fx
    c10 = field[i0, j0 + 1, k0] * (1.0 - fx) + field[i0 + 1, j0 + 1, k0] * fx
    c01 = field[i0, j0, k0 + 1] * (1.0 - fx) + field[i0 + 1, j0, k0 + 1] * fx
    c11 = field[i0, j0 + 1, k0 + 1] * (1.0 - fx) + field[i0 + 1, j0 + 1, k0 + 1] * fx
    c0 = c00 * (1.0 - fy) + c10 * fy
    c1 = c01 * (1.0 - fy) + c11 * fy
    out[c] = c0 * (1.0 - fz) + c1 * fz


@wp.kernel
def sor_sweep(
    color: int,
    omega: wp.float32,
    dx2_over_D: wp.float32,  # h
    uptake_max: wp.float32,  # concentration * um^3 / h per cell
    michaelis_k: wp.float32,  # concentration
    bc_x: int,
    bc_y: int,
    bc_z: int,
    density: wp.array3d(dtype=wp.float32),  # cells / um^3
    field: wp.array3d(dtype=wp.float32),
):
    i, j, k = wp.tid()
    if ((i + j + k) & 1) != color:
        return
    if is_fixed(i, j, k, field.shape[0], field.shape[1], field.shape[2], bc_x, bc_y, bc_z):
        return
    o = field[i, j, k]
    c = density[i, j, k] * uptake_max / (michaelis_k + o)
    gs = neighbor_sum(field, i, j, k) / (6.0 + dx2_over_D * c)
    field[i, j, k] = wp.max((1.0 - omega) * o + omega * gs, 0.0)


@wp.kernel
def steady_state_residual(
    dx2_over_D: wp.float32,
    uptake_max: wp.float32,
    michaelis_k: wp.float32,
    inv_reference: wp.float32,  # 1 / reference concentration
    bc_x: int,
    bc_y: int,
    bc_z: int,
    density: wp.array3d(dtype=wp.float32),
    field: wp.array3d(dtype=wp.float32),
    residual: wp.array(dtype=wp.float32),  # one element; max-norm
):
    """Dimensionless residual |sum_nb O - (6 + dx^2 c / D) O| / O_ref over updated nodes."""
    i, j, k = wp.tid()
    if is_fixed(i, j, k, field.shape[0], field.shape[1], field.shape[2], bc_x, bc_y, bc_z):
        return
    o = field[i, j, k]
    c = density[i, j, k] * uptake_max / (michaelis_k + o)
    r = neighbor_sum(field, i, j, k) - (6.0 + dx2_over_D * c) * o
    wp.atomic_max(residual, 0, wp.abs(r) * inv_reference)


@wp.kernel
def ftcs_step(
    dt: wp.float32,
    D_over_dx2: wp.float32,  # 1/h
    uptake_max: wp.float32,
    michaelis_k: wp.float32,
    bc_x: int,
    bc_y: int,
    bc_z: int,
    density: wp.array3d(dtype=wp.float32),
    field_in: wp.array3d(dtype=wp.float32),
    field_out: wp.array3d(dtype=wp.float32),
):
    """Explicit Euler / central differences reference step (stable for dt <= dx^2 / (6 D))."""
    i, j, k = wp.tid()
    o = field_in[i, j, k]
    if is_fixed(i, j, k, field_in.shape[0], field_in.shape[1], field_in.shape[2], bc_x, bc_y, bc_z):
        field_out[i, j, k] = o
        return
    lap = D_over_dx2 * (neighbor_sum(field_in, i, j, k) - 6.0 * o)
    uptake = density[i, j, k] * uptake_max * o / (michaelis_k + o)
    field_out[i, j, k] = o + dt * (lap - uptake)
