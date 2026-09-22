"""Warp kernels for the tissue region: SDF sampling and confinement of cells inside it.

Confinement law (docs/model.md), overdamped like the cell-cell contact:

    phi_i = sdf(x_i)                          signed distance, negative inside
    p_i   = phi_i + r_i                       how far the cell surface pokes out of the tissue
    v_i  += - wall_rate * p_i * n_i           if p_i > 0, n_i = grad(sdf)/|grad(sdf)| (outward)

``wall_rate`` [1/h] plays the role of k/gamma. The SDF and its node gradient are sampled
trilinearly; positions outside the grid are clamped to it.
"""

import warp as wp


@wp.func
def trilinear_cell(u: wp.vec3, nx: int, ny: int, nz: int):
    """Base node and fractions for a grid-space position, clamped to the grid."""
    ux = wp.clamp(u[0], 0.0, float(nx - 1))
    uy = wp.clamp(u[1], 0.0, float(ny - 1))
    uz = wp.clamp(u[2], 0.0, float(nz - 1))
    i0 = wp.min(int(wp.floor(ux)), nx - 2)
    j0 = wp.min(int(wp.floor(uy)), ny - 2)
    k0 = wp.min(int(wp.floor(uz)), nz - 2)
    return i0, j0, k0, ux - float(i0), uy - float(j0), uz - float(k0)


@wp.func
def sample_scalar(field: wp.array3d(dtype=wp.float32), i0: int, j0: int, k0: int, fx: float, fy: float, fz: float) -> float:
    c00 = field[i0, j0, k0] * (1.0 - fx) + field[i0 + 1, j0, k0] * fx
    c10 = field[i0, j0 + 1, k0] * (1.0 - fx) + field[i0 + 1, j0 + 1, k0] * fx
    c01 = field[i0, j0, k0 + 1] * (1.0 - fx) + field[i0 + 1, j0, k0 + 1] * fx
    c11 = field[i0, j0 + 1, k0 + 1] * (1.0 - fx) + field[i0 + 1, j0 + 1, k0 + 1] * fx
    return (c00 * (1.0 - fy) + c10 * fy) * (1.0 - fz) + (c01 * (1.0 - fy) + c11 * fy) * fz


@wp.func
def sample_vector(field: wp.array3d(dtype=wp.vec3), i0: int, j0: int, k0: int, fx: float, fy: float, fz: float) -> wp.vec3:
    c00 = field[i0, j0, k0] * (1.0 - fx) + field[i0 + 1, j0, k0] * fx
    c10 = field[i0, j0 + 1, k0] * (1.0 - fx) + field[i0 + 1, j0 + 1, k0] * fx
    c01 = field[i0, j0, k0 + 1] * (1.0 - fx) + field[i0 + 1, j0, k0 + 1] * fx
    c11 = field[i0, j0 + 1, k0 + 1] * (1.0 - fx) + field[i0 + 1, j0 + 1, k0 + 1] * fx
    return (c00 * (1.0 - fy) + c10 * fy) * (1.0 - fz) + (c01 * (1.0 - fy) + c11 * fy) * fz


@wp.kernel
def sample_sdf(
    sdf: wp.array3d(dtype=wp.float32),
    position: wp.array(dtype=wp.vec3),
    origin: wp.vec3,
    inv_dx: wp.float32,
    out: wp.array(dtype=wp.float32),
):
    c = wp.tid()
    i0, j0, k0, fx, fy, fz = trilinear_cell((position[c] - origin) * inv_dx, sdf.shape[0], sdf.shape[1], sdf.shape[2])
    out[c] = sample_scalar(sdf, i0, j0, k0, fx, fy, fz)


@wp.kernel
def confinement_velocities(
    sdf: wp.array3d(dtype=wp.float32),
    gradient: wp.array3d(dtype=wp.vec3),
    origin: wp.vec3,
    inv_dx: wp.float32,
    wall_rate: wp.float32,
    position: wp.array(dtype=wp.vec3),
    radius: wp.array(dtype=wp.float32),
    velocity: wp.array(dtype=wp.vec3),
):
    c = wp.tid()
    i0, j0, k0, fx, fy, fz = trilinear_cell((position[c] - origin) * inv_dx, sdf.shape[0], sdf.shape[1], sdf.shape[2])
    penetration = sample_scalar(sdf, i0, j0, k0, fx, fy, fz) + radius[c]
    if penetration <= 0.0:
        return
    n = sample_vector(gradient, i0, j0, k0, fx, fy, fz)
    length = wp.length(n)
    if length < 1.0e-8:
        return
    velocity[c] = velocity[c] - (wall_rate * penetration / length) * n
