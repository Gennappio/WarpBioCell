"""Warp kernels for overdamped contact mechanics between spherical cells.

Model (AGENTS.md, "Mechanics"):

    overlap_ij = r_i + r_j - |x_i - x_j|
    F_ij       = k * overlap_ij * n_ij        for overlap_ij > 0, n_ij = (x_i - x_j)/|x_i - x_j|
    dx_i/dt    = (1/gamma) * sum_j F_ij

Only the ratio ``k/gamma`` [1/h] enters the kernels. Each thread gathers the contributions of
its own cell, so no atomics are needed and the result is bitwise reproducible on a given
device for a given hash-grid build.
"""

import warp as wp

# Pairs closer than this [um] are skipped: their separation direction is undefined. Initial
# configurations and daughter placement must never create coincident centres.
COINCIDENT_EPS = wp.constant(1.0e-6)


@wp.kernel
def contact_velocities(
    grid: wp.uint64,
    position: wp.array(dtype=wp.vec3),
    radius: wp.array(dtype=wp.float32),
    query_radius: wp.float32,
    rate: wp.float32,  # k / gamma [1/h]
    velocity: wp.array(dtype=wp.vec3),
    contact_count: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    # Process cells in hash-grid order so neighbouring threads touch neighbouring memory.
    i = wp.hash_grid_point_id(grid, tid)

    x_i = position[i]
    r_i = radius[i]

    v = wp.vec3(0.0, 0.0, 0.0)
    n_contacts = int(0)

    for j in wp.hash_grid_query(grid, x_i, query_radius):
        if j != i:
            d_vec = x_i - position[j]
            d = wp.length(d_vec)
            overlap = r_i + radius[j] - d
            if overlap > 0.0 and d > COINCIDENT_EPS:
                v = v + (rate * overlap / d) * d_vec
                n_contacts += 1

    velocity[i] = v
    contact_count[i] = n_contacts


@wp.kernel
def integrate_positions(
    position: wp.array(dtype=wp.vec3),
    velocity: wp.array(dtype=wp.vec3),
    dt: wp.float32,
):
    i = wp.tid()
    position[i] = position[i] + velocity[i] * dt
