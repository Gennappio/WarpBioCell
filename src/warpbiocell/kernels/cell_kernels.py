"""Warp kernels for the cell lifecycle: death, contact inhibition, division, daughter placement.

Rules (AGENTS.md, "Proliferation", "Contact inhibition", "Death"):

    P(die during dt)    = 1 - exp(-death_rate * dt)          (oxygen-independent placeholder
                                                              until the oxygen milestone)
    crowded             = neighbor_count >= inhibition_threshold   -> QUIESCENT, no division
    P(divide during dt) = 1 - exp(-division_rate * dt)        for PROLIFERATIVE cells

Stated simplifications, to be revisited explicitly:
    * the cell cycle is memoryless: no refractory period after division, no minimum age;
    * a cell divides at most once per step and a daughter cannot divide in the step of its
      birth, so the expected free growth per step is (1 + p) with p = 1 - exp(-lambda dt),
      i.e. an effective rate ln(1 + p)/dt that is below lambda by O(lambda dt): 0.5% at
      dt = 0.25 h and lambda = 0.0289/h;
    * cell volume is not conserved at division: the daughter gets the parent's radius, so
      growth is implicit in the division rate; the resulting overlap is relaxed by mechanics;
    * the pair is placed symmetrically about the parent's centre, conserving the centre of mass.

Every random draw comes from the cell's own persistent stream (``rng_state``), so the outcome
depends only on the seed and the history of that cell, never on thread scheduling.
"""

import warp as wp

from warpbiocell.cells.model import NUM_STATES, STATE_DEAD, STATE_PROLIFERATIVE, STATE_QUIESCENT


@wp.kernel
def lifecycle_decide(
    dt: wp.float32,
    death_rate: wp.float32,
    division_rate: wp.float32,
    inhibition_threshold: wp.int32,
    cell_state: wp.array(dtype=wp.int32),
    age: wp.array(dtype=wp.float32),
    neighbor_count: wp.array(dtype=wp.int32),
    rng_state: wp.array(dtype=wp.uint32),
    divide_flag: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    divide_flag[i] = 0

    if cell_state[i] == STATE_DEAD:
        return

    age[i] = age[i] + dt
    rng = rng_state[i]

    # Death is evaluated first: a cell that dies this step neither divides nor changes state
    # otherwise. The draw is always consumed so the stream advances identically whether or not
    # death_rate is zero.
    p_death = 1.0 - wp.exp(-death_rate * dt)
    if wp.randf(rng) < p_death:
        cell_state[i] = STATE_DEAD
        rng_state[i] = rng
        return

    # HYPOXIC takes precedence over the crowding rule once oxygen exists (Milestone 4).
    if neighbor_count[i] >= inhibition_threshold:
        cell_state[i] = STATE_QUIESCENT
    else:
        cell_state[i] = STATE_PROLIFERATIVE
        p_divide = 1.0 - wp.exp(-division_rate * dt)
        if wp.randf(rng) < p_divide:
            divide_flag[i] = 1

    rng_state[i] = rng


@wp.kernel
def place_daughters(
    count: wp.int32,
    placement_factor: wp.float32,  # centre-to-centre distance / parent radius
    divide_flag: wp.array(dtype=wp.int32),
    division_offset: wp.array(dtype=wp.int32),  # inclusive prefix sum of divide_flag
    position: wp.array(dtype=wp.vec3),
    radius: wp.array(dtype=wp.float32),
    cell_state: wp.array(dtype=wp.int32),
    cell_type: wp.array(dtype=wp.int32),
    age: wp.array(dtype=wp.float32),
    rng_state: wp.array(dtype=wp.uint32),
    velocity: wp.array(dtype=wp.vec3),
    neighbor_count: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    if divide_flag[i] == 0:
        return

    # Deterministic slot: the k-th dividing cell (in slot order) fills slot count + k.
    j = count + division_offset[i] - 1

    rng = rng_state[i]
    direction = wp.sample_unit_sphere_surface(rng)
    rng_state[i] = rng

    half_offset = direction * (0.5 * placement_factor * radius[i])
    x_parent = position[i]
    position[i] = x_parent - half_offset
    position[j] = x_parent + half_offset

    radius[j] = radius[i]
    cell_state[j] = STATE_PROLIFERATIVE
    cell_type[j] = cell_type[i]
    age[j] = 0.0
    velocity[j] = wp.vec3(0.0, 0.0, 0.0)
    neighbor_count[j] = 0
    # rng_state[j] was initialised for every slot at population creation and is left untouched.


@wp.kernel
def count_states(cell_state: wp.array(dtype=wp.int32), counts: wp.array(dtype=wp.int32)):
    i = wp.tid()
    s = cell_state[i]
    if s >= 0 and s < NUM_STATES:
        wp.atomic_add(counts, s, 1)
