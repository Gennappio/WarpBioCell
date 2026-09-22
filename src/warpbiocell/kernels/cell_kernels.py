"""Warp kernels for the cell lifecycle: death, hypoxia, contact inhibition, division, placement.

Rules (AGENTS.md "Proliferation", "Contact inhibition", "Death"), with O the oxygen sampled
at the cell [mmHg]:

    lethal              = O < death_threshold                          (oxygen only), or
                        = O < death_threshold and G < glucose_death_threshold
                                                    when necrosis_requires_glucose (MicroC rule)
    death rate          = max(death_rate, anoxic_death_rate)   if lethal
                        = death_rate                            otherwise
    P(die during dt)    = 1 - exp(-death rate * dt)
    state               = HYPOXIC      if O < hypoxia_threshold
                        = QUIESCENT    else if neighbor_count >= inhibition_threshold
                        = PROLIFERATIVE otherwise
    oxygen_factor(O)    = 1                                   for O >= hypoxia_threshold
                        = (O - death_threshold) /
                          (hypoxia_threshold - death_threshold)   in between (linear ramp)
                        = 0                                   for O <= death_threshold
    glucose_factor(G)   = same ramp between glucose_death_threshold and glucose_threshold
    P(divide during dt) = 1 - exp(-division_rate * oxygen_factor(O) * glucose_factor(G) * dt)
                                                                            unless crowded

O and G are the oxygen [mmHg] and glucose [mM] sampled at the cell. With all thresholds at 0
and O = G = 0 (no field attached) the rules reduce to the oxygen-independent lifecycle.
HYPOXIC cells may still divide at the reduced rate; crowded cells never divide, whatever
their state.

Stated simplifications, to be revisited explicitly:
    * the cell cycle is memoryless: no refractory period after division, no minimum age;
    * a cell divides at most once per step and a daughter cannot divide in the step of its
      birth, so the expected free growth per step is (1 + p) with p = 1 - exp(-lambda dt),
      i.e. an effective rate ln(1 + p)/dt that is below lambda by O(lambda dt): 0.5% at
      dt = 0.25 h and lambda = 0.0289/h;
    * cell volume is not conserved at division: the daughter gets the parent's radius, so
      growth is implicit in the division rate; the resulting overlap is relaxed by mechanics;
    * the pair is placed symmetrically about the parent's centre, conserving the centre of mass;
    * the oxygen dependence of division is a linear ramp between the two thresholds and the
      death rate is a step at death_threshold (illustrative forms).

Every random draw comes from the cell's own persistent stream (``rng_state``), so the outcome
depends only on the seed and the history of that cell, never on thread scheduling.
"""

import warp as wp

from warpbiocell.cells.model import NUM_STATES, STATE_DEAD, STATE_HYPOXIC, STATE_PROLIFERATIVE, STATE_QUIESCENT


@wp.func
def oxygen_factor(o: float, hypoxia_threshold: float, death_threshold: float) -> float:
    if o >= hypoxia_threshold:
        return 1.0
    if o <= death_threshold:
        return 0.0
    return (o - death_threshold) / (hypoxia_threshold - death_threshold)


@wp.kernel
def lifecycle_decide(
    dt: wp.float32,
    death_rate: wp.float32,
    anoxic_death_rate: wp.float32,
    division_rate: wp.float32,
    inhibition_threshold: wp.int32,
    hypoxia_threshold: wp.float32,
    death_threshold: wp.float32,
    glucose_threshold: wp.float32,
    glucose_death_threshold: wp.float32,
    necrosis_requires_glucose: wp.int32,
    cell_state: wp.array(dtype=wp.int32),
    age: wp.array(dtype=wp.float32),
    neighbor_count: wp.array(dtype=wp.int32),
    oxygen_local: wp.array(dtype=wp.float32),
    glucose_local: wp.array(dtype=wp.float32),
    rng_state: wp.array(dtype=wp.uint32),
    divide_flag: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    divide_flag[i] = 0

    if cell_state[i] == STATE_DEAD:
        return

    age[i] = age[i] + dt
    rng = rng_state[i]
    o = oxygen_local[i]
    g = glucose_local[i]

    # Death is evaluated first: a cell that dies this step neither divides nor changes state
    # otherwise. The draw is always consumed so the stream advances identically whether or not
    # the death rate is zero.
    lethal = o < death_threshold
    if necrosis_requires_glucose != 0:
        lethal = lethal and (g < glucose_death_threshold)
    rate = death_rate
    if lethal:
        rate = wp.max(death_rate, anoxic_death_rate)
    p_death = 1.0 - wp.exp(-rate * dt)
    if wp.randf(rng) < p_death:
        cell_state[i] = STATE_DEAD
        rng_state[i] = rng
        return

    crowded = neighbor_count[i] >= inhibition_threshold
    if o < hypoxia_threshold:
        cell_state[i] = STATE_HYPOXIC
    elif crowded:
        cell_state[i] = STATE_QUIESCENT
    else:
        cell_state[i] = STATE_PROLIFERATIVE

    if not crowded:
        factor = oxygen_factor(o, hypoxia_threshold, death_threshold) * oxygen_factor(g, glucose_threshold, glucose_death_threshold)
        p_divide = 1.0 - wp.exp(-division_rate * factor * dt)
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
    oxygen_local: wp.array(dtype=wp.float32),
    glucose_local: wp.array(dtype=wp.float32),
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
    cell_state[j] = cell_state[i]  # the parent was PROLIFERATIVE or HYPOXIC; the daughter shares its environment
    cell_type[j] = cell_type[i]
    age[j] = 0.0
    oxygen_local[j] = oxygen_local[i]
    glucose_local[j] = glucose_local[i]
    velocity[j] = wp.vec3(0.0, 0.0, 0.0)
    neighbor_count[j] = 0
    # rng_state[j] was initialised for every slot at population creation and is left untouched.


@wp.kernel
def count_states(cell_state: wp.array(dtype=wp.int32), counts: wp.array(dtype=wp.int32)):
    i = wp.tid()
    s = cell_state[i]
    if s >= 0 and s < NUM_STATES:
        wp.atomic_add(counts, s, 1)
