"""Warp kernels for one Boolean network per cell.

State: ``states[cell, word]`` (uint64), node ``i`` at word ``i // 64``, bit ``i % 64``. Logic
programs are postfix (``network/expression.py``): the evaluator keeps a stack of booleans in
one uint64 register. Every random draw comes from the cell's own ``rng_state`` stream, so
the network dynamics are reproducible on a given device like everything else.

Update semantics (``network/model.py``): asynchronous random single-node updates, synchronous
sweeps, or a continuous-time Markov chain (Gillespie) with MaBoSS rates. Input nodes are
clamped from the environment (fields sampled at the cell, or constants) and never updated.
"""

import warp as wp

from warpbiocell.cells.model import STATE_DEAD

OP_VAR = wp.constant(0)
OP_NOT = wp.constant(1)
OP_AND = wp.constant(2)
OP_OR = wp.constant(3)
OP_CONST = wp.constant(4)

SOURCE_OXYGEN = wp.constant(0)
SOURCE_GLUCOSE = wp.constant(1)
SOURCE_EXTRA_A = wp.constant(2)
SOURCE_EXTRA_B = wp.constant(3)
SOURCE_CONSTANT = wp.constant(4)

FATE_PROLIFERATION = wp.constant(1)
FATE_APOPTOSIS = wp.constant(2)
FATE_GROWTH_ARREST = wp.constant(4)
FATE_NECROSIS = wp.constant(8)


@wp.func
def get_bit(states: wp.array2d(dtype=wp.uint64), cell: int, index: int) -> wp.uint64:
    return (states[cell, index >> 6] >> wp.uint64(index & 63)) & wp.uint64(1)


@wp.func
def set_bit(states: wp.array2d(dtype=wp.uint64), cell: int, index: int, value: wp.uint64):
    mask = wp.uint64(1) << wp.uint64(index & 63)
    word = states[cell, index >> 6]
    if value != wp.uint64(0):
        states[cell, index >> 6] = word | mask
    else:
        states[cell, index >> 6] = word & ~mask


@wp.func
def eval_program(
    ops: wp.array(dtype=wp.int32),
    args: wp.array(dtype=wp.int32),
    start: int,
    length: int,
    states: wp.array2d(dtype=wp.uint64),
    cell: int,
) -> wp.uint64:
    """Postfix evaluation with a boolean stack held in a uint64 (top = bit 0)."""
    stack = wp.uint64(0)
    for k in range(length):
        op = ops[start + k]
        a = args[start + k]
        if op == OP_VAR:
            stack = (stack << wp.uint64(1)) | get_bit(states, cell, a)
        elif op == OP_CONST:
            stack = (stack << wp.uint64(1)) | wp.uint64(a)
        elif op == OP_NOT:
            stack = stack ^ wp.uint64(1)
        elif op == OP_AND:
            b = stack & wp.uint64(1)
            stack = stack >> wp.uint64(1)
            stack = (stack & ~wp.uint64(1)) | ((stack & wp.uint64(1)) & b)
        elif op == OP_OR:
            b = stack & wp.uint64(1)
            stack = stack >> wp.uint64(1)
            stack = (stack & ~wp.uint64(1)) | ((stack & wp.uint64(1)) | b)
    return stack & wp.uint64(1)


@wp.kernel
def network_initialize(
    p_on: wp.array(dtype=wp.float32),
    rng_state: wp.array(dtype=wp.uint32),
    states: wp.array2d(dtype=wp.uint64),
):
    """Draw every node ON with probability ``p_on[node]`` from the cell's stream."""
    c = wp.tid()
    rng = rng_state[c]
    for w in range(states.shape[1]):
        states[c, w] = wp.uint64(0)
    for i in range(p_on.shape[0]):
        if wp.randf(rng) < p_on[i]:
            set_bit(states, c, i, wp.uint64(1))
    rng_state[c] = rng


@wp.kernel
def network_clamp_inputs(
    clamp_node: wp.array(dtype=wp.int32),
    clamp_source: wp.array(dtype=wp.int32),
    clamp_threshold: wp.array(dtype=wp.float32),
    clamp_above: wp.array(dtype=wp.int32),  # 1: ON when value > threshold; 0: ON when value < threshold
    oxygen_local: wp.array(dtype=wp.float32),
    glucose_local: wp.array(dtype=wp.float32),
    extra_a: wp.array(dtype=wp.float32),
    extra_b: wp.array(dtype=wp.float32),
    states: wp.array2d(dtype=wp.uint64),
):
    c = wp.tid()
    for k in range(clamp_node.shape[0]):
        source = clamp_source[k]
        value = float(0.0)
        if source == SOURCE_OXYGEN:
            value = oxygen_local[c]
        elif source == SOURCE_GLUCOSE:
            value = glucose_local[c]
        elif source == SOURCE_EXTRA_A:
            value = extra_a[c]
        elif source == SOURCE_EXTRA_B:
            value = extra_b[c]
        on = wp.uint64(0)
        if source == SOURCE_CONSTANT:
            if clamp_threshold[k] > 0.0:
                on = wp.uint64(1)
        elif clamp_above[k] != 0:
            if value > clamp_threshold[k]:
                on = wp.uint64(1)
        else:
            if value < clamp_threshold[k]:
                on = wp.uint64(1)
        set_bit(states, c, clamp_node[k], on)


@wp.kernel
def network_update_async(
    n_updates: int,
    ops: wp.array(dtype=wp.int32),
    args: wp.array(dtype=wp.int32),
    start: wp.array(dtype=wp.int32),
    length: wp.array(dtype=wp.int32),
    updatable: wp.array(dtype=wp.int32),
    rng_state: wp.array(dtype=wp.uint32),
    states: wp.array2d(dtype=wp.uint64),
):
    """``n_updates`` times: pick a non-input node uniformly at random, set it to its logic."""
    c = wp.tid()
    rng = rng_state[c]
    m = updatable.shape[0]
    for k in range(n_updates):
        pick = wp.min(int(wp.randf(rng) * float(m)), m - 1)
        node = updatable[pick]
        set_bit(states, c, node, eval_program(ops, args, start[node], length[node], states, c))
    rng_state[c] = rng


@wp.kernel
def network_update_sync(
    ops: wp.array(dtype=wp.int32),
    args: wp.array(dtype=wp.int32),
    start: wp.array(dtype=wp.int32),
    length: wp.array(dtype=wp.int32),
    updatable: wp.array(dtype=wp.int32),
    states: wp.array2d(dtype=wp.uint64),
    scratch: wp.array2d(dtype=wp.uint64),
):
    """One synchronous sweep: all non-input nodes take their logic value of the current state."""
    c = wp.tid()
    for w in range(states.shape[1]):
        scratch[c, w] = states[c, w]
    for k in range(updatable.shape[0]):
        node = updatable[k]
        set_bit(scratch, c, node, eval_program(ops, args, start[node], length[node], states, c))
    for w in range(states.shape[1]):
        states[c, w] = scratch[c, w]


@wp.kernel
def network_update_maboss(
    duration: wp.float32,  # time units to simulate
    max_events: int,
    ops: wp.array(dtype=wp.int32),
    args: wp.array(dtype=wp.int32),
    start: wp.array(dtype=wp.int32),
    length: wp.array(dtype=wp.int32),
    updatable: wp.array(dtype=wp.int32),
    rate_up_true: wp.array(dtype=wp.float32),
    rate_up_false: wp.array(dtype=wp.float32),
    rate_down_true: wp.array(dtype=wp.float32),
    rate_down_false: wp.array(dtype=wp.float32),
    rng_state: wp.array(dtype=wp.uint32),
    states: wp.array2d(dtype=wp.uint64),
):
    """Gillespie simulation of the MaBoSS continuous-time Markov chain for ``duration``."""
    c = wp.tid()
    rng = rng_state[c]
    t = float(0.0)
    m = updatable.shape[0]
    for event in range(max_events):
        total = float(0.0)
        for k in range(m):
            node = updatable[k]
            logic = eval_program(ops, args, start[node], length[node], states, c)
            if get_bit(states, c, node) == wp.uint64(0):
                if logic != wp.uint64(0):
                    total += rate_up_true[node]
                else:
                    total += rate_up_false[node]
            else:
                if logic != wp.uint64(0):
                    total += rate_down_true[node]
                else:
                    total += rate_down_false[node]
        if total <= 0.0:
            break
        u1 = wp.max(wp.randf(rng), 1.0e-12)
        tau = -wp.log(u1) / total
        if t + tau > duration:
            break
        t += tau
        target = wp.randf(rng) * total
        acc = float(0.0)
        chosen = int(-1)
        for k in range(m):
            if chosen < 0:
                node = updatable[k]
                logic = eval_program(ops, args, start[node], length[node], states, c)
                r = float(0.0)
                if get_bit(states, c, node) == wp.uint64(0):
                    if logic != wp.uint64(0):
                        r = rate_up_true[node]
                    else:
                        r = rate_up_false[node]
                else:
                    if logic != wp.uint64(0):
                        r = rate_down_true[node]
                    else:
                        r = rate_down_false[node]
                acc += r
                if acc >= target and r > 0.0:
                    chosen = node
        if chosen < 0:
            chosen = updatable[m - 1]
        set_bit(states, c, chosen, wp.uint64(1) - get_bit(states, c, chosen))
    rng_state[c] = rng


@wp.kernel
def network_read_fates(
    node_proliferation: int,
    node_apoptosis: int,
    node_growth_arrest: int,
    node_necrosis: int,
    states: wp.array2d(dtype=wp.uint64),
    fate_flags: wp.array(dtype=wp.int32),
):
    """Pack the four fate nodes (-1 = absent) into ``fate_flags`` bits."""
    c = wp.tid()
    flags = int(0)
    if node_proliferation >= 0 and get_bit(states, c, node_proliferation) != wp.uint64(0):
        flags |= FATE_PROLIFERATION
    if node_apoptosis >= 0 and get_bit(states, c, node_apoptosis) != wp.uint64(0):
        flags |= FATE_APOPTOSIS
    if node_growth_arrest >= 0 and get_bit(states, c, node_growth_arrest) != wp.uint64(0):
        flags |= FATE_GROWTH_ARREST
    if node_necrosis >= 0 and get_bit(states, c, node_necrosis) != wp.uint64(0):
        flags |= FATE_NECROSIS
    fate_flags[c] = flags


@wp.kernel
def network_inherit(
    count_before: int,
    divide_flag: wp.array(dtype=wp.int32),
    division_offset: wp.array(dtype=wp.int32),
    states: wp.array2d(dtype=wp.uint64),
):
    """Daughters (slots count_before + k, see place_daughters) copy their parent's network state."""
    i = wp.tid()
    if divide_flag[i] == 0:
        return
    j = count_before + division_offset[i] - 1
    for w in range(states.shape[1]):
        states[j, w] = states[i, w]


@wp.kernel
def network_count_on(
    nodes: wp.array(dtype=wp.int32),  # -1 entries are skipped
    living_only: int,
    cell_state: wp.array(dtype=wp.int32),
    states: wp.array2d(dtype=wp.uint64),
    counts: wp.array(dtype=wp.int32),
):
    """counts[k] += 1 for every (living) cell whose node nodes[k] is ON."""
    c = wp.tid()
    if living_only != 0 and cell_state[c] == STATE_DEAD:
        return
    for k in range(nodes.shape[0]):
        node = nodes[k]
        if node >= 0 and get_bit(states, c, node) != wp.uint64(0):
            wp.atomic_add(counts, k, 1)
