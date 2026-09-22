"""Boolean gene networks (Milestone 11): parsing, kernel evaluation against numpy, update
semantics against closed forms, clamping, fates, inheritance, and the MicroC network."""

from pathlib import Path

import numpy as np
import pytest
import warp as wp

from warpbiocell.cells.initialization import spherical_cluster
from warpbiocell.cells.lifecycle import LifecycleParams, lifecycle_step
from warpbiocell.cells.model import CellState
from warpbiocell.cells.state import CellPopulation
from warpbiocell.network import expression as ex
from warpbiocell.network.model import BooleanNetwork, NetworkError, Node, load_network, read_bnet, read_cfg
from warpbiocell.network.runtime import InputClamp, NetworkParams, NetworkRuntime

NETWORKS = Path(__file__).resolve().parents[1] / "configs" / "networks"
JAYA_BND, JAYA_CFG = NETWORKS / "microc_jaya.bnd", NETWORKS / "microc_jaya.cfg"


def _population(n, device, seed=0, capacity=None):
    pos = spherical_cluster(n, 8.0, jitter=0.1, seed=seed)
    return CellPopulation.from_numpy(pos, np.full(n, 8.0), capacity=capacity, device=device, seed=seed)


def _toy(**rates):
    """Input I, A = I, B = A & !C, C = B (a small feedback loop) with unit rates."""
    return BooleanNetwork(
        [
            Node("I"),
            Node("A", ex.parse("I"), p_on=rates.get("A", 0.0)),
            Node("B", ex.parse("A & !C"), p_on=rates.get("B", 0.0)),
            Node("C", ex.parse("B"), p_on=rates.get("C", 0.0)),
        ]
    )


# ---- expressions and readers -----------------------------------------------------------------


@pytest.mark.parametrize(
    "text, state, expected",
    [
        ("A & !B", {"A": 1, "B": 0}, True),
        ("A & !B", {"A": 1, "B": 1}, False),
        ("!(A | B) | C", {"A": 0, "B": 0, "C": 0}, True),
        ("A && B || !C", {"A": 1, "B": 1, "C": 1}, True),
        ("A AND NOT B", {"A": 1, "B": 0}, True),
        ("!!A", {"A": 1}, True),
        ("1", {}, True),
        ("A | 0", {"A": 0}, False),
    ],
)
def test_expression_parse_and_evaluate(text, state, expected):
    assert ex.evaluate(ex.parse(text), state) is expected


@pytest.mark.parametrize("text", ["A &", "(A | B", "A B", "& A", "A | )"])
def test_malformed_expressions_are_rejected(text):
    with pytest.raises(ex.ExpressionError):
        ex.parse(text)


def test_postfix_program_matches_reference_on_all_assignments():
    node = ex.parse("(A & !B) | (C & (B | !A)) & !(C & A)")
    index = {"A": 0, "B": 1, "C": 2}
    ops, args = ex.compile_postfix(node, index)
    assert ex.depth(node) <= 64
    for bits in range(8):
        state = {name: (bits >> i) & 1 for name, i in index.items()}
        stack = []
        for op, a in zip(ops, args):
            if op == ex.OP_VAR:
                stack.append(bool(state[[k for k, v in index.items() if v == a][0]]))
            elif op == ex.OP_CONST:
                stack.append(bool(a))
            elif op == ex.OP_NOT:
                stack[-1] = not stack[-1]
            else:
                b, a2 = stack.pop(), stack.pop()
                stack.append((a2 and b) if op == ex.OP_AND else (a2 or b))
        assert stack == [ex.evaluate(node, state)]


def test_bnet_reader_and_inputs(tmp_path):
    path = tmp_path / "toy.bnet"
    path.write_text("targets, factors\nI, I\nA, I\nB, A & !C\n# comment\nC, B\n")
    net = read_bnet(path)
    assert net.names == ["I", "A", "B", "C"]
    assert net.input_names == ["I"]
    assert net["I"].is_input and not net["A"].is_input
    assert net.n_words == 1


def test_unknown_reference_and_bad_format_are_errors(tmp_path):
    with pytest.raises(NetworkError):
        BooleanNetwork([Node("A", ex.parse("B"))])
    with pytest.raises(NetworkError):
        BooleanNetwork([Node("A"), Node("A")])
    with pytest.raises(NetworkError):
        load_network(tmp_path / "x.sbml")


def test_maboss_readers_on_the_microc_network():
    variables, p_on = read_cfg(JAYA_CFG)
    net = load_network(JAYA_BND, JAYA_CFG)
    assert net.n_nodes == 106 and net.n_words == 2
    assert len(net.input_names) == 25
    assert {"Oxygen_supply", "Glucose_supply", "MCT1_stimulus", "EGFR_stimulus", "DNA_damage"} <= set(net.input_names)
    assert {"Proliferation", "Apoptosis", "Growth_Arrest", "Necrosis"} <= set(net.index)
    # unit rates, genes start ON with probability 1/2, fate and input nodes OFF
    assert net["Proliferation"].rate_up == (1.0, 0.0) and net["Proliferation"].rate_down == (0.0, 1.0)
    assert net["p53"].p_on == 0.5 and net["Proliferation"].p_on == 0.0 and net["Oxygen_supply"].p_on == 0.0
    programs = net.programs()
    assert programs["updatable"].shape[0] == 106 - 25
    assert programs["length"].max() <= 64
    # the necrosis rule of the paper
    state = np.zeros(106, dtype=bool)
    assert net.targets(state)[net.index["Necrosis"]]
    state[net.index["Oxygen_supply"]] = True
    assert not net.targets(state)[net.index["Necrosis"]]


# ---- kernel evaluation -----------------------------------------------------------------------


def test_synchronous_kernel_matches_numpy_on_random_states(cpu):
    net = load_network(JAYA_BND, JAYA_CFG)
    n = 256
    pop = _population(n, cpu)
    rt = NetworkRuntime(NetworkParams(net, update="synchronous", updates_per_step=1), pop.capacity, device=cpu)
    rng = np.random.default_rng(5)
    states = rng.random((n, net.n_nodes)) < 0.5
    words = np.stack([net.pack(row) for row in states])
    rt.states.assign(np.concatenate([words, np.zeros((pop.capacity - n, net.n_words), dtype=np.uint64)]))
    rt.update(pop, dt_h=0.25)
    got = rt.states_numpy(pop)
    expected = np.stack([net.synchronous_step(row) for row in states])
    assert np.array_equal(got, expected)
    # a second sweep, from the kernel's own result
    rt.update(pop, dt_h=0.25)
    assert np.array_equal(rt.states_numpy(pop), np.stack([net.synchronous_step(row) for row in expected]))


def test_pack_unpack_roundtrip_across_words():
    net = load_network(JAYA_BND, JAYA_CFG)
    state = np.random.default_rng(1).random(net.n_nodes) < 0.5
    assert np.array_equal(net.unpack(net.pack(state)), state)


def test_initialization_uses_p_on_and_cell_streams(cpu):
    net = _toy(A=1.0, B=0.5, C=0.0)
    n = 4000
    pop = _population(n, cpu, seed=2)
    rt = NetworkRuntime(NetworkParams(net), pop.capacity, device=cpu)
    rt.initialize(pop)
    s = rt.states_numpy(pop)
    assert not s[:, 0].any() and s[:, 1].all() and not s[:, 3].any()
    assert abs(s[:, 2].mean() - 0.5) < 0.03
    # same seed -> same draws; the cells' streams advanced
    pop2 = _population(n, cpu, seed=2)
    rt2 = NetworkRuntime(NetworkParams(net), pop2.capacity, device=cpu)
    rt2.initialize(pop2)
    assert np.array_equal(rt2.states_numpy(pop2), s)
    assert not np.array_equal(pop2.rng_state.numpy()[:n], CellPopulation.from_numpy(pop2.positions_numpy(), np.full(n, 8.0), device=cpu, seed=2).rng_state.numpy()[:n])


# ---- update semantics against closed forms ----------------------------------------------------


def test_asynchronous_single_node_flip_probability(cpu):
    """One pick among m updatable nodes flips a given unstable node with probability 1/m per pick:
    after k picks P(still unflipped) = (1 - 1/m)^k."""
    net = _toy()
    n = 20000
    pop = _population(n, cpu, seed=7)
    rt = NetworkRuntime(NetworkParams(net, update="asynchronous", updates_per_step=2, inputs=(InputClamp("I", "constant", 1.0),)), pop.capacity, device=cpu)
    rt.initialize(pop)  # everything OFF, input I clamped ON -> A is unstable, B and C stable (B = A & !C with A OFF)
    rt.step(pop, dt_h=1.0)
    m = 3
    frac_a = rt.states_numpy(pop)[:, 1].mean()
    assert abs(frac_a - (1.0 - (1.0 - 1.0 / m) ** 2)) < 0.02


def test_maboss_single_node_kinetics(cpu):
    """A single node with logic 1 and rate_up r turns ON at exponential times: P(ON at t) = 1 - exp(-r t)."""
    r = 0.7
    net = BooleanNetwork([Node("X", ex.parse("1"), rate_up=(r, 0.0), rate_down=(0.0, 1.0))])
    n = 40000
    pop = _population(n, cpu, seed=11)
    rt = NetworkRuntime(NetworkParams(net, update="maboss", time_units_per_h=1.0), pop.capacity, device=cpu)
    rt.initialize(pop)
    for t in (0.5, 1.0, 2.0):
        rt.update(pop, dt_h=t if t == 0.5 else t - (0.5 if t == 1.0 else 1.0))
        frac = rt.states_numpy(pop)[:, 0].mean()
        assert abs(frac - (1.0 - np.exp(-r * t))) < 0.01, (t, frac)


def test_maboss_and_asynchronous_agree_on_stationary_fractions(cpu):
    """With unit rates the CTMC is the uniform asynchronous update: both give the same
    long-run fate fractions on the MicroC network."""
    net = load_network(JAYA_BND, JAYA_CFG)
    clamps = (InputClamp("Oxygen_supply", "constant", 1.0), InputClamp("Glucose_supply", "constant", 1.0), InputClamp("EGFR_stimulus", "constant", 1.0), InputClamp("cMET_stimulus", "constant", 1.0))
    fractions = {}
    for mode, kw in (("asynchronous", {"updates_per_step": 81 * 10}), ("maboss", {"time_units_per_h": 10.0})):
        pop = _population(3000, cpu, seed=4)
        rt = NetworkRuntime(NetworkParams(net, update=mode, inputs=clamps, **kw), pop.capacity, device=cpu)
        rt.initialize(pop)
        for _ in range(6):
            rt.step(pop, dt_h=1.0)
        counts = rt.fate_counts(pop)
        fractions[mode] = np.array([counts[k] for k in ("proliferation", "apoptosis", "growth_arrest", "necrosis")]) / pop.count
    assert np.allclose(fractions["asynchronous"], fractions["maboss"], atol=0.04), fractions
    assert fractions["maboss"][3] == 0.0  # oxygen and glucose supplied: no necrosis


def test_synchronous_toy_reaches_its_cycle(cpu):
    """I ON: A -> 1, then B = A & !C and C = B oscillate (period-4 cycle under synchronous update)."""
    net = _toy()
    pop = _population(8, cpu)
    rt = NetworkRuntime(NetworkParams(net, update="synchronous", updates_per_step=1, inputs=(InputClamp("I", "constant", 1.0),)), pop.capacity, device=cpu)
    rt.initialize(pop)
    seen = []
    for _ in range(8):
        rt.step(pop, dt_h=1.0)
        seen.append(tuple(rt.states_numpy(pop)[0].astype(int)))
    final = rt.states_numpy(pop)
    assert (final == final[0]).all()  # synchronous update is deterministic: every cell agrees
    assert seen[-1] == seen[-5] and seen[-1] != seen[-2]  # period 4, not a fixed point
    assert all(row[1] == 1 for row in seen)  # A follows the clamped input


# ---- clamping, fates, inheritance --------------------------------------------------------------


def test_inputs_are_clamped_from_fields_and_constants(cpu):
    net = load_network(JAYA_BND, JAYA_CFG)
    n = 100
    pop = _population(n, cpu)
    lactate = wp.zeros(pop.capacity, dtype=wp.float32, device=cpu)
    pop.extra_local["lactate"] = lactate
    o = np.zeros(pop.capacity, dtype=np.float32)
    o[: n // 2] = 30.0
    pop.oxygen_local.assign(o)
    g = np.full(pop.capacity, 5.0, dtype=np.float32)
    pop.glucose_local.assign(g)
    lac = np.zeros(pop.capacity, dtype=np.float32)
    lac[::2] = 3.0
    lactate.assign(lac)
    clamps = (
        InputClamp("Oxygen_supply", "oxygen", 15.7),
        InputClamp("Glucose_supply", "glucose", 4.0),
        InputClamp("MCT1_stimulus", "lactate", 1.5),
        InputClamp("DNA_damage", "glucose", 4.0, above=False),
        InputClamp("EGFR_stimulus", "constant", 1.0),
    )
    rt = NetworkRuntime(NetworkParams(net, inputs=clamps), pop.capacity, device=cpu, species_names=("oxygen", "glucose", "lactate"))
    rt.initialize(pop)
    rt.clamp_inputs(pop)
    s = rt.states_numpy(pop)
    idx = net.index
    assert s[: n // 2, idx["Oxygen_supply"]].all() and not s[n // 2 :, idx["Oxygen_supply"]].any()
    assert s[:, idx["Glucose_supply"]].all()
    assert np.array_equal(s[:, idx["MCT1_stimulus"]], lac[:n] > 1.5)
    assert not s[:, idx["DNA_damage"]].any()  # below-threshold clamp with glucose above it
    assert s[:, idx["EGFR_stimulus"]].all()


def test_unknown_sources_and_nodes_are_rejected(cpu):
    net = _toy()
    with pytest.raises(NetworkError):
        NetworkParams(net, inputs=(InputClamp("Z", "constant", 1.0),))
    with pytest.raises(NetworkError):
        NetworkParams(net, update="random")
    with pytest.raises(NetworkError):
        NetworkParams(net, outputs={"proliferation": "Z"})
    pop = _population(4, cpu)
    with pytest.raises(NetworkError):
        NetworkRuntime(NetworkParams(net, inputs=(InputClamp("I", "lactate", 1.0),)), pop.capacity, device=cpu, species_names=("oxygen",))


def test_fate_flags_pack_the_output_nodes(cpu):
    net = _toy()
    pop = _population(16, cpu)
    params = NetworkParams(net, outputs={"proliferation": "A", "apoptosis": "B", "growth_arrest": None, "necrosis": "C"})
    rt = NetworkRuntime(params, pop.capacity, device=cpu)
    words = np.zeros((pop.capacity, 1), dtype=np.uint64)
    words[0, 0] = 0b0010  # A
    words[1, 0] = 0b0100  # B
    words[2, 0] = 0b1000  # C
    words[3, 0] = 0b1110  # A, B, C
    rt.states.assign(words)
    rt.read_fates(pop)
    flags = pop.fate_flags_numpy()
    assert flags[:4].tolist() == [1, 2, 8, 1 | 2 | 8] and not flags[4:].any()


def test_daughters_inherit_the_parent_network(cpu):
    net = load_network(JAYA_BND, JAYA_CFG)
    n = 200
    pop = _population(n, cpu, capacity=3 * n, seed=9)
    rt = NetworkRuntime(NetworkParams(net), pop.capacity, device=cpu)
    rt.initialize(pop)
    before = rt.states_numpy(pop).copy()
    flags_before = pop.fate_flags_numpy().copy()
    params = LifecycleParams(division_rate=1.0e3, inhibition_threshold=10**6)  # everyone divides
    created = lifecycle_step(pop, params, dt=1.0)
    assert created == n
    rt.inherit(pop, n)
    after = rt.states_numpy(pop)
    assert np.array_equal(after[:n], before) and np.array_equal(after[n:], before)
    flags = pop.fate_flags_numpy()
    assert np.array_equal(flags[n:], flags_before)


# ---- lifecycle in network mode ----------------------------------------------------------------


def _flags_population(n, device, flags, seed=0):
    pop = _population(n, device, capacity=3 * n, seed=seed)
    f = np.zeros(pop.capacity, dtype=np.int32)
    f[:n] = flags
    pop.fate_flags.assign(f)
    return pop


def test_network_mode_divides_only_with_proliferation_on(cpu):
    n = 400
    flags = np.zeros(n, dtype=np.int32)
    flags[: n // 2] = 1  # Proliferation
    flags[n // 4 : n // 2] |= 4  # ... but Growth_Arrest too
    pop = _flags_population(n, cpu, flags)
    params = LifecycleParams(division_rate=1.0e3, inhibition_threshold=10**6, phenotype_model="network")
    created = lifecycle_step(pop, params, dt=1.0)
    assert created == n // 4
    states = pop.states_numpy()[:n]
    assert np.all(states[: n // 4] == int(CellState.PROLIFERATIVE))
    assert np.all(states[n // 4 :] == int(CellState.QUIESCENT))


def test_network_mode_death_rates(cpu):
    n = 30000
    flags = np.zeros(n, dtype=np.int32)
    flags[: n // 3] = 2  # Apoptosis
    flags[n // 3 : 2 * n // 3] = 8  # Necrosis
    pop = _flags_population(n, cpu, flags, seed=5)
    params = LifecycleParams(division_rate=0.0, phenotype_model="network", apoptosis_rate=0.5, necrosis_rate=1.0e3)
    lifecycle_step(pop, params, dt=1.0)
    dead = pop.states_numpy()[:n] == int(CellState.DEAD)
    assert abs(dead[: n // 3].mean() - (1.0 - np.exp(-0.5))) < 0.02
    assert dead[n // 3 : 2 * n // 3].all()
    assert not dead[2 * n // 3 :].any()


def test_rules_mode_ignores_fate_flags(cpu):
    n = 200
    pop = _flags_population(n, cpu, np.full(n, 8, dtype=np.int32))  # Necrosis everywhere
    params = LifecycleParams(division_rate=0.0, necrosis_rate=1.0e3)  # rules mode
    lifecycle_step(pop, params, dt=1.0)
    assert not (pop.states_numpy()[:n] == int(CellState.DEAD)).any()


def test_lifecycle_params_validate_network_fields():
    with pytest.raises(ValueError):
        LifecycleParams(phenotype_model="graph")
    with pytest.raises(ValueError):
        LifecycleParams(apoptosis_rate=-1.0)


# ---- the MicroC network end to end ------------------------------------------------------------


def test_microc_network_fates_follow_the_environment(cpu):
    """Necrosis only without oxygen and glucose; hypoxia with glucose keeps ATP through glycolysis."""
    net = load_network(JAYA_BND, JAYA_CFG)
    n = 3000
    pop = _population(n, cpu, seed=8)
    lactate = wp.zeros(pop.capacity, dtype=wp.float32, device=cpu)
    pop.extra_local["lactate"] = lactate
    clamps = (
        InputClamp("Oxygen_supply", "oxygen", 15.7),
        InputClamp("Glucose_supply", "glucose", 4.0),
        InputClamp("MCT1_stimulus", "lactate", 1.5),
        InputClamp("EGFR_stimulus", "constant", 1.0),
        InputClamp("cMET_stimulus", "constant", 1.0),
        InputClamp("FGFR_stimulus", "constant", 1.0),
    )
    rt = NetworkRuntime(NetworkParams(net, update="asynchronous", updates_per_step=400, inputs=clamps), pop.capacity, device=cpu, species_names=("oxygen", "glucose", "lactate"))

    def run(o2, glc):
        pop.oxygen_local.assign(np.full(pop.capacity, o2, dtype=np.float32))
        pop.glucose_local.assign(np.full(pop.capacity, glc, dtype=np.float32))
        rt.initialize(pop)
        for _ in range(20):
            rt.step(pop, dt_h=0.25)
        counts = rt.fate_counts(pop)
        idx = net.index
        s = rt.states_numpy(pop)
        return {k: v / n for k, v in counts.items()}, s[:, idx["glycoATP"]].mean(), s[:, idx["mitoATP"]].mean(), s[:, idx["ATP_Production_Rate"]].mean()

    normoxia, glyco_n, mito_n, atp_n = run(30.0, 5.0)
    hypoxia, glyco_h, mito_h, atp_h = run(5.0, 5.0)
    starved, _, _, atp_s = run(5.0, 1.0)
    assert normoxia["necrosis"] == 0.0
    assert 0.1 < normoxia["proliferation"] < 0.35 and 0.05 < normoxia["apoptosis"] < 0.35
    assert mito_h < 0.05 < mito_n and glyco_h > 0.9 and atp_h > 0.9  # Warburg switch keeps ATP
    assert abs(hypoxia["proliferation"] - normoxia["proliferation"]) < 0.08
    assert starved["necrosis"] > 0.95 and starved["proliferation"] < 0.02 and atp_s < 0.05


def test_network_dynamics_are_reproducible(cpu):
    net = load_network(JAYA_BND, JAYA_CFG)
    runs = []
    for _ in range(2):
        pop = _population(500, cpu, seed=21)
        rt = NetworkRuntime(NetworkParams(net, update="maboss", time_units_per_h=8.0, inputs=(InputClamp("EGFR_stimulus", "constant", 1.0),)), pop.capacity, device=cpu)
        rt.initialize(pop)
        for _ in range(4):
            rt.step(pop, dt_h=0.25)
        runs.append(rt.states_numpy(pop))
    assert np.array_equal(runs[0], runs[1])
