"""Per-cell Boolean networks on the device: parameters, state arrays and the per-step update.

Order inside a cell step (simulation/simulator.py): after the fields are sampled,
``clamp_inputs`` sets the input nodes from the environment, ``update`` advances every cell's
network, ``read_fates`` packs the fate nodes into ``population.fate_flags`` for the lifecycle
(``phenotype_model = network``). After division, ``inherit`` copies the parent's network to
the daughter.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import warp as wp

from warpbiocell.cells.state import CellPopulation
from warpbiocell.kernels.network_kernels import (
    FATE_APOPTOSIS,
    FATE_GROWTH_ARREST,
    FATE_NECROSIS,
    FATE_PROLIFERATION,
    network_clamp_inputs,
    network_count_on,
    network_inherit,
    network_initialize,
    network_read_fates,
    network_update_async,
    network_update_maboss,
    network_update_sync,
)
from warpbiocell.network.model import BooleanNetwork, NetworkError

SOURCES = {"oxygen": 0, "glucose": 1, "constant": 4}  # other species: extra_a (2), extra_b (3)
UPDATE_MODES = ("asynchronous", "synchronous", "maboss")
FATE_BITS = {"proliferation": FATE_PROLIFERATION, "apoptosis": FATE_APOPTOSIS, "growth_arrest": FATE_GROWTH_ARREST, "necrosis": FATE_NECROSIS}
FATE_ROLES = tuple(FATE_BITS)
DEFAULT_OUTPUT_NODES = {"proliferation": "Proliferation", "apoptosis": "Apoptosis", "growth_arrest": "Growth_Arrest", "necrosis": "Necrosis"}


@dataclass(frozen=True)
class InputClamp:
    """Input node driven by the environment: ``source`` is oxygen, glucose, a species name, or
    constant; ON when the value is above (or, with ``above=False``, below) ``threshold``.
    For ``constant`` the node is ON when ``threshold > 0``."""

    node: str
    source: str
    threshold: float = 0.0
    above: bool = True


@dataclass(frozen=True)
class NetworkParams:
    network: BooleanNetwork
    update: str = "asynchronous"
    updates_per_step: int = 100  # asynchronous: single-node updates; synchronous: sweeps
    time_units_per_h: float = 10.0  # maboss: network time units simulated per simulated hour
    max_events_per_step: int = 20000  # maboss: safety cap on Gillespie events per cell per step
    inputs: tuple = ()  # InputClamp...
    outputs: dict | None = None  # fate role -> node name; None: the conventional names that exist in the network
    initial_states: dict = field(default_factory=dict)  # node -> p_on override

    def __post_init__(self):
        if self.update not in UPDATE_MODES:
            raise NetworkError(f"update must be one of {UPDATE_MODES}, got {self.update!r}")
        if self.updates_per_step < 1 or self.time_units_per_h <= 0.0 or self.max_events_per_step < 1:
            raise NetworkError("updates_per_step, time_units_per_h and max_events_per_step must be positive")
        names = set(self.network.index)
        for clamp in self.inputs:
            if clamp.node not in names:
                raise NetworkError(f"input clamp refers to unknown node {clamp.node!r}")
        if self.outputs is None:
            object.__setattr__(self, "outputs", {role: (node if node in names else None) for role, node in DEFAULT_OUTPUT_NODES.items()})
        for role, name in self.outputs.items():
            if role not in FATE_BITS:
                raise NetworkError(f"unknown output role {role!r}; use {sorted(FATE_BITS)}")
            if name is not None and name not in names:
                raise NetworkError(f"output {role!r} refers to unknown node {name!r}")
        for name in self.initial_states:
            if name not in names:
                raise NetworkError(f"initial_states refers to unknown node {name!r}")

    def fate_node_indices(self) -> tuple[int, int, int, int]:
        idx = self.network.index
        return tuple(idx[self.outputs[role]] if self.outputs.get(role) else -1 for role in FATE_ROLES)


class NetworkRuntime:
    def __init__(self, params: NetworkParams, capacity: int, device: wp.context.Device | str | None = None, species_names: tuple[str, ...] = ()):
        self.params = params
        self.network = params.network
        self.device = wp.get_device(device)
        self.capacity = capacity
        arrays = self.network.programs()
        p_on = arrays["p_on"].copy()
        for name, p in params.initial_states.items():
            p_on[self.network.index[name]] = float(p)
        self._arrays = {k: wp.array(v, dtype=wp.float32 if v.dtype == np.float32 else wp.int32, device=self.device) for k, v in arrays.items()}
        self._arrays["p_on"] = wp.array(p_on, dtype=wp.float32, device=self.device)
        self.states = wp.zeros((capacity, self.network.n_words), dtype=wp.uint64, device=self.device)
        self._scratch = wp.zeros((capacity, self.network.n_words), dtype=wp.uint64, device=self.device) if params.update == "synchronous" else None
        self._fate_query = wp.array(np.array(self.fate_nodes, dtype=np.int32), dtype=wp.int32, device=self.device)
        self._fate_counts = wp.zeros(4, dtype=wp.int32, device=self.device)
        self.species_names = tuple(species_names)
        self._build_clamps()

    # ---- input mapping -----------------------------------------------------------------------

    def _build_clamps(self) -> None:
        extra = [n for n in self.species_names if n not in ("oxygen", "glucose")]
        nodes, sources, thresholds, above = [], [], [], []
        for clamp in self.params.inputs:
            if clamp.source in SOURCES:
                source = SOURCES[clamp.source]
            elif clamp.source in extra[:2]:
                source = 2 + extra.index(clamp.source)
            else:
                raise NetworkError(f"input clamp {clamp.node!r}: source {clamp.source!r} is not oxygen, glucose, constant or one of the first two extra species {extra[:2]}")
            nodes.append(self.network.index[clamp.node])
            sources.append(source)
            thresholds.append(float(clamp.threshold))
            above.append(1 if clamp.above else 0)
        self.extra_names = extra[:2]
        self._clamp = {
            "node": wp.array(np.array(nodes or [0], dtype=np.int32), dtype=wp.int32, device=self.device),
            "source": wp.array(np.array(sources or [4], dtype=np.int32), dtype=wp.int32, device=self.device),
            "threshold": wp.array(np.array(thresholds or [0.0], dtype=np.float32), dtype=wp.float32, device=self.device),
            "above": wp.array(np.array(above or [1], dtype=np.int32), dtype=wp.int32, device=self.device),
        }
        self._n_clamps = len(nodes)

    # ---- per-step operations ---------------------------------------------------------------

    def initialize(self, population: CellPopulation, count: int | None = None) -> None:
        """Random initial states for slots [0, count) from the cells' RNG streams."""
        n = population.count if count is None else count
        if n > 0:
            wp.launch(network_initialize, dim=n, inputs=[self._arrays["p_on"], population.rng_state, self.states], device=self.device)
            self.read_fates(population)

    def clamp_inputs(self, population: CellPopulation) -> None:
        n = population.count
        if n == 0 or self._n_clamps == 0:
            return
        extra_a = population.local_field(self.extra_names[0]) if len(self.extra_names) > 0 else population.oxygen_local
        extra_b = population.local_field(self.extra_names[1]) if len(self.extra_names) > 1 else population.oxygen_local
        wp.launch(
            network_clamp_inputs,
            dim=n,
            inputs=[self._clamp["node"], self._clamp["source"], self._clamp["threshold"], self._clamp["above"], population.oxygen_local, population.glucose_local, extra_a, extra_b, self.states],
            device=self.device,
        )

    def update(self, population: CellPopulation, dt_h: float) -> None:
        n = population.count
        if n == 0:
            return
        a = self._arrays
        common = [a["ops"], a["args"], a["start"], a["length"], a["updatable"]]
        if self.params.update == "asynchronous":
            wp.launch(network_update_async, dim=n, inputs=[self.params.updates_per_step, *common, population.rng_state, self.states], device=self.device)
        elif self.params.update == "synchronous":
            for _ in range(self.params.updates_per_step):
                wp.launch(network_update_sync, dim=n, inputs=[*common, self.states, self._scratch], device=self.device)
        else:
            wp.launch(
                network_update_maboss,
                dim=n,
                inputs=[float(dt_h * self.params.time_units_per_h), self.params.max_events_per_step, *common, a["rate_up_true"], a["rate_up_false"], a["rate_down_true"], a["rate_down_false"], population.rng_state, self.states],
                device=self.device,
            )

    def read_fates(self, population: CellPopulation) -> None:
        n = population.count
        if n == 0:
            return
        wp.launch(network_read_fates, dim=n, inputs=[*self.fate_nodes, self.states, population.fate_flags], device=self.device)

    @property
    def fate_nodes(self) -> tuple[int, int, int, int]:
        return self.params.fate_node_indices()

    def inherit(self, population: CellPopulation, count_before: int) -> None:
        """After a lifecycle step that created daughters: copy parents' networks to them
        (place_daughters already copied the fate flags)."""
        if count_before == 0 or population.count == count_before:
            return
        wp.launch(network_inherit, dim=count_before, inputs=[count_before, population.divide_flag, population.division_offset, self.states], device=self.device)

    def step(self, population: CellPopulation, dt_h: float) -> None:
        """clamp -> update -> fates, the per-cell-step sequence."""
        self.clamp_inputs(population)
        self.update(population, dt_h)
        self.read_fates(population)

    # ---- inspection --------------------------------------------------------------------------

    def states_numpy(self, population: CellPopulation) -> np.ndarray:
        """Boolean matrix (count, n_nodes)."""
        words = self.states.numpy()[: population.count]
        out = np.zeros((population.count, self.network.n_nodes), dtype=bool)
        for i in range(self.network.n_nodes):
            out[:, i] = (words[:, i // 64] >> np.uint64(i % 64)) & np.uint64(1)
        return out

    def count_on(self, population: CellPopulation, names: list[str], living_only: bool = True) -> np.ndarray:
        """Number of (living) cells with each named node ON; one host copy."""
        nodes = wp.array(np.array([self.network.index[n] for n in names], dtype=np.int32), dtype=wp.int32, device=self.device)
        counts = wp.zeros(len(names), dtype=wp.int32, device=self.device)
        if population.count > 0:
            wp.launch(network_count_on, dim=population.count, inputs=[nodes, 1 if living_only else 0, population.cell_state, self.states, counts], device=self.device)
        return counts.numpy()

    def fate_counts(self, population: CellPopulation) -> dict[str, int]:
        """Living cells with each fate node ON: proliferation, apoptosis, growth_arrest, necrosis."""
        self._fate_counts.zero_()
        if population.count > 0:
            wp.launch(network_count_on, dim=population.count, inputs=[self._fate_query, 1, population.cell_state, self.states, self._fate_counts], device=self.device)
        values = self._fate_counts.numpy()
        return {role: int(values[k]) for k, role in enumerate(FATE_ROLES)}
