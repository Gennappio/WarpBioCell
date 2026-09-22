"""Boolean network model: nodes, logic programs, rates, initial states; MaBoSS / BoolNet readers.

Semantics implemented by the kernels (``kernels/network_kernels.py``):

* **asynchronous**: ``updates_per_step`` times, pick one non-input node uniformly at random
  and set it to its logic value (uniform random asynchronous update — with unit rates this
  is the same process MaBoSS samples, which is how MicroC's network is configured);
* **synchronous**: every non-input node takes its logic value at once, ``updates_per_step`` times;
* **maboss**: continuous-time Markov chain (Gillespie) for ``dt`` time units; a node OFF turns
  ON at ``rate_up`` and a node ON turns OFF at ``rate_down``, each rate being one of two
  values depending on its logic (MaBoSS ``@logic ? a : b``).

Input nodes (no logic, or rates identically 0) never change on their own; the simulation
clamps them from the environment every step.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from warpbiocell.network import expression as ex

MAX_WORDS = 4  # kernels hold up to 4 x 64 = 256 nodes per cell


class NetworkError(ValueError):
    pass


@dataclass
class Node:
    name: str
    logic: object | None = None  # AST; None for inputs
    # MaBoSS rates: (value when logic is true, value when logic is false)
    rate_up: tuple[float, float] = (1.0, 0.0)
    rate_down: tuple[float, float] = (0.0, 1.0)
    p_on: float = 0.0  # initial probability of being ON

    @property
    def is_input(self) -> bool:
        return self.logic is None or (max(self.rate_up) == 0.0 and max(self.rate_down) == 0.0)


@dataclass
class BooleanNetwork:
    nodes: list[Node]
    source: str = ""
    variables: dict = field(default_factory=dict)

    # ---- construction --------------------------------------------------------------------

    def __post_init__(self):
        names = [n.name for n in self.nodes]
        if len(set(names)) != len(names):
            raise NetworkError("duplicate node names")
        if len(names) > 64 * MAX_WORDS:
            raise NetworkError(f"at most {64 * MAX_WORDS} nodes are supported, got {len(names)}")
        self.index = {name: i for i, name in enumerate(names)}
        for node in self.nodes:
            if node.logic is not None:
                missing = ex.variables(node.logic) - set(self.index)
                if missing:
                    raise NetworkError(f"node {node.name!r} references unknown nodes {sorted(missing)}")
                if ex.depth(node.logic) > 64:
                    raise NetworkError(f"node {node.name!r}: expression nests deeper than 64")

    @property
    def names(self) -> list[str]:
        return [n.name for n in self.nodes]

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_words(self) -> int:
        return (self.n_nodes + 63) // 64

    @property
    def input_names(self) -> list[str]:
        return [n.name for n in self.nodes if n.is_input]

    def __getitem__(self, name: str) -> Node:
        return self.nodes[self.index[name]]

    # ---- kernel arrays ---------------------------------------------------------------------

    def programs(self) -> dict[str, np.ndarray]:
        """Flattened postfix programs and per-node metadata as numpy arrays."""
        ops, args, start, length = [], [], [], []
        for node in self.nodes:
            start.append(len(ops))
            if node.logic is None:
                length.append(0)
                continue
            o, a = ex.compile_postfix(node.logic, self.index)
            ops.extend(o)
            args.extend(a)
            length.append(len(o))
        return {
            "ops": np.array(ops or [0], dtype=np.int32),
            "args": np.array(args or [0], dtype=np.int32),
            "start": np.array(start, dtype=np.int32),
            "length": np.array(length, dtype=np.int32),
            "updatable": np.array([i for i, n in enumerate(self.nodes) if not n.is_input], dtype=np.int32),
            "rate_up_true": np.array([n.rate_up[0] for n in self.nodes], dtype=np.float32),
            "rate_up_false": np.array([n.rate_up[1] for n in self.nodes], dtype=np.float32),
            "rate_down_true": np.array([n.rate_down[0] for n in self.nodes], dtype=np.float32),
            "rate_down_false": np.array([n.rate_down[1] for n in self.nodes], dtype=np.float32),
            "p_on": np.array([n.p_on for n in self.nodes], dtype=np.float32),
        }

    # ---- numpy reference ---------------------------------------------------------------------

    def targets(self, state: np.ndarray) -> np.ndarray:
        """Logic value of every node for a boolean state vector (inputs keep their state)."""
        mapping = {name: bool(state[i]) for name, i in self.index.items()}
        out = np.array(state, dtype=bool).copy()
        for i, node in enumerate(self.nodes):
            if not node.is_input:
                out[i] = ex.evaluate(node.logic, mapping)
        return out

    def synchronous_step(self, state: np.ndarray) -> np.ndarray:
        return self.targets(state)

    # ---- bit packing (matches the kernels' layout: node i -> word i // 64, bit i % 64) ------

    def pack(self, state: np.ndarray) -> np.ndarray:
        words = np.zeros(self.n_words, dtype=np.uint64)
        for i, bit in enumerate(np.asarray(state, dtype=bool)):
            if bit:
                words[i // 64] |= np.uint64(1) << np.uint64(i % 64)
        return words

    def unpack(self, words: np.ndarray) -> np.ndarray:
        words = np.asarray(words, dtype=np.uint64)
        return np.array([(int(words[i // 64]) >> (i % 64)) & 1 for i in range(self.n_nodes)], dtype=bool)


# ---- readers ------------------------------------------------------------------------------------

_COMMENT = re.compile(r"//[^\n]*|/\*.*?\*/", re.DOTALL)


def _strip_comments(text: str) -> str:
    return _COMMENT.sub("", text)


def read_cfg(path: str | Path) -> tuple[dict[str, float], dict[str, float]]:
    """MaBoSS .cfg: ``$var = value;`` and ``.istate`` lines -> (variables, p_on per node)."""
    variables: dict[str, float] = {}
    p_on: dict[str, float] = {}
    text = _strip_comments(Path(path).read_text())
    for statement in text.split(";"):
        s = statement.strip()
        if not s:
            continue
        m = re.match(r"^\$(\w+)\s*=\s*([-+0-9.eE]+)$", s)
        if m:
            variables[m.group(1)] = float(m.group(2))
            continue
        m = re.match(r"^\[?\s*(\w+)\s*\]?\.istate\s*=\s*(.+)$", s, re.DOTALL)
        if m:
            name, rhs = m.group(1), m.group(2).strip()
            if re.fullmatch(r"[01]", rhs):
                p_on[name] = float(rhs)
            else:  # "p0 [0], p1 [1]"
                probs = {int(v): float(p) for p, v in re.findall(r"([-+0-9.eE]+)\s*\[\s*([01])\s*\]", rhs)}
                total = sum(probs.values()) or 1.0
                p_on[name] = probs.get(1, 0.0) / total
            continue
        # max_time, sample_count, discrete_time, ... are simulation settings, not model.
    return variables, p_on


def _rate(text: str, variables: dict[str, float], has_logic: bool) -> tuple[float, float]:
    """``@logic ? a : b`` -> (a, b); a plain number or $var -> (v, v)."""
    text = text.strip()
    m = re.match(r"^@logic\s*\?\s*(\S+)\s*:\s*(\S+)$", text)
    if m:
        if not has_logic:
            raise NetworkError(f"rate {text!r} refers to @logic on a node without logic")
        return _number(m.group(1), variables), _number(m.group(2), variables)
    v = _number(text, variables)
    return v, v


def _number(token: str, variables: dict[str, float]) -> float:
    token = token.strip()
    if token.startswith("$"):
        if token[1:] not in variables:
            raise NetworkError(f"undefined variable {token} (is the .cfg file missing?)")
        return float(variables[token[1:]])
    try:
        return float(token)
    except ValueError as exc:
        raise NetworkError(f"cannot read rate {token!r}") from exc


def read_bnd(path: str | Path, cfg: str | Path | None = None) -> BooleanNetwork:
    """MaBoSS .bnd with ``node NAME { logic = ...; rate_up = ...; rate_down = ...; }`` blocks."""
    text = _strip_comments(Path(path).read_text())
    variables, p_on = read_cfg(cfg) if cfg else ({}, {})
    nodes = []
    for m in re.finditer(r"node\s+(\w+)\s*\{(.*?)\}", text, re.DOTALL):
        name, body = m.group(1), m.group(2)
        fields = {}
        for statement in body.split(";"):
            s = statement.strip()
            if s:
                key, _, value = s.partition("=")
                fields[key.strip()] = value.strip()
        logic = ex.parse(fields["logic"]) if "logic" in fields else None
        rate_up = _rate(fields.get("rate_up", "@logic ? 1 : 0" if logic is not None else "0"), variables, logic is not None)
        rate_down = _rate(fields.get("rate_down", "@logic ? 0 : 1" if logic is not None else "0"), variables, logic is not None)
        nodes.append(Node(name=name, logic=logic, rate_up=rate_up, rate_down=rate_down, p_on=p_on.get(name, 0.0)))
    if not nodes:
        raise NetworkError(f"no node blocks found in {path}")
    return BooleanNetwork(nodes, source=str(path), variables=variables)


def read_bnet(path: str | Path) -> BooleanNetwork:
    """BoolNet .bnet: ``targets, factors`` lines; ``A, A`` (or a missing rule) marks an input."""
    nodes: dict[str, Node] = {}
    for raw in Path(path).read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.lower().replace(" ", "").startswith("targets,factors"):
            continue
        name, _, rule = line.partition(",")
        name, rule = name.strip(), rule.strip()
        if not name or not rule:
            raise NetworkError(f"cannot read line {raw!r}")
        logic = None if rule == name else ex.parse(rule)
        nodes[name] = Node(name=name, logic=logic, rate_up=(1.0, 0.0) if logic is not None else (0.0, 0.0), rate_down=(0.0, 1.0) if logic is not None else (0.0, 0.0))
    if not nodes:
        raise NetworkError(f"no rules found in {path}")
    return BooleanNetwork(list(nodes.values()), source=str(path))


def load_network(path: str | Path, cfg: str | Path | None = None) -> BooleanNetwork:
    path = Path(path)
    if path.suffix == ".bnet":
        return read_bnet(path)
    if path.suffix == ".bnd":
        return read_bnd(path, cfg)
    raise NetworkError(f"unknown network format {path.suffix!r} (use .bnd or .bnet)")
