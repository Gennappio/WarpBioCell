"""Boolean expressions: parse ``A & !(B | C)`` into a postfix program the kernels interpret.

Grammar (precedence low -> high): ``|`` (or), ``&`` (and), ``!`` (not), atoms (identifier,
``0``/``1``/``true``/``false``, parenthesised expression). MaBoSS also accepts ``||``, ``&&``,
``AND``/``OR``/``NOT``; all are handled.

Program encoding, shared with ``kernels/network_kernels.py``: two int32 arrays ``ops`` and
``args``. ``OP_VAR a`` pushes node ``a``'s state, ``OP_CONST a`` pushes ``a`` (0/1), ``OP_NOT``
negates the top, ``OP_AND`` / ``OP_OR`` combine the top two. The evaluator keeps its stack
of booleans in one uint64, so a program may nest at most 64 deep.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

OP_VAR, OP_NOT, OP_AND, OP_OR, OP_CONST = 0, 1, 2, 3, 4

_TOKEN = re.compile(r"\s*(?:(\|\||&&|[()&|!])|(AND|OR|NOT)\b|([A-Za-z_][A-Za-z0-9_]*)|(\d+))", re.IGNORECASE)


class ExpressionError(ValueError):
    pass


@dataclass(frozen=True)
class Var:
    name: str


@dataclass(frozen=True)
class Const:
    value: bool


@dataclass(frozen=True)
class Not:
    child: object


@dataclass(frozen=True)
class And:
    left: object
    right: object


@dataclass(frozen=True)
class Or:
    left: object
    right: object


def tokenize(text: str) -> list[str]:
    tokens = []
    pos = 0
    text = text.strip()
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        if not m or m.end() == pos:
            raise ExpressionError(f"cannot tokenize {text[pos:pos + 20]!r} in {text!r}")
        sym, word, ident, number = m.groups()
        if sym:
            tokens.append({"||": "|", "&&": "&"}.get(sym, sym))
        elif word:
            tokens.append({"and": "&", "or": "|", "not": "!"}[word.lower()])
        elif ident:
            tokens.append(ident)
        else:
            tokens.append(number)
        pos = m.end()
    return tokens


def parse(text: str):
    """Expression text -> AST."""
    tokens = tokenize(text)
    pos = 0

    def peek():
        return tokens[pos] if pos < len(tokens) else None

    def take(expected=None):
        nonlocal pos
        tok = peek()
        if tok is None or (expected is not None and tok != expected):
            raise ExpressionError(f"expected {expected!r}, got {tok!r} in {text!r}")
        pos += 1
        return tok

    def parse_or():
        node = parse_and()
        while peek() == "|":
            take()
            node = Or(node, parse_and())
        return node

    def parse_and():
        node = parse_not()
        while peek() == "&":
            take()
            node = And(node, parse_not())
        return node

    def parse_not():
        if peek() == "!":
            take()
            return Not(parse_not())
        return parse_atom()

    def parse_atom():
        tok = take()
        if tok == "(":
            node = parse_or()
            take(")")
            return node
        if tok in ("0", "1"):
            return Const(tok == "1")
        if tok.lower() in ("true", "false"):
            return Const(tok.lower() == "true")
        if tok.isdigit():
            raise ExpressionError(f"unexpected number {tok!r} in {text!r}")
        if tok in ("|", "&", "!", ")"):
            raise ExpressionError(f"unexpected {tok!r} in {text!r}")
        return Var(tok)

    node = parse_or()
    if peek() is not None:
        raise ExpressionError(f"trailing tokens {tokens[pos:]} in {text!r}")
    return node


def variables(node) -> set[str]:
    if isinstance(node, Var):
        return {node.name}
    if isinstance(node, Const):
        return set()
    if isinstance(node, Not):
        return variables(node.child)
    return variables(node.left) | variables(node.right)


def evaluate(node, state: dict) -> bool:
    """Reference evaluation with a name -> bool mapping."""
    if isinstance(node, Var):
        return bool(state[node.name])
    if isinstance(node, Const):
        return node.value
    if isinstance(node, Not):
        return not evaluate(node.child, state)
    if isinstance(node, And):
        return evaluate(node.left, state) and evaluate(node.right, state)
    return evaluate(node.left, state) or evaluate(node.right, state)


def compile_postfix(node, index: dict) -> tuple[list[int], list[int]]:
    """AST -> (ops, args) with node names resolved through ``index``."""
    ops, args = [], []

    def emit(op, arg=0):
        ops.append(op)
        args.append(arg)

    def walk(n):
        if isinstance(n, Var):
            if n.name not in index:
                raise ExpressionError(f"unknown node {n.name!r}")
            emit(OP_VAR, index[n.name])
        elif isinstance(n, Const):
            emit(OP_CONST, 1 if n.value else 0)
        elif isinstance(n, Not):
            walk(n.child)
            emit(OP_NOT)
        elif isinstance(n, And):
            walk(n.left)
            walk(n.right)
            emit(OP_AND)
        else:
            walk(n.left)
            walk(n.right)
            emit(OP_OR)

    walk(node)
    return ops, args


def depth(node) -> int:
    """Stack depth the postfix evaluation needs (must stay <= 64)."""
    if isinstance(node, (Var, Const)):
        return 1
    if isinstance(node, Not):
        return depth(node.child)
    return max(depth(node.left), depth(node.right) + 1)
