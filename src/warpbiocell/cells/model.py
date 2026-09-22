"""Cell states and types shared by Python code and Warp kernels.

``cell_state`` is the single source of truth for a cell's condition (no separate ``alive``
flag). States are mutually exclusive and evaluated with the precedence
DEAD > HYPOXIC > QUIESCENT > PROLIFERATIVE. HYPOXIC is defined here but no rule sets it until
the oxygen milestone.
"""

from enum import IntEnum

import warp as wp


class CellState(IntEnum):
    PROLIFERATIVE = 0  # alive, adequately oxygenated, free to divide
    QUIESCENT = 1  # alive, adequately oxygenated, contact-inhibited
    HYPOXIC = 2  # alive, local oxygen below hypoxia threshold (Milestone 4)
    DEAD = 3  # irreversible; still occupies space


class CellType(IntEnum):
    TUMOR = 0  # single phenotype in the MVP


# Kernel-side constants (wp.constant values are inlined at compile time).
STATE_PROLIFERATIVE = wp.constant(int(CellState.PROLIFERATIVE))
STATE_QUIESCENT = wp.constant(int(CellState.QUIESCENT))
STATE_HYPOXIC = wp.constant(int(CellState.HYPOXIC))
STATE_DEAD = wp.constant(int(CellState.DEAD))
NUM_STATES = len(CellState)
