"""Population-level metrics. Each call copies a handful of numbers to the host."""

from __future__ import annotations

import numpy as np
import warp as wp

from warpbiocell.cells.model import NUM_STATES, CellState
from warpbiocell.cells.state import CellPopulation
from warpbiocell.kernels.cell_kernels import count_states


def state_counts(population: CellPopulation) -> dict[CellState, int]:
    """Number of cells in each state, computed on the device with integer atomics (exact)."""
    counts = wp.zeros(NUM_STATES, dtype=wp.int32, device=population.device)
    if population.count > 0:
        wp.launch(count_states, dim=population.count, inputs=[population.cell_state, counts], device=population.device)
    values = counts.numpy()
    return {state: int(values[state]) for state in CellState}


def population_summary(population: CellPopulation) -> dict[str, float]:
    """Counts plus geometric summaries used by the growth examples and tests."""
    counts = state_counts(population)
    x = population.positions_numpy()
    if x.shape[0] == 0:
        centre_distance = np.zeros(0)
    else:
        centre_distance = np.linalg.norm(x - x.mean(axis=0), axis=1)
    return {
        "cells": population.count,
        "living_cells": population.count - counts[CellState.DEAD],
        "dead_cells": counts[CellState.DEAD],
        "proliferative_cells": counts[CellState.PROLIFERATIVE],
        "quiescent_cells": counts[CellState.QUIESCENT],
        "hypoxic_cells": counts[CellState.HYPOXIC],
        "spheroid_radius": float(np.percentile(centre_distance, 99)) if centre_distance.size else 0.0,
        "mean_radius": float(population.radii_numpy().mean()) if population.count else 0.0,
    }
