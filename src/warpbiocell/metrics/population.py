"""Population-level metrics. Each call copies a handful of numbers to the host."""

from __future__ import annotations

import numpy as np
import warp as wp

from warpbiocell.cells.model import NUM_STATES, CellState
from warpbiocell.cells.state import CellPopulation
from warpbiocell.fields.oxygen import OxygenField
from warpbiocell.kernels.cell_kernels import count_states


def state_counts(population: CellPopulation) -> dict[CellState, int]:
    """Number of cells in each state, computed on the device with integer atomics (exact)."""
    counts = wp.zeros(NUM_STATES, dtype=wp.int32, device=population.device)
    if population.count > 0:
        wp.launch(count_states, dim=population.count, inputs=[population.cell_state, counts], device=population.device)
    values = counts.numpy()
    return {state: int(values[state]) for state in CellState}


def population_summary(population: CellPopulation, oxygen: OxygenField | None = None) -> dict[str, float]:
    """Counts plus geometric and oxygen summaries (AGENTS.md "Outputs"), all in one host copy each."""
    counts = state_counts(population)
    x = population.positions_numpy()
    if x.shape[0] == 0:
        centre_distance = np.zeros(0)
    else:
        centre_distance = np.linalg.norm(x - x.mean(axis=0), axis=1)
    summary = {
        "cells": population.count,
        "living_cells": population.count - counts[CellState.DEAD],
        "dead_cells": counts[CellState.DEAD],
        "proliferative_cells": counts[CellState.PROLIFERATIVE],
        "quiescent_cells": counts[CellState.QUIESCENT],
        "hypoxic_cells": counts[CellState.HYPOXIC],
        "spheroid_radius": float(np.percentile(centre_distance, 99)) if centre_distance.size else 0.0,
        "mean_radius": float(population.radii_numpy().mean()) if population.count else 0.0,
    }
    if oxygen is not None:
        summary.update(oxygen_summary(population, oxygen))
    return summary


def oxygen_summary(population: CellPopulation, oxygen: OxygenField) -> dict[str, float]:
    """Oxygen at living cells and over the updated grid nodes [mmHg]."""
    alive = population.states_numpy() != int(CellState.DEAD)
    at_cells = population.oxygen_numpy()[alive]
    grid = oxygen.numpy()[oxygen.field.interior_mask()]
    return {
        "oxygen_cells_mean": float(at_cells.mean()) if at_cells.size else float("nan"),
        "oxygen_cells_min": float(at_cells.min()) if at_cells.size else float("nan"),
        "oxygen_grid_mean": float(grid.mean()),
        "oxygen_grid_min": float(grid.min()),
        "field_sweeps": oxygen.last_report.sweeps if oxygen.last_report else 0,
    }


def radial_profile(population: CellPopulation, n_bins: int = 10) -> dict[str, np.ndarray]:
    """Mean oxygen and state fractions in shells around the population centroid."""
    x = population.positions_numpy()
    r = np.linalg.norm(x - x.mean(axis=0), axis=1)
    o = population.oxygen_numpy()
    s = population.states_numpy()
    edges = np.linspace(0.0, r.max() + 1e-6, n_bins + 1)
    which = np.minimum(np.digitize(r, edges) - 1, n_bins - 1)
    out = {"r_outer": edges[1:], "cells": np.bincount(which, minlength=n_bins)}
    out["oxygen_mean"] = np.array([o[which == b].mean() if np.any(which == b) else np.nan for b in range(n_bins)])
    for state in CellState:
        out[f"fraction_{state.name.lower()}"] = np.array(
            [np.mean(s[which == b] == int(state)) if np.any(which == b) else np.nan for b in range(n_bins)]
        )
    return out
