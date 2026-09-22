"""Derived quantities from a run's metrics rows (the list of dicts in RunResult.metrics)."""

from __future__ import annotations

import math


def onset(rows: list[dict], column: str, radius_column: str = "spheroid_radius_um") -> tuple[float, float]:
    """Time [h] and spheroid radius [um] of the first row where ``column`` is positive; NaN if never."""
    for row in rows:
        if float(row[column]) > 0.0:
            return float(row["time_h"]), float(row[radius_column])
    return math.nan, math.nan


def onset_summary(rows: list[dict]) -> dict[str, float]:
    t_h, r_h = onset(rows, "hypoxic_cells")
    t_n, r_n = onset(rows, "dead_cells")
    return {
        "hypoxia_onset_time_h": t_h,
        "hypoxia_onset_radius_um": r_h,
        "necrosis_onset_time_h": t_n,
        "necrosis_onset_radius_um": r_n,
    }
