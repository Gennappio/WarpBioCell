"""Experiment: build every model object from a configuration, run it, record everything.

    experiment = Experiment(config)
    result = experiment.run(output_dir)      # or run() for an in-memory result
    result.metrics[-1]["living_cells"]
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import warp as wp

from warpbiocell.cells.initialization import spherical_cluster
from warpbiocell.cells.lifecycle import CapacityError
from warpbiocell.cells.mechanics import contact_substep, make_neighbor_grid
from warpbiocell.cells.state import CellPopulation
from warpbiocell.device import resolve_device, synchronize
from warpbiocell.fields.oxygen import OxygenField
from warpbiocell.geometry.region import TissueRegion
from warpbiocell.geometry.seeding import fill_region
from warpbiocell.geometry.shapes import Sphere
from warpbiocell.io.checkpoints import save_checkpoint
from warpbiocell.io.run_output import RunOutput
from warpbiocell.metrics.population import population_summary, radial_profile, shell_radii
from warpbiocell.simulation.config import ExperimentConfig
from warpbiocell.simulation.simulator import cell_step


@dataclass
class RunResult:
    status: str  # "completed" | "capacity_exceeded"
    metrics: list[dict] = field(default_factory=list)
    profiles: list[tuple[float, dict]] = field(default_factory=list)
    summary: dict | None = None
    output_dir: Path | None = None
    wall_s: float = 0.0

    def metric(self, name: str) -> np.ndarray:
        return np.array([row[name] for row in self.metrics], dtype=float)


class Experiment:
    def __init__(self, config: ExperimentConfig, device: str | None = None, config_path: str | Path | None = None):
        self.config = config
        self.config_path = Path(config_path) if config_path else None
        self.warnings = config.validate()
        wp.config.quiet = True
        self.device = resolve_device(device or config.simulation.device)

        cells = config.cells
        self.region = self._build_region() if config.geometry.enabled else None
        positions = self._initial_positions()
        self.population = CellPopulation.from_numpy(
            positions, np.full(positions.shape[0], cells.radius_um), capacity=cells.max_cells, device=self.device, seed=config.simulation.seed
        )
        self.contact = config.contact_params()
        self.grid = make_neighbor_grid(self.contact, self.device)
        self.lifecycle = config.lifecycle_params()
        self.stepping = config.stepping()
        fixed_outside = self.region if (self.region is not None and config.geometry.oxygen_source == "tissue_surface") else None
        self.oxygen = (
            OxygenField(config.grid_geometry(), config.oxygen_params(), config.solver_settings(), device=self.device, fixed_outside=fixed_outside)
            if config.oxygen.enabled
            else None
        )
        self.confine = self.region if (self.region is not None and config.geometry.confine) else None
        self.time_h = 0.0

    # ---- construction ----------------------------------------------------------------------

    def _build_region(self) -> TissueRegion:
        g = self.config.geometry
        geometry = self.config.grid_geometry()
        if g.shape == "mask":
            from warpbiocell.geometry.masks import load_nifti_mask, region_from_mask

            mask, voxel = load_nifti_mask(g.mask_file, g.mask_label)
            return region_from_mask(mask, voxel, geometry, mask_origin=tuple(g.mask_origin_um), device=self.device, name=Path(g.mask_file).name)
        return TissueRegion.from_shape(self.config.tissue_shape(), geometry, device=self.device)

    def _initial_positions(self) -> np.ndarray:
        cfg = self.config
        cells, g, seed = cfg.cells, cfg.geometry, cfg.simulation.seed
        if self.region is not None and g.seed_fill:
            within = Sphere(center=tuple(g.center_um), radius=g.seed_within_radius_um) if g.seed_within_radius_um else None
            return fill_region(
                self.region, cells.radius_um, spacing_factor=cells.initial_spacing_factor, jitter=cells.initial_jitter,
                seed=seed, margin=g.seed_margin_um, max_cells=cells.max_cells, within=within,
            )
        positions = spherical_cluster(cells.initial_count, cells.radius_um, spacing_factor=cells.initial_spacing_factor, jitter=cells.initial_jitter, seed=seed)
        if self.region is not None:
            positions += np.asarray(g.center_um, dtype=np.float32)
            outside = int((~self.region.contains(positions, margin=cells.radius_um)).sum())
            if outside:
                self.warnings.append(f"{outside} of the {positions.shape[0]} initial cells lie outside the tissue region")
        return positions

    # ---- observation ------------------------------------------------------------------------

    def observe(self, wall_s: float = 0.0, field_sweeps: int = 0) -> tuple[dict, dict]:
        """Metrics row (AGENTS.md "Outputs") and the radial profile it was derived from."""
        summary = population_summary(self.population, self.oxygen)
        profile = radial_profile(self.population, shell_width=self.config.output.shell_width_um)
        radii = shell_radii(profile)
        outside = 0
        outside_tissue = 0
        if self.oxygen is not None or self.region is not None:
            positions = self.population.positions_numpy()
            if self.oxygen is not None:
                outside = int((~self.oxygen.geometry.contains(positions)).sum())
            if self.region is not None:
                outside_tissue = int((~self.region.contains(positions)).sum())
        row = {
            "time_h": self.time_h,
            "cells": summary["cells"],
            "living_cells": summary["living_cells"],
            "dead_cells": summary["dead_cells"],
            "proliferative_cells": summary["proliferative_cells"],
            "quiescent_cells": summary["quiescent_cells"],
            "hypoxic_cells": summary["hypoxic_cells"],
            "mean_radius_um": summary["mean_radius"],
            "spheroid_radius_um": summary["spheroid_radius"],
            "necrotic_radius_um": radii["necrotic_radius"],
            "hypoxic_radius_um": radii["hypoxic_radius"],
            "non_proliferative_radius_um": radii["non_proliferative_radius"],
            "viable_rim_um": summary["spheroid_radius"] - radii["necrotic_radius"],
            "proliferating_rim_um": summary["spheroid_radius"] - radii["non_proliferative_radius"],
            "oxygen_cells_mean_mmHg": summary.get("oxygen_cells_mean", float("nan")),
            "oxygen_cells_min_mmHg": summary.get("oxygen_cells_min", float("nan")),
            "oxygen_grid_mean_mmHg": summary.get("oxygen_grid_mean", float("nan")),
            "oxygen_grid_min_mmHg": summary.get("oxygen_grid_min", float("nan")),
            "field_sweeps": field_sweeps,
            "cells_outside_grid": outside,
            "cells_outside_tissue": outside_tissue,
            "wall_s": wall_s,
        }
        return row, profile

    # ---- run ----------------------------------------------------------------------------------

    def run(self, output_dir: str | Path | None = None, log: Callable[[str], None] | None = None) -> RunResult:
        cfg = self.config
        out = cfg.output
        stepping = self.stepping
        every = lambda hours: max(1, int(round(hours / stepping.dt_cells)))  # noqa: E731
        metrics_every, profile_every, checkpoint_every = every(out.metrics_every_h), every(out.profile_every_h), every(out.checkpoint_every_h)

        output = None
        if output_dir is not None:
            output = RunOutput(Path(output_dir), self.config_path, cfg.to_dict(), self.device, cfg.simulation.seed)
            output.metadata["warnings"] = list(self.warnings)
            if self.region is not None:
                output.metadata["tissue"] = {
                    "name": self.region.name,
                    "volume_um3": self.region.volume(),
                    "seeded_cells": self.population.count,
                    "confined": self.confine is not None,
                    "oxygen_source": cfg.geometry.oxygen_source,
                }
            output.write_metadata()
        for message in self.warnings:
            if log:
                log(f"warning: {message}")

        result = RunResult(status="completed", output_dir=Path(output_dir) if output_dir else None)
        contact_substep(self.population, self.grid, self.contact, dt=0.0)  # neighbour counts of the initial state
        if self.oxygen is not None:
            self.oxygen.update(self.population)

        def record(wall, sweeps, metrics=True, profile=True, checkpoint=True):
            row, prof = self.observe(wall, sweeps)
            if metrics:
                result.metrics.append(row)
                if output:
                    output.write_metrics(row)
                if log:
                    log(
                        f"t={self.time_h:7.2f} h  cells={row['cells']:7d}  prolif={row['proliferative_cells']:6d}  "
                        f"hypoxic={row['hypoxic_cells']:6d}  dead={row['dead_cells']:6d}  R={row['spheroid_radius_um']:6.1f} um  "
                        f"O2min={row['oxygen_cells_min_mmHg']:6.2f}  wall={wall:7.1f} s"
                    )
                if row["cells_outside_grid"] and log:
                    log(f"warning: {row['cells_outside_grid']} cells lie outside the oxygen grid and are ignored by the field")
            if profile:
                result.profiles.append((self.time_h, prof))
                if output:
                    output.write_profile(self.time_h, prof)
            if checkpoint and output:
                save_checkpoint(output.checkpoint_path(self.time_h), self.population, self.oxygen, self.time_h)
            return row

        wall = 0.0
        t_start = time.perf_counter()
        record(0.0, 0)
        n_steps = cfg.n_steps
        status = "completed"
        sweeps = 0
        last_row = result.metrics[-1]
        for step in range(1, n_steps + 1):
            try:
                report = cell_step(self.population, self.grid, self.lifecycle, self.contact, stepping, oxygen=self.oxygen, region=self.confine)
            except CapacityError as exc:
                status = "capacity_exceeded"
                if log:
                    log(f"stopping: {exc}")
                break
            sweeps = report.field_sweeps
            self.time_h = step * stepping.dt_cells
            if not report.field_converged and log:
                log(f"warning: field solve reached max_sweeps at t={self.time_h:.2f} h (residual {report.field_residual:.2e})")
            last = step == n_steps
            due = (step % metrics_every == 0 or last, step % profile_every == 0 or last, step % checkpoint_every == 0 or last)
            if any(due):
                synchronize(self.device)
                wall = time.perf_counter() - t_start
                row = record(wall, sweeps, *due)
                if due[0]:
                    last_row = row
        synchronize(self.device)
        wall = time.perf_counter() - t_start
        if status != "completed":
            last_row = record(wall, sweeps)

        result.status = status
        result.wall_s = wall
        completed_steps = int(round(self.time_h / stepping.dt_cells))
        result.summary = {**last_row, "steps": completed_steps, "status": status}
        if output:
            if out.figures:
                try:
                    from warpbiocell.visualization.figures import make_run_figures

                    made = make_run_figures(output.directory, result, self.population, self.oxygen, self.region)
                    if log:
                        log(f"figures: {', '.join(p.name for p in made) if made else 'none'}")
                except ImportError as exc:
                    if log:
                        log(f"figures skipped: {exc}")
            output.finish(status, wall, result.summary)
        return result
