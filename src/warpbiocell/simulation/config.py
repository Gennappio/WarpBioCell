"""Experiment configuration: one YAML file -> validated dataclasses -> model parameter objects.

Every key carries its unit in its name. Unknown keys are errors (a typo must not silently
fall back to a default). Provenance labels (illustrative / estimated / literature-derived /
fitted) live as comments in the YAML next to each value; the runner copies the file verbatim
into the run directory so they travel with the results.
"""

from __future__ import annotations

import dataclasses
import typing
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from warpbiocell.cells.lifecycle import LifecycleParams
from warpbiocell.cells.mechanics import ContactParams
from warpbiocell.fields.diffusion import SolverSettings
from warpbiocell.fields.oxygen import OxygenParams
from warpbiocell.fields.scalar_field import GridGeometry
from warpbiocell.simulation.simulator import TimeStepping


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class SimulationConfig:
    duration_h: float = 120.0
    dt_cells_h: float = 0.25
    dt_mechanics_h: float = 0.025
    seed: int = 0
    device: str = "auto"


@dataclass(frozen=True)
class CellsConfig:
    initial_count: int = 1000
    max_cells: int = 100_000
    radius_um: float = 8.0
    initial_spacing_factor: float = 1.0  # lattice spacing / cell diameter of the initial cluster
    initial_jitter: float = 0.1  # uniform jitter amplitude / cell radius


@dataclass(frozen=True)
class MechanicsConfig:
    stiffness: float = 10.0  # k [force/um]
    damping: float = 1.0  # gamma [force h/um]; only k/gamma [1/h] matters
    margin_um: float = 2.0  # neighbour query radius = 2 r + margin


@dataclass(frozen=True)
class LifecycleConfig:
    division_rate_per_h: float = 0.0289  # ln(2)/24 h
    death_rate_per_h: float = 0.0
    anoxic_death_rate_per_h: float = 0.5
    inhibition_threshold: int = 8  # neighbours within the query radius
    hypoxia_threshold_mmHg: float = 8.0
    death_threshold_mmHg: float = 2.0
    placement_factor: float = 1.0


@dataclass(frozen=True)
class GridConfig:
    box_um: float = 800.0  # cubic domain edge, centred on the origin
    dx_um: float = 20.0


@dataclass(frozen=True)
class SolverConfig:
    omega: float = 1.8
    tolerance: float = 1.0e-5
    max_sweeps: int = 2000
    check_every: int = 10


@dataclass(frozen=True)
class OxygenConfig:
    enabled: bool = True
    diffusion_um2_per_h: float = 7.2e6
    uptake_max_mmHg_um3_per_h: float = 1.4e8
    michaelis_k_mmHg: float = 3.4
    boundary_mmHg: float = 38.0
    grid: GridConfig = field(default_factory=GridConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)


@dataclass(frozen=True)
class OutputConfig:
    metrics_every_h: float = 1.0
    profile_every_h: float = 12.0
    checkpoint_every_h: float = 24.0
    shell_width_um: float = 20.0  # radial profile resolution
    figures: bool = True


@dataclass(frozen=True)
class ExperimentConfig:
    name: str = "tumor_spheroid"
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    cells: CellsConfig = field(default_factory=CellsConfig)
    mechanics: MechanicsConfig = field(default_factory=MechanicsConfig)
    lifecycle: LifecycleConfig = field(default_factory=LifecycleConfig)
    oxygen: OxygenConfig = field(default_factory=OxygenConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    # ---- model objects -------------------------------------------------------------------

    def stepping(self) -> TimeStepping:
        return TimeStepping(dt_cells=self.simulation.dt_cells_h, dt_mechanics=self.simulation.dt_mechanics_h)

    def contact_params(self) -> ContactParams:
        m = self.mechanics
        return ContactParams(stiffness=m.stiffness, damping=m.damping, max_radius=self.cells.radius_um, margin=m.margin_um)

    def lifecycle_params(self) -> LifecycleParams:
        lc = self.lifecycle
        return LifecycleParams(
            division_rate=lc.division_rate_per_h,
            death_rate=lc.death_rate_per_h,
            anoxic_death_rate=lc.anoxic_death_rate_per_h,
            inhibition_threshold=lc.inhibition_threshold,
            hypoxia_threshold=lc.hypoxia_threshold_mmHg if self.oxygen.enabled else 0.0,
            death_threshold=lc.death_threshold_mmHg if self.oxygen.enabled else 0.0,
            placement_factor=lc.placement_factor,
        )

    def oxygen_params(self) -> OxygenParams:
        o = self.oxygen
        return OxygenParams(
            diffusion_coefficient=o.diffusion_um2_per_h,
            uptake_max=o.uptake_max_mmHg_um3_per_h,
            michaelis_k=o.michaelis_k_mmHg,
            boundary_value=o.boundary_mmHg,
        )

    def solver_settings(self) -> SolverSettings:
        s = self.oxygen.solver
        return SolverSettings(omega=s.omega, tolerance=s.tolerance, max_sweeps=s.max_sweeps, check_every=s.check_every)

    def grid_geometry(self) -> GridGeometry:
        return GridGeometry.centered_cube(self.oxygen.grid.box_um, self.oxygen.grid.dx_um)

    @property
    def n_steps(self) -> int:
        return int(round(self.simulation.duration_h / self.simulation.dt_cells_h))

    # ---- validation -------------------------------------------------------------------------

    def validate(self) -> list[str]:
        """Raise ConfigError on inconsistencies; return a list of warnings."""
        warnings: list[str] = []
        sim, cells = self.simulation, self.cells
        if sim.duration_h <= 0.0:
            raise ConfigError("simulation.duration_h must be positive")
        if cells.initial_count < 1 or cells.max_cells < cells.initial_count:
            raise ConfigError("cells.initial_count must be >= 1 and <= cells.max_cells")
        if cells.radius_um <= 0.0:
            raise ConfigError("cells.radius_um must be positive")
        if self.output.metrics_every_h <= 0.0 or self.output.shell_width_um <= 0.0:
            raise ConfigError("output intervals and shell width must be positive")
        # Parameter objects validate their own ranges.
        stepping = self.stepping()
        self.contact_params().check_substep(stepping.dt_mechanics)
        self.lifecycle_params()
        if self.oxygen.enabled:
            self.oxygen_params()
            self.solver_settings()
            geom = self.grid_geometry()
            # Initial cluster must fit with at least one voxel of margin on every side.
            cluster_radius = cells.radius_um * cells.initial_spacing_factor * (cells.initial_count / 0.52) ** (1.0 / 3.0)
            half = 0.5 * self.oxygen.grid.box_um
            if cluster_radius + geom.dx > half:
                raise ConfigError(
                    f"initial cluster radius ~{cluster_radius:.0f} um does not fit in the {self.oxygen.grid.box_um:.0f} um box with a one-voxel margin"
                )
            if self.lifecycle.hypoxia_threshold_mmHg >= self.oxygen.boundary_mmHg:
                warnings.append("hypoxia threshold is at or above the boundary oxygen: every cell will be hypoxic")
        free_growth = cells.initial_count * np.exp(self.lifecycle.division_rate_per_h * sim.duration_h)
        if free_growth > cells.max_cells:
            warnings.append(
                f"free exponential growth would reach {free_growth:.0f} cells > max_cells={cells.max_cells}; "
                "the run stops with status 'capacity_exceeded' if inhibition and death do not limit it"
            )
        return warnings

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


# ---- loading ----------------------------------------------------------------------------------


def _build(cls, data, path: str):
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected a mapping, got {type(data).__name__}")
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(data) - names)
    if unknown:
        raise ConfigError(f"{path}: unknown keys {unknown}; valid keys are {sorted(names)}")
    kwargs = {}
    for name, value in data.items():
        target = hints[name]
        if dataclasses.is_dataclass(target):
            kwargs[name] = _build(target, value, f"{path}.{name}")
        else:
            kwargs[name] = _coerce(target, value, f"{path}.{name}")
    try:
        return cls(**kwargs)
    except TypeError as exc:
        raise ConfigError(f"{path}: {exc}") from exc


def _coerce(target, value, path):
    if target is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{path}: expected true/false, got {value!r}")
        return value
    if target is int:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) != value:
            raise ConfigError(f"{path}: expected an integer, got {value!r}")
        return int(value)
    if target is float:
        if isinstance(value, str):
            # YAML 1.1 reads "7.2e6" as a string (the exponent needs a sign); accept numeric strings.
            try:
                return float(value)
            except ValueError:
                raise ConfigError(f"{path}: expected a number, got {value!r}") from None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{path}: expected a number, got {value!r}")
        return float(value)
    if target is str:
        if not isinstance(value, str):
            raise ConfigError(f"{path}: expected a string, got {value!r}")
        return value
    return value


def config_from_dict(data: dict) -> ExperimentConfig:
    return _build(ExperimentConfig, data or {}, "config")


def apply_overrides(data: dict, overrides: list[str]) -> dict:
    """Apply ``section.key=value`` overrides (values parsed as YAML) to a nested dict, in place."""
    for item in overrides:
        if "=" not in item:
            raise ConfigError(f"override {item!r} must look like section.key=value")
        key, raw = item.split("=", 1)
        parts = key.strip().split(".")
        node = data
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ConfigError(f"override {item!r}: {part} is not a section")
        node[parts[-1]] = yaml.safe_load(raw)
    return data


def load_config(path: str | Path, overrides: list[str] | None = None) -> ExperimentConfig:
    with Path(path).open() as f:
        data = yaml.safe_load(f) or {}
    if overrides:
        apply_overrides(data, overrides)
    return config_from_dict(data)
