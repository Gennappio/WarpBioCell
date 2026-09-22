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
from warpbiocell.fields.species import SpeciesParams, by_state
from warpbiocell.network.model import BooleanNetwork, load_network
from warpbiocell.network.runtime import InputClamp, NetworkParams
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
    wall_rate_per_h: float = 10.0  # confinement rate against a tissue surface (geometry.confine)


@dataclass(frozen=True)
class LifecycleConfig:
    division_rate_per_h: float = 0.0289  # ln(2)/24 h
    death_rate_per_h: float = 0.0
    anoxic_death_rate_per_h: float = 0.5
    inhibition_threshold: int = 8  # neighbours within the query radius
    hypoxia_threshold_mmHg: float = 8.0
    death_threshold_mmHg: float = 2.0
    glucose_threshold_mM: float = 0.0  # division ramps down below it; 0 disables (needs a glucose species)
    glucose_death_threshold_mM: float = 0.0  # with necrosis_requires_glucose: death needs O2 AND glucose below thresholds
    necrosis_requires_glucose: bool = False  # MicroC's necrosis rule
    placement_factor: float = 1.0
    phenotype_model: str = "rules"  # rules | network (the network section must be enabled)
    apoptosis_rate_per_h: float = 0.0  # network mode: death rate while the Apoptosis fate node is ON
    necrosis_rate_per_h: float = 0.0  # network mode: death rate while the Necrosis fate node is ON


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
    uptake_state_factors: dict = field(default_factory=dict)  # e.g. {hypoxic: 0.5}: multipliers per living state
    grid: GridConfig = field(default_factory=GridConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)


@dataclass(frozen=True)
class SpeciesConfig:
    """A metabolic species beyond oxygen (Milestone 9), solved on the oxygen grid.

    uptake_max_by_state    [unit um^3/h per cell] Michaelis-Menten uptake per living state
    production_by_state    [unit um^3/h per cell] zero-order production per state
    production_source      name of an earlier species whose S/(K_S + S) scales the production
                           (lactate from glucose); null for constant production
    """

    name: str = ""
    unit: str = "mM"
    diffusion_um2_per_h: float = 2.4e5
    michaelis_k: float = 0.04
    boundary_value: float = 5.0
    uptake_max_by_state: dict = field(default_factory=dict)
    production_by_state: dict = field(default_factory=dict)
    production_source: str | None = None

    def params(self) -> SpeciesParams:
        return SpeciesParams(
            name=self.name,
            unit=self.unit,
            diffusion=self.diffusion_um2_per_h,
            michaelis_k=self.michaelis_k,
            boundary_value=self.boundary_value,
            uptake_max=by_state(self.uptake_max_by_state or None),
            production=by_state(self.production_by_state or None),
            production_source=self.production_source,
        )


@dataclass(frozen=True)
class GeometryConfig:
    """Tissue region (Milestone 7). The SDF lives on the oxygen grid (``oxygen.grid``).

    shape          sphere | ellipsoid | spheres | mask
    center_um      centre of the sphere / ellipsoid
    radii_um       sphere radius (first entry) or ellipsoid semi-axes
    spheres        for ``spheres``: list of {center_um: [x, y, z], radius_um: r} (union)
    mask_file      for ``mask``: NIfTI label image (needs the ``masks`` extra); voxel (0,0,0)
                   of the mask sits at ``mask_origin_um``; ``mask_label`` selects one label
    confine        push cells back inside the tissue (wall contact, mechanics.wall_rate_per_h)
    oxygen_source  tissue_surface (every node outside the tissue holds boundary_mmHg) | box
    seed_fill      fill the tissue with cells at initialisation (cells.initial_count ignored)
    seed_margin_um extra distance from the surface kept free of cell centres at seeding
    seed_within_radius_um   optional: seed only inside a sphere of this radius around center_um
    """

    enabled: bool = False
    shape: str = "sphere"
    center_um: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    radii_um: list = field(default_factory=lambda: [250.0, 200.0, 150.0])
    spheres: list = field(default_factory=list)
    mask_file: str | None = None
    mask_label: int | None = None
    mask_origin_um: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    confine: bool = True
    oxygen_source: str = "tissue_surface"
    seed_fill: bool = True
    seed_margin_um: float = 0.0
    seed_within_radius_um: float | None = None


@dataclass(frozen=True)
class NetworkInputConfig:
    """An input node clamped from the environment every step.

    source     oxygen | glucose | <name of one of the first two other species> | constant
    threshold  the node is ON when the value sampled at the cell is above it (below it with
               ``above: false``); for ``constant`` the node is ON when threshold > 0
    """

    node: str = ""
    source: str = "oxygen"
    threshold: float = 0.0
    above: bool = True


@dataclass(frozen=True)
class NetworkConfig:
    """Per-cell Boolean gene network (Milestone 11), see docs/model.md section 4b.

    file                 MaBoSS ``.bnd`` or BoolNet ``.bnet`` model (relative to the config file)
    cfg                  MaBoSS ``.cfg`` with ``$variables`` and initial states (optional)
    update               asynchronous (random single-node updates, = MaBoSS with unit rates) |
                         synchronous (whole-network sweeps) | maboss (continuous-time Markov
                         chain with the model's rates, Gillespie)
    updates_per_step     asynchronous: single-node updates per cell step; synchronous: sweeps
    time_units_per_h     maboss: network time units simulated per simulated hour
    max_events_per_step  maboss: cap on Gillespie events per cell per step
    inputs               list of NetworkInputConfig
    outputs              fate role (proliferation, apoptosis, growth_arrest, necrosis) -> node
                         name, or null for a role the network has no node for; empty: the
                         conventional names (Proliferation, ...) that exist in the network
    initial_states       node -> probability of starting ON (overrides the .cfg istate)
    """

    enabled: bool = False
    file: str = ""
    cfg: str | None = None
    update: str = "asynchronous"
    updates_per_step: int = 100
    time_units_per_h: float = 10.0
    max_events_per_step: int = 20000
    inputs: list = field(default_factory=list)
    outputs: dict = field(default_factory=dict)
    initial_states: dict = field(default_factory=dict)


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
    species: list = field(default_factory=list)  # list[SpeciesConfig], solved after oxygen in this order
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    # ---- model objects -------------------------------------------------------------------

    def stepping(self) -> TimeStepping:
        return TimeStepping(dt_cells=self.simulation.dt_cells_h, dt_mechanics=self.simulation.dt_mechanics_h)

    def contact_params(self) -> ContactParams:
        m = self.mechanics
        return ContactParams(stiffness=m.stiffness, damping=m.damping, max_radius=self.cells.radius_um, margin=m.margin_um, wall_rate=m.wall_rate_per_h)

    def lifecycle_params(self) -> LifecycleParams:
        lc = self.lifecycle
        return LifecycleParams(
            division_rate=lc.division_rate_per_h,
            death_rate=lc.death_rate_per_h,
            anoxic_death_rate=lc.anoxic_death_rate_per_h,
            inhibition_threshold=lc.inhibition_threshold,
            hypoxia_threshold=lc.hypoxia_threshold_mmHg if self.oxygen.enabled else 0.0,
            death_threshold=lc.death_threshold_mmHg if self.oxygen.enabled else 0.0,
            glucose_threshold=lc.glucose_threshold_mM if self.has_glucose else 0.0,
            glucose_death_threshold=lc.glucose_death_threshold_mM if self.has_glucose else 0.0,
            necrosis_requires_glucose=lc.necrosis_requires_glucose and self.has_glucose,
            placement_factor=lc.placement_factor,
            phenotype_model=lc.phenotype_model,
            apoptosis_rate=lc.apoptosis_rate_per_h,
            necrosis_rate=lc.necrosis_rate_per_h,
        )

    def network_files(self, base_dir: str | Path | None = None) -> tuple[Path, Path | None]:
        """Model and .cfg paths, relative paths resolved against ``base_dir`` (the config file's
        directory) when the file exists there, else against the working directory."""
        n = self.network
        if not n.file:
            raise ConfigError("network.file is required when the network is enabled")
        return _resolve_path(n.file, base_dir), (_resolve_path(n.cfg, base_dir) if n.cfg else None)

    def load_network(self, base_dir: str | Path | None = None) -> BooleanNetwork:
        model, cfg = self.network_files(base_dir)
        return load_network(model, cfg)

    def network_params(self, base_dir: str | Path | None = None, network: BooleanNetwork | None = None) -> NetworkParams:
        n = self.network
        return NetworkParams(
            network=network or self.load_network(base_dir),
            update=n.update,
            updates_per_step=n.updates_per_step,
            time_units_per_h=n.time_units_per_h,
            max_events_per_step=n.max_events_per_step,
            inputs=tuple(InputClamp(node=i.node, source=i.source, threshold=i.threshold, above=i.above) for i in n.inputs),
            outputs=dict(n.outputs) or None,
            initial_states=dict(n.initial_states),
        )

    @property
    def has_glucose(self) -> bool:
        return self.oxygen.enabled and any(sp.name == "glucose" for sp in self.species)

    def species_params(self) -> list[SpeciesParams]:
        return [sp.params() for sp in self.species]

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

    def tissue_shape(self):
        """The synthetic shape described by ``geometry`` (None for masks or when disabled)."""
        from warpbiocell.geometry.shapes import Ellipsoid, Sphere, Union

        g = self.geometry
        if not g.enabled or g.shape == "mask":
            return None
        if g.shape == "sphere":
            return Sphere(center=tuple(g.center_um), radius=float(g.radii_um[0]))
        if g.shape == "ellipsoid":
            return Ellipsoid(center=tuple(g.center_um), radii=tuple(float(r) for r in g.radii_um))
        if g.shape == "spheres":
            return Union(tuple(Sphere(center=tuple(s["center_um"]), radius=float(s["radius_um"])) for s in g.spheres))
        raise ConfigError(f"geometry.shape must be sphere, ellipsoid, spheres or mask, got {g.shape!r}")

    @property
    def n_steps(self) -> int:
        return int(round(self.simulation.duration_h / self.simulation.dt_cells_h))

    # ---- validation -------------------------------------------------------------------------

    def validate(self, base_dir: str | Path | None = None) -> list[str]:
        """Raise ConfigError on inconsistencies; return a list of warnings. ``base_dir`` resolves
        relative file names (network models) as in :meth:`network_files`."""
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
            by_state(self.oxygen.uptake_state_factors or None, default=1.0)
            names = ["oxygen"]
            for sp in self.species:
                if not sp.name or sp.name in names:
                    raise ConfigError(f"species need unique, non-empty names other than 'oxygen' (got {sp.name!r})")
                if sp.production_source is not None and sp.production_source not in names:
                    raise ConfigError(f"species {sp.name!r}: production_source must be an earlier species")
                sp.params()  # validates rates and state names
                names.append(sp.name)
            if self.lifecycle.necrosis_requires_glucose and not self.has_glucose:
                warnings.append("lifecycle.necrosis_requires_glucose is set but there is no species named 'glucose': oxygen-only death applies")
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
        g = self.geometry
        if g.enabled:
            if g.oxygen_source not in ("tissue_surface", "box"):
                raise ConfigError("geometry.oxygen_source must be tissue_surface or box")
            if len(g.center_um) != 3 or len(g.mask_origin_um) != 3:
                raise ConfigError("geometry.center_um and mask_origin_um need three components")
            if g.shape == "mask":
                if not g.mask_file:
                    raise ConfigError("geometry.shape = mask needs geometry.mask_file")
            else:
                shape = self.tissue_shape()  # validates shape name and radii
                if g.shape == "spheres" and not g.spheres:
                    raise ConfigError("geometry.shape = spheres needs a non-empty geometry.spheres list")
                geom = self.grid_geometry()
                lo, hi = shape.bounds
                if np.any(lo < np.asarray(geom.origin) + geom.dx) or np.any(hi > np.asarray(geom.upper) - geom.dx):
                    raise ConfigError("the tissue shape does not fit in oxygen.grid with a one-voxel margin")
            if g.seed_within_radius_um is not None and g.seed_within_radius_um <= 0.0:
                raise ConfigError("geometry.seed_within_radius_um must be positive")
            self.contact_params().check_substep(stepping.dt_mechanics)  # includes wall_rate
        n = self.network
        if self.lifecycle.phenotype_model == "network" and not n.enabled:
            raise ConfigError("lifecycle.phenotype_model = network needs network.enabled = true")
        if n.enabled:
            try:
                params = self.network_params(base_dir)
            except (OSError, ValueError) as exc:  # NetworkError, ExpressionError are ValueErrors
                raise ConfigError(f"network: {exc}") from exc
            available = ["constant"]
            if self.oxygen.enabled:
                names = [sp.name for sp in self.species]
                available += ["oxygen"] + [n for n in names if n == "glucose"] + [n for n in names if n != "glucose"][:2]
            for clamp in params.inputs:
                if clamp.source not in available:
                    raise ConfigError(f"network input {clamp.node!r}: source {clamp.source!r} is not one of {available}")
            clamped = {c.node for c in params.inputs}
            p_on = {**{node.name: node.p_on for node in params.network.nodes}, **n.initial_states}
            frozen_random = [name for name in params.network.input_names if name not in clamped and 0.0 < p_on[name] < 1.0]
            if frozen_random:
                warnings.append(f"network input nodes without a clamp are drawn once and then frozen: {frozen_random}")
            if self.lifecycle.phenotype_model == "rules":
                warnings.append("the network runs and its fate counts are recorded, but lifecycle.phenotype_model = rules decides the phenotypes")
            elif not any(params.outputs.values()):
                raise ConfigError("lifecycle.phenotype_model = network needs at least one network output node")
        free_growth = cells.initial_count * np.exp(self.lifecycle.division_rate_per_h * sim.duration_h)
        if free_growth > cells.max_cells and self.lifecycle.phenotype_model == "rules":  # in network mode division needs Proliferation ON
            warnings.append(
                f"free exponential growth would reach {free_growth:.0f} cells > max_cells={cells.max_cells}; "
                "the run stops with status 'capacity_exceeded' if inhibition and death do not limit it"
            )
        return warnings

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


# ---- loading ----------------------------------------------------------------------------------

_LIST_ITEMS = {(ExperimentConfig, "species"): SpeciesConfig, (NetworkConfig, "inputs"): NetworkInputConfig}


def _resolve_path(name: str | Path, base_dir: str | Path | None) -> Path:
    path = Path(name).expanduser()
    if path.is_absolute() or base_dir is None:
        return path
    candidate = Path(base_dir) / path
    return candidate if candidate.exists() or not path.exists() else path


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
        elif (cls, name) in _LIST_ITEMS:
            if not isinstance(value, list):
                raise ConfigError(f"{path}.{name} must be a list")
            kwargs[name] = [_build(_LIST_ITEMS[(cls, name)], item, f"{path}.{name}[{i}]") for i, item in enumerate(value)]
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


def resolve_file_references(data: dict, base_dir: str | Path) -> dict:
    """Make the file names a configuration refers to (network model and .cfg) absolute, relative
    names being taken from ``base_dir`` (the YAML's directory) when they exist there."""
    network = data.get("network")
    if isinstance(network, dict):
        for key in ("file", "cfg"):
            if isinstance(network.get(key), str) and network[key]:
                network[key] = str(_resolve_path(network[key], base_dir))
    return data


def load_config(path: str | Path, overrides: list[str] | None = None) -> ExperimentConfig:
    path = Path(path)
    with path.open() as f:
        data = yaml.safe_load(f) or {}
    if overrides:
        apply_overrides(data, overrides)
    resolve_file_references(data, path.parent)
    return config_from_dict(data)
