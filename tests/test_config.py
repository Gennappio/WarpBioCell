from pathlib import Path

import pytest

from warpbiocell.simulation.config import ConfigError, apply_overrides, config_from_dict, load_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
REPO_CONFIG = CONFIG_DIR / "tumor_spheroid.yaml"


@pytest.mark.parametrize("path", sorted(CONFIG_DIR.glob("*.yaml")), ids=lambda p: p.stem)
def test_repository_configs_load_and_validate(path):
    config = load_config(path)
    assert config.name == path.stem
    assert config.validate() == []
    assert config.n_steps == int(config.simulation.duration_h / config.simulation.dt_cells_h)


def test_baseline_config_values():
    config = load_config(REPO_CONFIG)
    assert config.oxygen.diffusion_um2_per_h == 7.2e6
    assert load_config(CONFIG_DIR / "microc_oxygen.yaml").lifecycle.hypoxia_threshold_mmHg == 15.7


def test_defaults_are_a_valid_experiment():
    config = config_from_dict({})
    assert config.validate() == []
    assert config.lifecycle_params().hypoxia_threshold == 8.0
    assert config.stepping().mechanics_substeps == 10


def test_unknown_key_is_an_error_with_its_path():
    with pytest.raises(ConfigError, match=r"config\.oxygen: unknown keys \['boundary_mmhg'\]"):
        config_from_dict({"oxygen": {"boundary_mmhg": 40.0}})


def test_type_errors_are_reported():
    with pytest.raises(ConfigError, match="cells.initial_count"):
        config_from_dict({"cells": {"initial_count": 12.5}})
    with pytest.raises(ConfigError, match="oxygen.enabled"):
        config_from_dict({"oxygen": {"enabled": "yes"}})
    with pytest.raises(ConfigError, match="simulation.device"):
        config_from_dict({"simulation": {"device": 0}})


def test_numeric_strings_are_accepted_for_floats():
    # PyYAML reads 7.2e6 (no exponent sign) as a string.
    config = config_from_dict({"oxygen": {"diffusion_um2_per_h": "7.2e6"}})
    assert config.oxygen.diffusion_um2_per_h == 7.2e6
    with pytest.raises(ConfigError):
        config_from_dict({"oxygen": {"diffusion_um2_per_h": "fast"}})


def test_overrides_apply_to_nested_keys_with_yaml_typing():
    data = {"oxygen": {"boundary_mmHg": 38.0}}
    apply_overrides(data, ["oxygen.boundary_mmHg=150", "oxygen.grid.box_um=1200", "oxygen.enabled=false", "name=culture"])
    config = config_from_dict(data)
    assert config.oxygen.boundary_mmHg == 150.0
    assert config.oxygen.grid.box_um == 1200.0
    assert config.oxygen.enabled is False
    assert config.name == "culture"
    with pytest.raises(ConfigError):
        apply_overrides({}, ["no_equals_sign"])


def test_validation_catches_inconsistent_setups():
    with pytest.raises(ConfigError, match="does not fit"):
        config_from_dict({"cells": {"initial_count": 50_000}, "oxygen": {"grid": {"box_um": 400.0}}}).validate()
    with pytest.raises(ValueError, match="overshoot"):
        config_from_dict({"simulation": {"dt_mechanics_h": 0.1}}).validate()
    with pytest.raises(ConfigError):
        config_from_dict({"cells": {"initial_count": 10, "max_cells": 5}}).validate()
    warnings = config_from_dict({"cells": {"initial_count": 1000, "max_cells": 2000}, "simulation": {"duration_h": 240.0}}).validate()
    assert any("max_cells" in w for w in warnings)


def test_oxygen_disabled_makes_lifecycle_oxygen_independent():
    config = config_from_dict({"oxygen": {"enabled": False}})
    params = config.lifecycle_params()
    assert params.hypoxia_threshold == 0.0 and params.death_threshold == 0.0
