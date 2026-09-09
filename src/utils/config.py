"""Configuration loading, merging, and validation utility for RiceGuard."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from src.utils.paths import get_configs_dir, get_project_root


class ConfigDict(dict):
    """Dictionary subclass supporting attribute-style access (e.g. cfg.project.name)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k, v in self.items():
            if isinstance(v, dict) and not isinstance(v, ConfigDict):
                self[k] = ConfigDict(v)

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"Configuration key '{key}' not found.")

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def to_dict(self) -> Dict[str, Any]:
        """Convert ConfigDict back to a standard Python dictionary."""
        out = {}
        for k, v in self.items():
            if isinstance(v, ConfigDict):
                out[k] = v.to_dict()
            else:
                out[k] = v
        return out


def load_yaml(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Load and parse a YAML file into a Python dictionary.

    Args:
        file_path: Relative or absolute path to the YAML file.

    Returns:
        Dict[str, Any]: Parsed configuration dictionary.
    """
    path = Path(file_path)
    if not path.is_absolute():
        path = get_project_root() / path

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return data if data is not None else {}


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override dictionary into base dictionary.

    Args:
        base: Base configuration dictionary.
        override: Override dictionary containing new or modified keys.

    Returns:
        Dict[str, Any]: Merged dictionary.
    """
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def validate_config(
    cfg: Dict[str, Any],
    required_keys: Optional[List[str]] = None,
) -> bool:
    """Validate that required top-level or dotted keys exist in the configuration.

    Args:
        cfg: Configuration dictionary.
        required_keys: List of mandatory keys (e.g. ['project.name', 'seed']).

    Returns:
        bool: True if valid.

    Raises:
        ValueError: If any required key is missing.
    """
    if required_keys is None:
        required_keys = ["project.name", "seed", "device.use_cuda"]

    for key_path in required_keys:
        parts = key_path.split(".")
        current = cfg
        for part in parts:
            if not isinstance(current, dict) or part not in current:
                raise ValueError(f"Missing required configuration key: '{key_path}'")
            current = current[part]

    return True


def load_config(
    config_name: Optional[str] = None,
    config_dir: Optional[Union[str, Path]] = None,
    overrides: Optional[Dict[str, Any]] = None,
    as_object: bool = True,
) -> Union[ConfigDict, Dict[str, Any]]:
    """Load base configuration and optionally merge additional configs and overrides.

    Args:
        config_name: Optional name of specific config file (e.g. 'data', 'model.yaml').
        config_dir: Path to directory containing configs (default: configs/).
        overrides: Optional runtime dictionary overrides.
        as_object: If True, return ConfigDict with attribute-access; else dict.

    Returns:
        Union[ConfigDict, Dict[str, Any]]: Merged and validated configuration.
    """
    cdir = Path(config_dir) if config_dir else get_configs_dir()

    # Always load base.yaml first as the foundation
    base_path = cdir / "base.yaml"
    cfg = load_yaml(base_path)

    # Merge additional config if requested
    if config_name:
        if not config_name.endswith(".yaml") and not config_name.endswith(".yml"):
            config_name = f"{config_name}.yaml"
        spec_path = cdir / config_name
        if spec_path.exists() and spec_path != base_path:
            specific_cfg = load_yaml(spec_path)
            cfg = deep_merge(cfg, specific_cfg)

    # Apply programmatic runtime overrides
    if overrides:
        cfg = deep_merge(cfg, overrides)

    # Validate mandatory core keys
    validate_config(cfg)

    return ConfigDict(cfg) if as_object else cfg


def load_all_configs(config_dir: Optional[Union[str, Path]] = None) -> ConfigDict:
    """Load and merge all standard config files into a single unified configuration."""
    cdir = Path(config_dir) if config_dir else get_configs_dir()

    # Load base config
    unified = load_yaml(cdir / "base.yaml")

    standard_files = [
        "paths.yaml",
        "data.yaml",
        "model.yaml",
        "training.yaml",
        "calibration.yaml",
        "uncertainty.yaml",
        "xai.yaml",
        "robustness.yaml",
    ]

    for fname in standard_files:
        fpath = cdir / fname
        if fpath.exists():
            mod_cfg = load_yaml(fpath)
            unified = deep_merge(unified, mod_cfg)

    validate_config(unified)
    return ConfigDict(unified)
