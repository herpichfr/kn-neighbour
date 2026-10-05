"""Load and validate the YAML configuration."""

import hashlib
from pathlib import Path

import yaml

REQUIRED_KEYS = ("catalogue", "columns", "cosmology", "redshift_window",
                 "tracers", "environment", "output")
REQUIRED_COLUMNS = ("ra", "dec", "z_spec", "z_phot")
OPTIONAL_COLUMNS = ("z_spec_err", "z_phot_odds")
TRACER_CHOICES = ("spec", "all")


class ConfigError(ValueError):
    """Raised when the configuration file is missing keys or has bad values."""


def _is_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def validate_config(cfg):
    """Check the structure of a configuration dictionary.

    Raises ConfigError with a message naming the offending key.
    """
    if not isinstance(cfg, dict):
        raise ConfigError("configuration must be a YAML mapping")

    for key in REQUIRED_KEYS:
        if key not in cfg:
            raise ConfigError(f"missing required key '{key}'")

    for key in ("catalogue", "output"):
        if not isinstance(cfg[key], str) or not cfg[key]:
            raise ConfigError(f"'{key}' must be a non-empty path string")

    columns = cfg["columns"]
    if not isinstance(columns, dict):
        raise ConfigError("'columns' must be a mapping canonical name -> catalogue column")
    for name in REQUIRED_COLUMNS:
        if name not in columns:
            raise ConfigError(f"missing required key 'columns.{name}'")
    for name, col in columns.items():
        if name not in REQUIRED_COLUMNS + OPTIONAL_COLUMNS:
            raise ConfigError(f"unknown key 'columns.{name}'")
        if not isinstance(col, str) or not col:
            raise ConfigError(f"'columns.{name}' must be a column name string")

    cosmo = cfg["cosmology"]
    if not isinstance(cosmo, dict):
        raise ConfigError("'cosmology' must be a mapping with H0 and Om0")
    for name in ("H0", "Om0"):
        if name not in cosmo:
            raise ConfigError(f"missing required key 'cosmology.{name}'")
        if not _is_number(cosmo[name]):
            raise ConfigError(f"'cosmology.{name}' must be a number")

    window = cfg["redshift_window"]
    if (not isinstance(window, (list, tuple)) or len(window) != 2
            or not all(_is_number(w) for w in window) or not window[0] < window[1]):
        raise ConfigError("'redshift_window' must be [z_lo, z_hi] with z_lo < z_hi")

    if cfg["tracers"] not in TRACER_CHOICES:
        raise ConfigError(f"'tracers' must be one of {TRACER_CHOICES}, got {cfg['tracers']!r}")

    env = cfg["environment"]
    if not isinstance(env, list) or not env:
        raise ConfigError("'environment' must be a non-empty list")
    for i, entry in enumerate(env):
        if not isinstance(entry, dict):
            raise ConfigError(f"'environment[{i}]' must be a mapping")
        for name in ("name", "k"):
            if name not in entry:
                raise ConfigError(f"missing required key 'environment[{i}].{name}'")
        if not isinstance(entry["name"], str):
            raise ConfigError(f"'environment[{i}].name' must be a string")
        k = entry["k"]
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ConfigError(f"'environment[{i}].k' must be an integer >= 1")


def load_config(path):
    """Read and validate a YAML config file.

    Returns the configuration dictionary with an extra '_meta' entry holding
    the resolved config path, the repository root (the config file's parent's
    parent) and the sha256 of the file. Relative paths in the config are
    resolved against that root, see resolve_path.
    """
    path = Path(path).resolve()
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    raw = path.read_bytes()
    cfg = yaml.safe_load(raw)
    if cfg is None:
        raise ConfigError(f"config file is empty: {path}")
    validate_config(cfg)
    cfg["_meta"] = {
        "path": str(path),
        "root": str(path.parent.parent),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    return cfg


def resolve_path(cfg, p):
    """Resolve a path from the config against the repository root."""
    p = Path(p)
    if p.is_absolute():
        return p
    root = cfg.get("_meta", {}).get("root", ".")
    return Path(root) / p
