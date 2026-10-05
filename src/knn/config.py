"""Load and validate the YAML configuration."""

import hashlib
from pathlib import Path

import yaml

from .redshift import PHOTOZ_MODES

REQUIRED_KEYS = ("catalogue", "columns", "cosmology", "redshift_window",
                 "tracers", "environment", "output")
REQUIRED_COLUMNS = ("ra", "dec", "z_spec", "z_phot")
OPTIONAL_COLUMNS = ("z_spec_err", "z_phot_odds")
TRACER_CHOICES = ("spec", "all")
PHOTOZ_DEFAULTS = {"mode": "gauss", "n_draws": 200, "seed": 42,
                   "pdf_max_std": 0.5, "sigma": None, "bias": None}
ENVIRONMENT_KEYS = ("name", "k", "dv", "n_sigma", "photoz_mode")


class ConfigError(ValueError):
    """Raised when the configuration file is missing keys or has bad values."""


def _is_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _is_int(x):
    return isinstance(x, int) and not isinstance(x, bool)


def _check_window(value, key):
    if (not isinstance(value, (list, tuple)) or len(value) != 2
            or not all(_is_number(w) for w in value) or not value[0] < value[1]):
        raise ConfigError(f"'{key}' must be [z_lo, z_hi] with z_lo < z_hi")


def _validate_photoz(photoz):
    if not isinstance(photoz, dict):
        raise ConfigError("'photoz' must be a mapping")
    for key in photoz:
        if key not in PHOTOZ_DEFAULTS:
            raise ConfigError(f"unknown key 'photoz.{key}'")
    if photoz.get("mode", "gauss") not in PHOTOZ_MODES:
        raise ConfigError(f"'photoz.mode' must be one of {PHOTOZ_MODES}, got {photoz['mode']!r}")
    if "n_draws" in photoz and (not _is_int(photoz["n_draws"]) or photoz["n_draws"] < 1):
        raise ConfigError("'photoz.n_draws' must be an integer >= 1")
    if "seed" in photoz and not _is_int(photoz["seed"]):
        raise ConfigError("'photoz.seed' must be an integer")
    if "pdf_max_std" in photoz and (not _is_number(photoz["pdf_max_std"])
                                    or photoz["pdf_max_std"] <= 0):
        raise ConfigError("'photoz.pdf_max_std' must be a number > 0")
    if photoz.get("sigma") is not None and (not _is_number(photoz["sigma"])
                                            or photoz["sigma"] <= 0):
        raise ConfigError("'photoz.sigma' must be a number > 0 or null")
    if photoz.get("bias") is not None and not _is_number(photoz["bias"]):
        raise ConfigError("'photoz.bias' must be a number or null")


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
    _check_window(window, "redshift_window")

    if cfg.get("tracer_window") is not None:
        tw = cfg["tracer_window"]
        _check_window(tw, "tracer_window")
        if tw[0] > window[0] or tw[1] < window[1]:
            raise ConfigError("'tracer_window' must contain 'redshift_window'")

    if cfg.get("photoz") is not None:
        _validate_photoz(cfg["photoz"])

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
        for name in entry:
            if name not in ENVIRONMENT_KEYS:
                raise ConfigError(f"unknown key 'environment[{i}].{name}'")
        for name in ("dv", "n_sigma"):
            if name in entry and (not _is_number(entry[name]) or entry[name] <= 0):
                raise ConfigError(f"'environment[{i}].{name}' must be a number > 0")
        if entry.get("photoz_mode", "point") not in PHOTOZ_MODES:
            raise ConfigError(
                f"'environment[{i}].photoz_mode' must be one of {PHOTOZ_MODES}")


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


def photoz_settings(cfg):
    """The 'photoz' section with defaults filled in (PHOTOZ_DEFAULTS)."""
    return {**PHOTOZ_DEFAULTS, **(cfg.get("photoz") or {})}


def tracer_window(cfg):
    """Redshift window of the tracers; defaults to redshift_window."""
    tw = cfg.get("tracer_window")
    return list(tw) if tw is not None else list(cfg["redshift_window"])


def resolve_path(cfg, p):
    """Resolve a path from the config against the repository root."""
    p = Path(p)
    if p.is_absolute():
        return p
    root = cfg.get("_meta", {}).get("root", ".")
    return Path(root) / p
