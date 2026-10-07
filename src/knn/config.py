"""Load and validate the YAML configuration."""

import hashlib
from pathlib import Path

import yaml

from .concentration import CONC_DEFAULTS, SAMPLES
from .indicators import INDICATOR_DEFAULTS
from .mass import MASS_DEFAULTS, METHODS
from .phasespace import CLUSTER_KEYS
from .photometry import (EXTINCTION_DEFAULTS, KCORRECTION_DEFAULTS, PHOTOMETRY_DEFAULTS,
                         PHOTOMETRY_SETS)
from .redshift import PHOTOZ_MODES

REQUIRED_KEYS = ("catalogue", "columns", "cosmology", "redshift_window",
                 "tracers", "environment", "output")
REQUIRED_COLUMNS = ("ra", "dec", "z_spec", "z_phot")
OPTIONAL_COLUMNS = ("z_spec_err", "z_phot_odds")
TRACER_CHOICES = ("spec", "all")
PHOTOZ_DEFAULTS = {"mode": "gauss", "n_draws": 200, "seed": 42,
                   "pdf_max_std": 0.5, "sigma": None, "bias": None}
ENVIRONMENT_KEYS = ("name", "k", "dv", "n_sigma", "photoz_mode", "weight")
# per-entry photoz_mode: 'none' leaves the photo-z rows NaN
ENTRY_PHOTOZ_MODES = PHOTOZ_MODES + ("none",)
WEIGHTS = ("count", "mass")
OPTIONAL_SECTIONS = ("photometry", "extinction", "kcorrection", "mass", "indicators",
                     "joins", "cluster", "concentration", "app")
# explorer (knn serve): minimum number of galaxies in a trend bin, else the bin is empty
APP_DEFAULTS = {"min_per_bin": 10}
JOIN_KEYS = ("name", "path", "columns", "prefix", "ra", "dec", "radius_arcsec",
             "unescape_latex", "int_columns")


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


def _section(value, key, allowed):
    if not isinstance(value, dict):
        raise ConfigError(f"'{key}' must be a mapping")
    for name in value:
        if name not in allowed:
            raise ConfigError(f"unknown key '{key}.{name}'")


def _check_pair(value, key):
    if (not isinstance(value, (list, tuple)) or len(value) != 2
            or not all(_is_number(v) for v in value) or not value[0] < value[1]):
        raise ConfigError(f"'{key}' must be [lo, hi] with lo < hi")


def _validate_photometry(sec, cfg):
    _section(sec, "photometry", PHOTOMETRY_DEFAULTS)
    custom = sec.get("sets")
    if custom is not None:
        if not isinstance(custom, dict):
            raise ConfigError("'photometry.sets' must be a mapping")
        for name, spec in custom.items():
            if (not isinstance(spec, dict) or spec.get("system") not in ("sdss", "decam")
                    or not isinstance(spec.get("bands"), dict)):
                raise ConfigError(f"'photometry.sets.{name}' needs 'system' (sdss | decam) "
                                  "and a 'bands' mapping band -> column")
    known = set(PHOTOMETRY_SETS) | set(custom or {})
    for key in ("set", "ur_set", "wise_set"):
        if sec.get(key, PHOTOMETRY_DEFAULTS[key]) not in known:
            raise ConfigError(f"'photometry.{key}' must be one of {sorted(known)}")
    if sec.get("wise_system", "ab") not in ("ab", "vega"):
        raise ConfigError("'photometry.wise_system' must be 'ab' or 'vega'")
    if "missing_above" in sec and (not _is_number(sec["missing_above"])
                                   or sec["missing_above"] <= 0):
        raise ConfigError("'photometry.missing_above' must be a number > 0")


def _validate_extinction(sec, cfg):
    _section(sec, "extinction", EXTINCTION_DEFAULTS)
    ebv = sec.get("ebv")
    if ebv is not None and (not isinstance(ebv, str) or not ebv):
        raise ConfigError("'extinction.ebv' must be null, 'sfd' or a column name")
    if sec.get("sfd_dir") is not None and not isinstance(sec["sfd_dir"], str):
        raise ConfigError("'extinction.sfd_dir' must be a path string or null")
    coeffs = sec.get("coefficients")
    if coeffs is not None and (not isinstance(coeffs, dict)
                               or not all(_is_number(v) for v in coeffs.values())):
        raise ConfigError("'extinction.coefficients' must map band -> A/E(B-V) or be null")


def _validate_kcorrection(sec, cfg):
    _section(sec, "kcorrection", KCORRECTION_DEFAULTS)
    if "apply" in sec and not isinstance(sec["apply"], bool):
        raise ConfigError("'kcorrection.apply' must be true or false")


def _validate_mass(sec, cfg):
    _section(sec, "mass", MASS_DEFAULTS)
    if sec.get("method", "taylor") not in METHODS:
        raise ConfigError(f"'mass.method' must be one of {METHODS}")
    if sec.get("fallback") is not None and sec["fallback"] not in METHODS:
        raise ConfigError(f"'mass.fallback' must be null or one of {METHODS}")
    limits = sec.get("colour_limits")
    if limits is not None:
        _section(limits, "mass.colour_limits", MASS_DEFAULTS["colour_limits"])
        for name, pair in limits.items():
            _check_pair(pair, f"mass.colour_limits.{name}")


def _validate_indicators(sec, cfg):
    _section(sec, "indicators", INDICATOR_DEFAULTS)
    rs = sec.get("red_sequence")
    if rs is not None:
        _section(rs, "indicators.red_sequence", INDICATOR_DEFAULTS["red_sequence"])
        if rs.get("mass_range") is not None:
            _check_pair(rs["mass_range"], "indicators.red_sequence.mass_range")
        if "clip" in rs and (not isinstance(rs["clip"], (list, tuple)) or len(rs["clip"]) != 2
                             or not all(_is_number(c) and c > 0 for c in rs["clip"])):
            raise ConfigError("'indicators.red_sequence.clip' must be two numbers > 0")
        for name in ("pivot", "bin_width", "n_sigma_quench"):
            if name in rs and not _is_number(rs[name]):
                raise ConfigError(f"'indicators.red_sequence.{name}' must be a number")
        for name in ("min_per_bin", "max_iter"):
            if name in rs and (not _is_int(rs[name]) or rs[name] < 1):
                raise ConfigError(f"'indicators.red_sequence.{name}' must be an integer >= 1")
    if "w2w3_sf_threshold" in sec and not _is_number(sec["w2w3_sf_threshold"]):
        raise ConfigError("'indicators.w2w3_sf_threshold' must be a number")


def _validate_joins(sec, cfg):
    if not isinstance(sec, list):
        raise ConfigError("'joins' must be a list")
    names = set()
    for i, entry in enumerate(sec):
        if not isinstance(entry, dict):
            raise ConfigError(f"'joins[{i}]' must be a mapping")
        for key in entry:
            if key not in JOIN_KEYS:
                raise ConfigError(f"unknown key 'joins[{i}].{key}'")
        for key in ("name", "path", "columns"):
            if key not in entry:
                raise ConfigError(f"missing required key 'joins[{i}].{key}'")
        for key in ("name", "path"):
            if not isinstance(entry[key], str) or not entry[key]:
                raise ConfigError(f"'joins[{i}].{key}' must be a non-empty string")
        if entry["name"] in names:
            raise ConfigError(f"'joins[{i}].name' {entry['name']!r} is used twice")
        names.add(entry["name"])
        cols = entry["columns"]
        ok = (isinstance(cols, list) and cols and all(isinstance(c, str) for c in cols)) or (
            isinstance(cols, dict) and cols
            and all(isinstance(k, str) and isinstance(v, str) for k, v in cols.items()))
        if not ok:
            raise ConfigError(f"'joins[{i}].columns' must be a non-empty list of column "
                              "names or a mapping column -> output name")
        if "radius_arcsec" in entry and (not _is_number(entry["radius_arcsec"])
                                         or entry["radius_arcsec"] <= 0):
            raise ConfigError(f"'joins[{i}].radius_arcsec' must be a number > 0")
        if "unescape_latex" in entry and not isinstance(entry["unescape_latex"], bool):
            raise ConfigError(f"'joins[{i}].unescape_latex' must be true or false")
        if "int_columns" in entry and not isinstance(entry["int_columns"], list):
            raise ConfigError(f"'joins[{i}].int_columns' must be a list")


def _validate_cluster(sec, cfg):
    _section(sec, "cluster", CLUSTER_KEYS)
    for key, value in sec.items():
        if value is not None and not _is_number(value):
            raise ConfigError(f"'cluster.{key}' must be a number or null")
    for key in ("r200_mpc", "sigma_v"):
        if sec.get(key) is not None and sec[key] <= 0:
            raise ConfigError(f"'cluster.{key}' must be > 0")
    if (sec.get("ra") is None) != (sec.get("dec") is None):
        raise ConfigError("'cluster.ra' and 'cluster.dec' must be given together")
    if sec.get("ra") is not None and sec.get("z") is None:
        raise ConfigError("'cluster.z' is needed to compute the projected radius "
                          "from 'cluster.ra' and 'cluster.dec'")


def _validate_concentration(sec, cfg):
    _section(sec, "concentration", CONC_DEFAULTS)
    if sec.get("sample", CONC_DEFAULTS["sample"]) not in SAMPLES:
        raise ConfigError(f"'concentration.sample' must be one of {SAMPLES}")
    if sec.get("z_range") is not None:
        _check_pair(sec["z_range"], "concentration.z_range")
    weight = sec.get("weight", CONC_DEFAULTS["weight"])
    if not isinstance(weight, str) or not weight:
        raise ConfigError("'concentration.weight' must be 'count', 'mass' or a column name")
    clip = sec.get("weight_clip", CONC_DEFAULTS["weight_clip"])
    if clip is not None and (not _is_number(clip) or not 0 < clip <= 100):
        raise ConfigError("'concentration.weight_clip' must be null or a percentile in (0, 100]")
    for key in ("bandwidth_mpc", "pixel_mpc"):
        if sec.get(key) is not None and (not _is_number(sec[key]) or sec[key] <= 0):
            raise ConfigError(f"'concentration.{key}' must be a number > 0 or null")
    if "contour_step_sigma" in sec and (not _is_number(sec["contour_step_sigma"])
                                        or sec["contour_step_sigma"] <= 0):
        raise ConfigError("'concentration.contour_step_sigma' must be a number > 0")
    if "min_prominence_sigma" in sec and (not _is_number(sec["min_prominence_sigma"])
                                          or sec["min_prominence_sigma"] < 0):
        raise ConfigError("'concentration.min_prominence_sigma' must be a number >= 0")
    if weight == "mass" and not cfg.get("photometry"):
        raise ConfigError("'concentration.weight' is 'mass': needs a 'photometry' section "
                          "(the stellar masses)")
    if (cfg.get("cluster") or {}).get("z") is None:
        raise ConfigError("'concentration' needs 'cluster.z' (the distance that sets the Mpc "
                          "scale)")


def _validate_app(sec, cfg):
    _section(sec, "app", APP_DEFAULTS)
    if "min_per_bin" in sec and (not _is_int(sec["min_per_bin"]) or sec["min_per_bin"] < 1):
        raise ConfigError("'app.min_per_bin' must be an integer >= 1")


_SECTION_VALIDATORS = {
    "photometry": _validate_photometry, "extinction": _validate_extinction,
    "kcorrection": _validate_kcorrection, "mass": _validate_mass,
    "indicators": _validate_indicators, "joins": _validate_joins,
    "cluster": _validate_cluster, "concentration": _validate_concentration,
    "app": _validate_app,
}


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
        if entry.get("photoz_mode", "point") not in ENTRY_PHOTOZ_MODES:
            raise ConfigError(
                f"'environment[{i}].photoz_mode' must be one of {ENTRY_PHOTOZ_MODES}")
        if entry.get("weight", "count") not in WEIGHTS:
            raise ConfigError(f"'environment[{i}].weight' must be one of {WEIGHTS}")
        if entry.get("weight") == "mass" and not cfg.get("photometry"):
            raise ConfigError(
                f"'environment[{i}].weight' is 'mass': needs a 'photometry' section "
                "(the stellar masses of the tracers)")

    for key in OPTIONAL_SECTIONS:
        if key in cfg and cfg[key] is not None:
            _SECTION_VALIDATORS[key](cfg[key], cfg)


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


def app_settings(cfg):
    """The 'app' section with defaults filled in (APP_DEFAULTS)."""
    return {**APP_DEFAULTS, **(cfg.get("app") or {})}


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
