"""Session layer of the interactive explorer (no Dash, no browser).

An Explorer holds the enriched catalogue and everything needed to recompute
an environment estimator with other parameters: the tracer set and the
photo-z redshift draws. Its methods are what the web app (knn.app) calls:

    environment(name, k, ...)  estimator columns for all rows (cached)
    select(filters)            boolean mask of a sample selection
    trend(x, y, mask, ...)     running median / fraction in bins of x, per mass bin and z_type
    concentration(mask, ...)   concentration map, contours, peaks and ridge network (cached)
    model(mask, spec, ...)     ridge model of selected peaks and paths on that map (cached)
    mass_table(columns, ...)   statistics inside / outside the 1 sigma contour
    numeric_columns(), boolean_columns()
    snapshot(state), load_snapshot(text)

Tracers: knn.build writes <output stem>_tracers.parquet next to the enriched
parquet (tracer positions, log mass at z_best, row in the enriched table).
Explorer.from_config reads both files when the provenance sidecar's config
sha256 equals the sha256 of the current config file and the tracer table
exists; otherwise it runs the build in memory (nothing is written; run
`knn build` to cache it). Photo-z draws are made with knn.photoz_env with the
config seed, on the photo-z rows in table order, so the default parameters
reproduce the columns of the build exactly.
"""

import collections
import datetime
import hashlib
import inspect
import json
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import norm

from . import __version__
from .build import ESTIMATOR_PARAMS, _git_head, build_in_memory, tracers_path
from .concentration import (compute_concentration, compute_model, conc_settings, default_sample,
                            default_spec)
from .config import app_settings, photoz_settings, resolve_path
from .environment import ESTIMATORS, Points, get_estimator
from .geometry import make_cosmology
from .phasespace import cluster_settings
from .photoz_env import make_draws, photoz_environment
from .redshift import PHOTOZ_MODES, resolve_photoz_params

NO_PHOTOZ = ("knn3d",)
SNAPSHOT_VERSION = 1
NONE_LABEL = "(none)"
MORPH_COLUMN = "morph_Ser_n"
GROUP_COLUMN = "grp_photz_id_group_final"
FILTER_KEYS = ("z_range", "z_types", "p_in_window_min", "odds_min", "logm_range", "mr_max",
               "flag_member", "substructure", "group", "quenched", "morphology", "conc_region",
               "conc_model", "conc_params")
ENV_CACHE_SIZE = 16
MASK_CACHE_SIZE = 8
CONC_CACHE_SIZE = 8
MODEL_CACHE_SIZE = 8

_BOOKKEEPING_PREFIXES = ("join_", "pdf_", "e_", "ID", "z_significant")
_BOOKKEEPING_SUFFIXES = ("_lo", "_hi")
_BOOKKEEPING_NAMES = {"in_window", "in_overlap_region", "cx", "cy", "cz", "f_z", "v_err",
                      "z_spec_err", "e_z", "THETA", "flag_morph", "morph_flag_morph",
                      "morph_flag_sersic", "flag_member"}
_PREFERRED = ("env", "logmstar", "M_r", "M_i", "col_gr", "col_gi", "col_ur", "col_w2w3_vega",
              "rs_offset", "morph_", "ps_", "conc_", "z_best", "p_in_window", "z_phot_odds", "ra",
              "dec")


def wilson_interval(k, n, conf=0.68):
    """Wilson score interval (lo, hi) of a binomial fraction k / n.

    conf is the central probability (0.68 = about one sigma). k and n may be
    arrays; n = 0 gives NaN.
    """
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    z = norm.ppf(0.5 + conf / 2.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = k / n
        denom = 1.0 + z ** 2 / n
        centre = (p + z ** 2 / (2.0 * n)) / denom
        half = z * np.sqrt(p * (1.0 - p) / n + z ** 2 / (4.0 * n ** 2)) / denom
    return centre - half, centre + half


def is_boolean_like(values):
    """True for a bool / nullable boolean array or an object array of bools and NA."""
    s = pd.Series(values)
    if pd.api.types.is_bool_dtype(s.dtype):
        return True
    if s.dtype == object:
        kept = s.dropna()
        return len(kept) > 0 and all(isinstance(v, (bool, np.bool_)) for v in kept)
    return False


def _as_float(values):
    """float array of a numeric or boolean(-like) column; NA -> NaN."""
    s = pd.Series(values)
    if is_boolean_like(s):
        return s.astype("boolean").astype("Float64").to_numpy(dtype=float, na_value=np.nan)
    return pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)


def _mass_groups(mass_bins):
    """[(label, lo, hi, last)] from a list of edges; [('all', None, None, True)] if none."""
    if mass_bins is None or len(mass_bins) < 2:
        return [("all", None, None, True)]
    edges = sorted(float(e) for e in mass_bins)
    n = len(edges) - 1
    return [(f"{edges[i]:g}-{edges[i + 1]:g}", edges[i], edges[i + 1], i == n - 1)
            for i in range(n)]


TREND_COLUMNS = ("z_type", "mass_bin", "mass_lo", "mass_hi", "bin", "x", "x_lo", "x_hi", "n",
                 "y", "y_lo", "y_hi", "kind")


def trend_table(x, y, mask=None, mass=None, z_type=None, mass_bins=None, nbins=8,
                split_by_ztype=True, min_count=10, conf=0.68):
    """Running median or fraction of y in quantile bins of x (arrays of equal length).

    Groups: every mass bin (edges mass_bins on the array mass; no bins = one
    group 'all') times every z_type ('spec', 'phot'; one group 'all' when
    split_by_ztype is False or z_type is None). Within a group the galaxies
    with mask True and finite x and y are cut into nbins bins at the quantiles
    of x (fewer when x has ties). For a continuous y the bin holds the median
    and the 16th / 84th percentiles; for a boolean y (is_boolean_like) the
    fraction True and its Wilson interval of central probability conf. A bin
    with fewer than min_count galaxies has y, y_lo, y_hi = NaN (n is kept).
    A mass bin is [lo, hi), the last one [lo, hi].

    Returns a DataFrame with TREND_COLUMNS; x is the median x of the bin,
    x_lo and x_hi its edges.
    """
    boolean = is_boolean_like(y)
    xf, yf = _as_float(x), _as_float(y)
    n_all = len(xf)
    keep = np.ones(n_all, dtype=bool) if mask is None else np.asarray(mask, dtype=bool).copy()
    keep &= np.isfinite(xf) & np.isfinite(yf)
    zt = None if z_type is None else np.asarray(z_type, dtype=object)
    types = ["spec", "phot"] if (split_by_ztype and zt is not None) else [None]
    groups = _mass_groups(mass_bins)
    mf = None
    if groups[0][1] is not None:
        if mass is None:
            raise ValueError("mass_bins given but no mass")
        mf = _as_float(mass)
        keep &= np.isfinite(mf)

    rows = []
    for ztype in types:
        in_type = keep if ztype is None else keep & (zt == ztype)
        for label, lo, hi, last in groups:
            sel = in_type
            if lo is not None:
                sel = sel & (mf >= lo) & ((mf <= hi) if last else (mf < hi))
            idx = np.flatnonzero(sel)
            if len(idx) == 0:
                continue
            xs, ys = xf[idx], yf[idx]
            edges = np.unique(np.quantile(xs, np.linspace(0.0, 1.0, nbins + 1)))
            if len(edges) < 2:
                edges = np.array([edges[0], edges[0]])
            which = (np.zeros(len(xs), dtype=int) if len(edges) == 2 and edges[0] == edges[1]
                     else np.searchsorted(edges[1:-1], xs, side="right"))
            for b in range(max(len(edges) - 1, 1)):
                in_bin = which == b
                n = int(in_bin.sum())
                if n == 0:
                    continue
                y_med = y_lo = y_hi = np.nan
                if n >= min_count:
                    if boolean:
                        n_true = float(ys[in_bin].sum())
                        y_med = n_true / n
                        y_lo, y_hi = (float(v) for v in wilson_interval(n_true, n, conf))
                    else:
                        y_lo, y_med, y_hi = np.percentile(ys[in_bin], [16, 50, 84])
                rows.append({"z_type": "all" if ztype is None else ztype, "mass_bin": label,
                             "mass_lo": lo, "mass_hi": hi, "bin": b,
                             "x": float(np.median(xs[in_bin])), "x_lo": float(edges[b]),
                             "x_hi": float(edges[min(b + 1, len(edges) - 1)]), "n": n,
                             "y": y_med, "y_lo": y_lo, "y_hi": y_hi,
                             "kind": "fraction" if boolean else "median"})
    return pd.DataFrame(rows, columns=list(TREND_COLUMNS))


MASS_TABLE_COLUMNS = ("region", "z_type", "stat", "label", "value", "lo", "hi", "n")
MASS_TABLE_REGIONS = ("inside", "outside")
MASS_TABLE_ZTYPES = ("spec", "phot", "all")
# (stat, label, kind, column): kind 'median' = median and 16-84 percentiles, 'fraction' =
# fraction of a 0/1 column with its Wilson interval, 'total' = summed mass
MASS_TABLE_STATS = (
    ("n", "N galaxies", "count", None),
    ("logm", "log M* [M_sun], median [16-84%]", "median", "logmstar"),
    ("mtot", "total M* [log M_sun] (p_in_window weighted)", "total", "logmstar"),
    ("conc", "log map value [weight/Mpc^2] at the galaxy, median [16-84%]", "median",
     "conc_log_density"),
    ("dist", "distance to the nearest ridge [Mpc], median [16-84%]", "median",
     "conc_ridge_dist_mpc"),
    ("quenched", "quenched fraction [Wilson 68%]", "fraction", "quenched"),
    ("env", "environment log(1+delta), median [16-84%]", "median", "env"),
    ("col_gr", "g-r, median [16-84%]", "median", "col_gr"),
    ("r_r200", "R/R200 from the catalogue centre, median [16-84%]", "median",
     "ps_R_over_r200"),
    ("ser_n", "Sersic n, median [16-84%]", "median", MORPH_COLUMN),
    ("morph_c", "concentration C, median [16-84%]", "median", "morph_C"),
    ("in_group", "fraction in a group (id >= 0) [Wilson 68%]", "fraction", GROUP_COLUMN),
)


def mass_stats_table(frame, inside, z_type, mask=None, env=None, conf=0.68):
    """Tidy table of statistics inside and outside the 1 sigma contour of the map.

    frame: DataFrame with one row per galaxy; inside: boolean array (conc_inside);
    z_type: array of 'spec' / 'phot'; mask: boolean array of the galaxies used
    (default all); env: float array, the current estimator's log(1+delta)
    (default: no environment row). Groups: region (inside, outside) x z_type
    (spec, phot, all). A statistic whose column is missing from frame is left
    out. Rows (MASS_TABLE_STATS): N; for 'median' the median and the 16th /
    84th percentiles of the finite values; for 'total' the log10 of the summed
    mass 10**logmstar x p_in_window (p_in_window 1 where missing) of the
    galaxies with a mass; for 'fraction' the share of True / 1 (group: id >= 0)
    and its Wilson interval of central probability conf. A group without
    values has value, lo, hi NaN and n = 0.

    Returns a DataFrame with MASS_TABLE_COLUMNS; n is the number of galaxies
    with a value (the number of galaxies for the N row).
    """
    n_all = len(frame)
    use = np.ones(n_all, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    inside = np.asarray(inside, dtype=bool)
    z_type = np.asarray(z_type, dtype=object)
    p_win = (frame["p_in_window"].to_numpy(float) if "p_in_window" in frame.columns
             else np.ones(n_all))
    p_win = np.where(np.isfinite(p_win), p_win, 1.0)

    def column(name):
        if name == "env":
            return None if env is None else np.asarray(env, dtype=float)
        if name == GROUP_COLUMN:
            if name not in frame.columns:
                return None
            ids = _as_float(frame[name])
            return np.where(np.isfinite(ids), (ids >= 0).astype(float), np.nan)
        return _as_float(frame[name]) if name in frame.columns else None

    rows = []
    for stat, label, kind, name in MASS_TABLE_STATS:
        values = None if name is None else column(name)
        if name is not None and values is None:
            continue
        for region in MASS_TABLE_REGIONS:
            in_region = use & (inside if region == "inside" else ~inside)
            for ztype in MASS_TABLE_ZTYPES:
                sel = in_region if ztype == "all" else in_region & (z_type == ztype)
                value = lo = hi = np.nan
                if kind == "count":
                    n, value = int(sel.sum()), float(sel.sum())
                else:
                    v = values[sel]
                    good = np.isfinite(v)
                    n = int(good.sum())
                    if n and kind == "median":
                        lo, value, hi = (float(x) for x in np.percentile(v[good], [16, 50, 84]))
                    elif n and kind == "fraction":
                        k = float(v[good].sum())
                        value = k / n
                        lo, hi = (float(x) for x in wilson_interval(k, n, conf))
                    elif n and kind == "total":
                        value = float(np.log10(np.sum(p_win[sel][good] * 10.0 ** v[good])))
                rows.append({"region": region, "z_type": ztype, "stat": stat, "label": label,
                             "value": value, "lo": lo, "hi": hi, "n": n})
    return pd.DataFrame(rows, columns=list(MASS_TABLE_COLUMNS))


def _plain(value):
    """YAML-safe copy of a snapshot state (numpy -> python, tuples -> lists)."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return [_plain(v) for v in (sorted(value) if isinstance(value, set) else value)]
    if isinstance(value, np.generic):
        return value.item()
    return value


class Explorer:
    """Catalogue, tracers and caches behind the explorer. See the module docstring.

    df is the enriched catalogue (rows = galaxies in the redshift window),
    tracers the tracer table written by knn.build. Use Explorer.from_config.
    Methods are thread safe (one lock; an environment is computed once).
    """

    def __init__(self, cfg, df, tracers, source="frames"):
        self.cfg = cfg
        self.df = df.reset_index(drop=True)
        self.tracers = tracers.reset_index(drop=True)
        self.source = source
        self.window = list(cfg["redshift_window"])
        self.settings = photoz_settings(cfg)
        self.cosmo = make_cosmology(cfg)
        self.min_count = int(app_settings(cfg)["min_per_bin"])
        self.config_sha256 = cfg.get("_meta", {}).get("sha256")
        self.z_type = self.df["z_type"].to_numpy(object)
        self.is_spec = self.z_type == "spec"
        self.is_phot = self.z_type == "phot"
        self.params = resolve_photoz_params(self.df, self.settings, self.window)
        self.conc_settings = (conc_settings(cfg) if cfg.get("concentration") is not None
                              else None)
        self._lock = threading.RLock()
        self._env_cache = collections.OrderedDict()
        self._mask_cache = collections.OrderedDict()
        self._conc_cache = collections.OrderedDict()
        self._model_cache = collections.OrderedDict()
        self._draws = {}
        self._tracer_sets = {}
        self._git = None
        self._numeric, self._boolean = self._classify_columns()
        self.ids = self._make_ids()

    def _make_ids(self):
        """String id of every row: ID_final where present, else ''."""
        ids = np.full(len(self.df), "", dtype=object)
        if "ID_final" in self.df.columns:
            v = self.df["ID_final"].to_numpy(float)
            ok = np.isfinite(v)
            ids[ok] = [str(int(x)) for x in v[ok]]
        return ids

    # construction -------------------------------------------------------

    @classmethod
    def from_config(cls, cfg, rebuild=False):
        """Explorer of a loaded config (knn.config.load_config).

        Reads the enriched parquet and the tracer table when both exist and
        the provenance config sha256 equals cfg['_meta']['sha256']; otherwise
        (or with rebuild=True) runs the build in memory. The attribute
        'source' is 'parquet' or 'build'.
        """
        out_path = resolve_path(cfg, cfg["output"])
        tracer_file = tracers_path(out_path)
        prov = Path(str(out_path) + ".provenance.json")
        fresh = False
        if not rebuild and out_path.is_file() and tracer_file.is_file() and prov.is_file():
            try:
                recorded = json.loads(prov.read_text()).get("config_sha256")
            except (OSError, ValueError):
                recorded = None
            fresh = recorded is not None and recorded == cfg.get("_meta", {}).get("sha256")
        if fresh:
            return cls(cfg, pd.read_parquet(out_path), pd.read_parquet(tracer_file),
                       source="parquet")
        df, tracers = build_in_memory(cfg)
        return cls(cfg, df, tracers, source="build")

    # columns ------------------------------------------------------------

    def _classify_columns(self):
        numeric, boolean = [], []
        for col in self.df.columns:
            if (col in _BOOKKEEPING_NAMES or col.startswith(_BOOKKEEPING_PREFIXES)
                    or col.endswith(_BOOKKEEPING_SUFFIXES)):
                continue
            series = self.df[col]
            if is_boolean_like(series):
                boolean.append(col)
            elif pd.api.types.is_numeric_dtype(series.dtype):
                numeric.append(col)

        def rank(col):
            for i, pref in enumerate(_PREFERRED):
                if col == pref or (pref.endswith("_") and col.startswith(pref)):
                    return i
            return len(_PREFERRED)

        numeric = sorted(numeric, key=rank)       # stable: catalogue order within a rank
        return numeric, boolean

    def numeric_columns(self, include_env=True):
        """Numeric columns for dropdowns (bookkeeping excluded), 'env' first."""
        return (["env"] if include_env else []) + list(self._numeric)

    def boolean_columns(self):
        """Boolean (nullable) indicator columns, e.g. quenched, sf_w2w3."""
        return list(self._boolean)

    def has_column(self, name):
        return name in self.df.columns

    def estimators(self):
        return sorted(ESTIMATORS)

    def default_environment(self):
        """Parameters of the config's default environment entry for the app.

        The first entry that measures photo-z rows, else the first entry.
        Returns {'name', 'k', 'weight', 'photoz_mode', 'dv', 'n_sigma'}.
        """
        entries = self.cfg["environment"]
        entry = next((e for e in entries if e.get("photoz_mode") != "none"), entries[0])
        return {"name": entry["name"], "k": int(entry["k"]),
                "weight": entry.get("weight", "count"),
                "photoz_mode": entry.get("photoz_mode", self.settings["mode"]),
                "dv": entry.get("dv"), "n_sigma": entry.get("n_sigma")}

    # environment --------------------------------------------------------

    def _mode(self, name, photoz_mode):
        if name in NO_PHOTOZ:
            return "none"
        mode = self.settings["mode"] if photoz_mode is None else photoz_mode
        if mode not in PHOTOZ_MODES + ("none",):
            raise ValueError(f"unknown photo-z mode {mode!r}; choose from "
                             f"{PHOTOZ_MODES + ('none',)}")
        return mode

    @staticmethod
    def _estimator_params(estimator, params):
        """Parameters the estimator takes and that differ from its defaults (None = default)."""
        accepted = inspect.signature(estimator).parameters
        extra = {}
        for key, value in params.items():
            if key not in ESTIMATOR_PARAMS:
                raise TypeError(f"unknown environment parameter {key!r}; "
                                f"known: {ESTIMATOR_PARAMS}")
            if value is not None and key in accepted and float(value) != accepted[key].default:
                extra[key] = float(value)
        return extra

    def _tracer_set(self, weight):
        """(Points, position in them of every table row or -1) for a weight."""
        if weight not in self._tracer_sets:
            t = self.tracers
            keep = np.ones(len(t), dtype=bool)
            mass = None
            if weight == "mass":
                logm = t["logm_point"].to_numpy(float)
                keep = np.isfinite(logm)
                if not keep.any():
                    raise ValueError("weight='mass' needs stellar masses: the config has no "
                                     "'photometry' section or no tracer has a mass")
                mass = 10.0 ** logm[keep]
            pts = Points(t[["cx", "cy", "cz"]].to_numpy(float)[keep],
                         t["ra"].to_numpy(float)[keep], t["dec"].to_numpy(float)[keep],
                         t["z_best"].to_numpy(float)[keep], mass=mass)
            out_row = t["out_row"].to_numpy(int)[keep]
            pos = np.full(len(self.df), -1, dtype=np.intp)
            inside = out_row >= 0
            pos[out_row[inside]] = np.flatnonzero(inside)
            self._tracer_sets[weight] = (pts, pos)
        return self._tracer_sets[weight]

    def _draw_set(self, mode):
        if mode not in self._draws:
            rows = np.flatnonzero(self.is_phot)
            self._draws[mode] = make_draws(self.df.iloc[rows], mode, self.settings,
                                           self.params, self.window, self.cosmo)
        return self._draws[mode]

    def _compute_environment(self, name, k, weight, mode, extra):
        estimator = get_estimator(name)
        trc, pos = self._tracer_set(weight)
        kw = dict(extra)
        if weight == "mass":
            kw["weight"] = "mass"
        df = self.df
        spec_rows = np.flatnonzero(self.is_spec)
        phot_rows = np.flatnonzero(self.is_phot)
        sub = df.iloc[spec_rows]
        targets = Points(sub[["cx", "cy", "cz"]].to_numpy(float), sub["ra"].to_numpy(float),
                         sub["dec"].to_numpy(float), sub["z_best"].to_numpy(float))
        sigma = self.params["sigma"]
        spec_result = estimator(trc, targets, k, self_index=pos[spec_rows], sigma=sigma, **kw)
        phot_result = {}
        if len(phot_rows) and mode != "none":
            phot_result = photoz_environment(estimator, trc, pos[phot_rows],
                                             self._draw_set(mode), k, list(spec_result),
                                             sigma=sigma, **kw)
        suffix = "_mass" if weight == "mass" else ""
        cols = {}
        for quantity, values in spec_result.items():
            col = f"{quantity}_{name}_k{k}{suffix}"
            med, lo, hi = (np.full(len(df), np.nan) for _ in range(3))
            med[spec_rows], lo[spec_rows], hi[spec_rows] = values, values, values
            if phot_result:
                p_lo, p_med, p_hi = phot_result[quantity]
                med[phot_rows], lo[phot_rows], hi[phot_rows] = p_med, p_lo, p_hi
            cols[col], cols[col + "_lo"], cols[col + "_hi"] = med, lo, hi
        return pd.DataFrame(cols, index=df.index)

    def environment(self, name, k, weight="count", photoz_mode=None, **params):
        """Estimator columns for every row, named as in the build.

        Columns '<q>_<name>_k<k>[_mass]' and their _lo, _hi for every
        quantity q of the estimator (r, rho or sigma, log1p_delta). Spec-z
        rows hold the value (_lo = _hi = value); photo-z rows the median and
        the 16th / 84th percentiles over the redshift draws inside the
        window (mode: gauss | point | pdf; None = config photoz.mode). knn3d
        photo-z rows are NaN whatever the mode (REPORT_photoz_validation.md).
        params: dv (km/s), n_sigma for sigma2d; a parameter the estimator
        does not take, or that equals the estimator's default, is ignored (same
        cache entry). Results are cached (LRU, ENV_CACHE_SIZE):
        the same arguments return the identical DataFrame object, which
        callers must not modify.
        """
        estimator = get_estimator(name)
        k = int(k)
        if k < 1:
            raise ValueError("k must be >= 1")
        if weight not in ("count", "mass"):
            raise ValueError(f"unknown weight {weight!r}; choose from count, mass")
        mode = self._mode(name, photoz_mode)
        extra = self._estimator_params(estimator, params)
        key = (name, k, weight, mode, tuple(sorted(extra.items())))
        with self._lock:
            hit = self._env_cache.get(key)
            if hit is not None:
                self._env_cache.move_to_end(key)
                return hit
            result = self._compute_environment(name, k, weight, mode, extra)
            self._env_cache[key] = result
            while len(self._env_cache) > ENV_CACHE_SIZE:
                self._env_cache.popitem(last=False)
            return result

    def env_columns(self, name, k, weight="count"):
        """(value, lo, hi) column names of log1p_delta for an estimator entry."""
        col = f"log1p_delta_{name}_k{int(k)}" + ("_mass" if weight == "mass" else "")
        return col, col + "_lo", col + "_hi"

    def env(self, name, k, weight="count", photoz_mode=None, **params):
        """log1p_delta of an estimator as a DataFrame with columns env, env_lo, env_hi."""
        frame = self.environment(name, k, weight, photoz_mode, **params)
        value, lo, hi = self.env_columns(name, k, weight)
        return pd.DataFrame({"env": frame[value].to_numpy(), "env_lo": frame[lo].to_numpy(),
                             "env_hi": frame[hi].to_numpy()}, index=frame.index)

    # selection ----------------------------------------------------------

    def _need(self, col, key):
        if col not in self.df.columns:
            raise KeyError(f"filter '{key}' needs the catalogue column '{col}'")
        return self.df[col]

    def select(self, filters=None):
        """Boolean mask (one entry per row) of the galaxies passing every filter.

        filters is a dict; absent or None entries are not applied:

        z_range [lo, hi]          z_best inside (inclusive)
        z_types [...]             subset of 'spec', 'phot'
        p_in_window_min v         p_in_window >= v (spec-z rows have 1)
        odds_min v                z_phot_odds >= v for photo-z rows; spec-z rows pass
        logm_range [lo, hi]       logmstar inside; rows without a mass fail
        mr_max v                  M_r <= v; rows without M_r fail
        flag_member [...]         flag_member in the values
        substructure [...]        substructure label in the list; '(none)' = no label
        group [...]               grp_photz_id_group_final in the values; '(none)' = unmatched
        quenched 'any'|'yes'|'no'
        morphology True           has a morphology match (morph_Ser_n finite)
        conc_region 'any'|'inside'|'outside'   inside / outside the 1 sigma contour of the
                                  map of the galaxies passing every other filter
        conc_model {...}          a model spec (knn.concentration.compute_model: peaks and
                                  paths); with conc_region, inside / outside is the 1 sigma
                                  band of that model (model_inside of Explorer.model)
                                  instead of the 1 sigma contour; empty = the contour
        conc_params {...}         keyword arguments of Explorer.concentration (weight,
                                  bandwidth_mpc, ...) for conc_region and conc_model

        An empty list means no filter. A filter whose column is missing from
        the catalogue raises KeyError. Unknown keys raise ValueError.
        """
        f = dict(filters or {})
        for key in f:
            if key not in FILTER_KEYS:
                raise ValueError(f"unknown filter {key!r}; known: {FILTER_KEYS}")
        cache_key = json.dumps(_plain(f), sort_keys=True, default=str)
        with self._lock:
            if cache_key in self._mask_cache:
                self._mask_cache.move_to_end(cache_key)
                return self._mask_cache[cache_key].copy()
        df = self.df
        mask = np.ones(len(df), dtype=bool)

        def active(key):
            v = f.get(key)
            return v is not None and not (isinstance(v, (list, tuple)) and len(v) == 0)

        if active("z_range"):
            lo, hi = f["z_range"]
            z = df["z_best"].to_numpy(float)
            mask &= (z >= lo) & (z <= hi)
        if active("z_types"):
            mask &= np.isin(self.z_type, list(f["z_types"]))
        if active("p_in_window_min") and f["p_in_window_min"] > 0:
            p = self._need("p_in_window", "p_in_window_min").to_numpy(float)
            with np.errstate(invalid="ignore"):
                mask &= self.is_spec | (p >= f["p_in_window_min"])
        if active("odds_min") and f["odds_min"] > 0:
            odds = self._need("z_phot_odds", "odds_min").to_numpy(float)
            with np.errstate(invalid="ignore"):
                mask &= ~self.is_phot | (odds >= f["odds_min"])
        if active("logm_range"):
            lo, hi = f["logm_range"]
            m = self._need("logmstar", "logm_range").to_numpy(float)
            with np.errstate(invalid="ignore"):
                mask &= (m >= lo) & (m <= hi)
        if active("mr_max"):
            mr = self._need("M_r", "mr_max").to_numpy(float)
            with np.errstate(invalid="ignore"):
                mask &= mr <= f["mr_max"]
        if active("flag_member"):
            fm = self._need("flag_member", "flag_member").to_numpy(float)
            mask &= np.isin(fm, [float(v) for v in f["flag_member"]])
        if active("substructure"):
            sub = self._need("substructure", "substructure")
            wanted = [v for v in f["substructure"] if v != NONE_LABEL]
            ok = sub.isin(wanted).to_numpy()
            if NONE_LABEL in f["substructure"]:
                ok |= sub.isna().to_numpy()
            mask &= ok
        if active("group"):
            grp = self._need(GROUP_COLUMN, "group")
            wanted = [int(v) for v in f["group"] if v != NONE_LABEL]
            ok = grp.isin(wanted).to_numpy(dtype=bool, na_value=False)
            if NONE_LABEL in f["group"]:
                ok |= grp.isna().to_numpy()
            mask &= ok
        if active("quenched") and f["quenched"] != "any":
            q = self._need("quenched", "quenched")
            want = {"yes": True, "no": False}[f["quenched"]]
            mask &= (q == want).to_numpy(dtype=bool, na_value=False)
        if active("morphology") and f["morphology"]:
            mask &= np.isfinite(self._need(MORPH_COLUMN, "morphology").to_numpy(float))
        if active("conc_region") and f["conc_region"] != "any":
            if f["conc_region"] not in MASS_TABLE_REGIONS:
                raise ValueError(f"conc_region must be any, inside or outside, got "
                                 f"{f['conc_region']!r}")
            spec = f.get("conc_model")
            empty = spec is None or (isinstance(spec, dict)
                                     and not (spec.get("peaks") or spec.get("paths")))
            if not empty:
                columns, _ = self.model(mask.copy(), spec, **(f.get("conc_params") or {}))
                mask &= columns["model_inside"] == (f["conc_region"] == "inside")
            else:
                columns, _ = self.concentration(mask.copy(), **(f.get("conc_params") or {}))
                mask &= columns["conc_inside"] == (f["conc_region"] == "inside")
        with self._lock:
            self._mask_cache[cache_key] = mask
            while len(self._mask_cache) > MASK_CACHE_SIZE:
                self._mask_cache.popitem(last=False)
        return mask.copy()

    def choices(self):
        """Values for the multi-select controls: flag_member, substructure, group."""
        out = {}
        if "flag_member" in self.df:
            out["flag_member"] = sorted(float(v) for v in self.df["flag_member"].dropna().unique())
        if "substructure" in self.df:
            out["substructure"] = sorted(self.df["substructure"].dropna().unique())
        if GROUP_COLUMN in self.df:
            out["group"] = sorted(int(v) for v in self.df[GROUP_COLUMN].dropna().unique())
        return out

    def counts(self, mask):
        """(n spec-z, n photo-z) among the rows of mask."""
        mask = np.asarray(mask, dtype=bool)
        return int((mask & self.is_spec).sum()), int((mask & self.is_phot).sum())

    # concentration -------------------------------------------------------

    def default_concentration(self):
        """The concentration controls of the config for the app.

        Keys: weight, bandwidth_mpc, contour_step_sigma, min_prominence_sigma. None for
        bandwidth_mpc means Scott's rule. Needs a 'concentration' section in the config.
        """
        if self.conc_settings is None:
            raise ValueError("the config has no 'concentration' section")
        s = self.conc_settings
        return {"weight": s["weight"], "bandwidth_mpc": s["bandwidth_mpc"],
                "contour_step_sigma": s["contour_step_sigma"],
                "min_prominence_sigma": s["min_prominence_sigma"]}

    def _conc_key(self, mask, params):
        """(settings, mask, cache key) of a concentration request."""
        if self.conc_settings is None:
            raise ValueError("the config has no 'concentration' section")
        settings = conc_settings(self.cfg, **params)
        if mask is None:
            mask = default_sample(self.df, settings)
        mask = np.asarray(mask, dtype=bool)
        key = (hashlib.sha1(np.packbits(mask).tobytes()).hexdigest(),
               json.dumps(settings, sort_keys=True, default=str))
        return settings, mask, key

    def concentration(self, mask=None, **params):
        """(columns, result) of knn.concentration.compute_concentration for the galaxies of mask.

        mask: boolean array, default the build sample (z_range and sample of the
        config). params override the 'concentration' section of the config (weight,
        weight_clip, bandwidth_mpc, pixel_mpc, contour_step_sigma, min_prominence_sigma;
        None = the config value). The survey footprint is that of all the rows of the
        catalogue, whatever the mask. Results are cached (LRU, CONC_CACHE_SIZE): the same
        arguments return the identical objects, which callers must not modify. Raises
        ValueError without a 'concentration' section and ConcentrationError for a sample
        that is too small.
        """
        settings, mask, key = self._conc_key(mask, params)
        with self._lock:
            hit = self._conc_cache.get(key)
            if hit is not None:
                self._conc_cache.move_to_end(key)
                return hit
            hit = compute_concentration(self.df, mask, settings, self.cosmo,
                                        cluster_settings(self.cfg)["z"])
            self._conc_cache[key] = hit
            while len(self._conc_cache) > CONC_CACHE_SIZE:
                self._conc_cache.popitem(last=False)
            return hit

    def model(self, mask=None, spec=None, **params):
        """(columns, model) of knn.concentration.compute_model on Explorer.concentration(mask, ...).

        spec: {"peaks": [[ra, dec], ...], "paths": [{"label", "ra", "dec"}, ...]}; None =
        default_spec of the map (every peak and ridge segment). params as for
        concentration. Results are cached (LRU, MODEL_CACHE_SIZE) on the concentration key
        and the spec: the same arguments return the identical objects, which callers must
        not modify. Raises as concentration does, and ValueError (ConcentrationError) for a
        spec that is empty, malformed or not JSON-serialisable.
        """
        _, mask, conc_key = self._conc_key(mask, params)
        columns, result = self.concentration(mask, **params)
        spec = default_spec(result) if spec is None else _plain(spec)
        try:
            key = (conc_key, json.dumps(spec, sort_keys=True))
        except (TypeError, ValueError) as err:
            raise ValueError(f"the model spec is not JSON-serialisable: {err}") from None
        with self._lock:
            hit = self._model_cache.get(key)
            if hit is not None:
                self._model_cache.move_to_end(key)
                return hit
            hit = compute_model(self.df, result, spec)
            self._model_cache[key] = hit
            while len(self._model_cache) > MODEL_CACHE_SIZE:
                self._model_cache.popitem(last=False)
            return hit

    def mass_table(self, columns, mask=None, env=None, inside=None):
        """mass_stats_table of the galaxies of mask, inside / outside as in columns.

        columns: the first element of Explorer.concentration (conc_inside, conc_density,
        conc_ridge_dist_mpc); env: array of the current environment (log(1+delta), one
        entry per row) or None; inside: boolean array (one entry per row) that replaces
        columns['conc_inside'], e.g. model_inside of Explorer.model.
        """
        needed = {name for _, _, _, name in MASS_TABLE_STATS if name} | {"p_in_window"}
        frame = pd.DataFrame({c: self.df[c] for c in needed if c in self.df.columns})
        dens = columns["conc_density"]
        with np.errstate(invalid="ignore", divide="ignore"):
            frame["conc_log_density"] = np.where(dens > 0,
                                                 np.log10(np.where(dens > 0, dens, 1.0)), np.nan)
        frame["conc_ridge_dist_mpc"] = columns["conc_ridge_dist_mpc"]
        return mass_stats_table(frame, columns["conc_inside"] if inside is None else inside,
                                self.z_type, mask, env)

    # trends -------------------------------------------------------------

    def _resolve(self, value):
        if isinstance(value, str):
            return self.df[value]
        arr = pd.Series(value)
        if len(arr) != len(self.df):
            raise ValueError("array arguments need one entry per catalogue row")
        return arr

    def trend(self, x, y, mask=None, mass_bins=None, nbins=8, split_by_ztype=True,
              mass="logmstar"):
        """Tidy trend table of y against x (see trend_table).

        x, y: catalogue column names, or arrays with one entry per row (for
        example the 'env' column of Explorer.env). mask: boolean array of the
        galaxies used. mass_bins: edges in log mass (None or fewer than two =
        no mass binning). The minimum count per bin is app.min_per_bin of the
        config (default 10).
        """
        mass_values = None
        if mass_bins is not None and len(mass_bins) >= 2:
            mass_values = self._resolve(mass)
        return trend_table(self._resolve(x), self._resolve(y), mask, mass_values, self.z_type,
                           mass_bins, nbins, split_by_ztype, self.min_count)

    # snapshots ----------------------------------------------------------

    def git_head(self):
        if self._git is None:
            self._git = _git_head(self.cfg.get("_meta", {}).get("root", ".")) or ""
        return self._git or None

    def snapshot(self, state):
        """YAML text of the UI state with the config sha256, git sha and a timestamp."""
        record = {
            "knn_snapshot": SNAPSHOT_VERSION,
            "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "knn_version": __version__,
            "config_path": self.cfg.get("_meta", {}).get("path"),
            "config_sha256": self.config_sha256,
            "git_head": self.git_head(),
            "state": _plain(state),
        }
        return yaml.safe_dump(record, sort_keys=False, default_flow_style=False)

    def load_snapshot(self, text):
        """Parse snapshot text. Returns the record (key 'state' holds the UI state)
        plus 'matches_config': whether its config sha256 is the current one.
        Raises ValueError for text that is not a snapshot."""
        try:
            record = yaml.safe_load(text)
        except yaml.YAMLError as err:
            raise ValueError(f"not a snapshot: {err}") from None
        if (not isinstance(record, dict) or "knn_snapshot" not in record
                or not isinstance(record.get("state"), dict)):
            raise ValueError("not a knn snapshot (no 'knn_snapshot' and 'state' entries)")
        if record["knn_snapshot"] != SNAPSHOT_VERSION:
            raise ValueError(f"unsupported snapshot version {record['knn_snapshot']!r}")
        record["matches_config"] = record.get("config_sha256") == self.config_sha256
        return record
