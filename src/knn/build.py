"""Build the enriched catalogue."""

import datetime
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .config import photoz_settings, resolve_path, tracer_window
from .environment import Points, get_estimator
from .geometry import comoving_distance, comoving_xyz, make_cosmology
from .indicators import colours, compute_indicators, indicator_settings
from .io import load_catalogue
from .joins import apply_joins
from .mass import QUANTITIES, compute_masses, mass_settings, select_mass
from .phasespace import cluster_settings, phase_space
from .photometry import prepare_photometry
from .photoz_env import make_draws, photoz_environment, summarise
from .redshift import assign_redshifts, resolve_photoz_params

ESTIMATOR_PARAMS = ("dv", "n_sigma")
TRACER_SUFFIX = "_tracers.parquet"


def _git_head(root):
    """sha of HEAD in the repository at root, or None."""
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                             capture_output=True, text=True, check=True)
        return res.stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def write_provenance(cfg, output, build_info=None):
    """Write <output>.provenance.json next to the output file.

    build_info, if given, is stored under 'build_info'.
    """
    meta = cfg.get("_meta", {})
    config = {k: v for k, v in cfg.items() if k != "_meta"}
    record = {
        "config": config,
        "config_sha256": meta.get("sha256"),
        "git_head": _git_head(meta.get("root", ".")),
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "knn_version": __version__,
    }
    if build_info is not None:
        record["build_info"] = build_info
    path = Path(str(output) + ".provenance.json")
    path.write_text(json.dumps(record, indent=2, default=str) + "\n")
    return path


def _mass_block(phot, full, cosmo, ms, phot_rows, draws):
    """Stellar mass and magnitude columns over the rows of full.

    Spec-z rows (and photo-z rows without draws) hold the value at z_best,
    with _lo = _hi = the value. For photo-z rows with draws the value is
    the median over the redshift draws inside the window and _lo, _hi are
    the 16th and 84th percentiles: the distance (hence luminosity and mass)
    uncertainty. Returns (columns, logm_point), logm_point the log mass at
    z_best of every row (the mass of a tracer).
    """
    z_best = full["z_best"].to_numpy(float)
    point = compute_masses(phot, z_best, cosmo, ms)
    logm, method = select_mass(point, ms["method"], ms["fallback"])
    point = {**point, "logmstar": logm}
    cols = {}
    for name in QUANTITIES:
        for suffix in ("", "_lo", "_hi"):
            cols[name + suffix] = point[name].copy()
    if draws is not None and len(phot_rows):
        dres = compute_masses(phot, draws.z, cosmo, ms, rows=phot_rows[draws.gal_idx])
        dres["logmstar"], _ = select_mass(dres, ms["method"], ms["fallback"])
        for name in QUANTITIES:
            lo, med, hi = summarise(dres[name], draws)
            cols[name][phot_rows] = med
            cols[name + "_lo"][phot_rows] = lo
            cols[name + "_hi"][phot_rows] = hi
    cols["mass_method"] = method
    extra = colours(phot, z_best)
    cols["col_gi"], cols["col_gr"] = point["gi"], point["gr"]
    cols["col_ur"], cols["col_w2w3_vega"] = extra["ur"], extra["w2w3_vega"]
    return cols, logm


def _add_columns(frame, columns):
    """Assign new columns, refusing to overwrite a catalogue column."""
    for col, values in columns.items():
        if col in frame.columns:
            raise ValueError(f"cannot add column '{col}': the catalogue already has it")
        frame[col] = values


def _run(cfg):
    """Run the pipeline of build; returns (enriched, tracer_table, build_info)."""
    df = load_catalogue(resolve_path(cfg, cfg["catalogue"]), cfg["columns"])
    df = assign_redshifts(df, cfg["redshift_window"])
    window = cfg["redshift_window"]
    tw = tracer_window(cfg)
    settings = photoz_settings(cfg)
    params = resolve_photoz_params(df, settings, window)

    in_tracer_window = ((df["z_best"] > tw[0]) & (df["z_best"] < tw[1])).to_numpy()
    full = df[df["in_window"].to_numpy() | in_tracer_window].reset_index(drop=True)
    in_tracer_window = ((full["z_best"] > tw[0]) & (full["z_best"] < tw[1])).to_numpy()
    in_window = full["in_window"].to_numpy()

    cosmo = make_cosmology(cfg)
    xyz = comoving_xyz(full["ra"], full["dec"], full["z_best"], cosmo)

    is_spec = (full["z_type"] == "spec").to_numpy()
    is_phot = (full["z_type"] == "phot").to_numpy()
    is_tracer = in_tracer_window & is_spec if cfg["tracers"] == "spec" else in_tracer_window
    tracer_pos = np.full(len(full), -1, dtype=np.intp)
    tracer_pos[is_tracer] = np.arange(is_tracer.sum())
    tracers = Points(xyz[is_tracer], full["ra"].to_numpy()[is_tracer],
                     full["dec"].to_numpy()[is_tracer], full["z_best"].to_numpy()[is_tracer])

    spec_rows = np.flatnonzero(is_spec & in_window)
    phot_rows = np.flatnonzero(is_phot & in_window)
    spec_targets = Points(xyz[spec_rows], full["ra"].to_numpy()[spec_rows],
                          full["dec"].to_numpy()[spec_rows], full["z_best"].to_numpy()[spec_rows])

    draw_cache = {}

    def draws_for(mode):
        if mode not in draw_cache:
            draw_cache[mode] = make_draws(full.iloc[phot_rows], mode, settings, params,
                                          window, cosmo)
        return draw_cache[mode]

    # phase 3: photometry -> stellar masses (the tracers of mass-weighted entries need them)
    phot = None
    mass_columns, logm_point = {}, None
    ms = mass_settings(cfg)
    if cfg.get("photometry"):
        phot = prepare_photometry(full, cfg, cfg.get("_meta", {}).get("root", "."))
        draws = draws_for(settings["mode"]) if len(phot_rows) else None
        mass_columns, logm_point = _mass_block(phot, full, cosmo, ms, phot_rows, draws)

    tracer_sets = {"count": (tracers, tracer_pos, int(is_tracer.sum()))}

    def tracer_set(weight):
        """(Points, position of each row in them, n) of the tracers for a weight."""
        if weight not in tracer_sets:
            has_mass = is_tracer & np.isfinite(logm_point)
            pos = np.full(len(full), -1, dtype=np.intp)
            pos[has_mass] = np.arange(has_mass.sum())
            pts = Points(xyz[has_mass], full["ra"].to_numpy()[has_mass],
                         full["dec"].to_numpy()[has_mass], full["z_best"].to_numpy()[has_mass],
                         mass=10.0 ** logm_point[has_mass])
            tracer_sets[weight] = (pts, pos, int(has_mass.sum()))
        return tracer_sets[weight]

    new_columns = {}
    env_info = []
    for entry in cfg["environment"]:
        name, k = entry["name"], entry["k"]
        estimator = get_estimator(name)
        extra = {key: entry[key] for key in ESTIMATOR_PARAMS if key in entry}
        weight = entry.get("weight", "count")
        if weight == "mass":
            extra["weight"] = "mass"
        suffix = "_mass" if weight == "mass" else ""
        trc, trc_pos, n_trc = tracer_set(weight)
        mode = entry.get("photoz_mode", settings["mode"])
        env_info.append({"name": name, "k": k, "weight": weight, "photoz_mode": mode,
                         "n_tracers": n_trc})
        spec_result = estimator(trc, spec_targets, k, self_index=trc_pos[spec_rows],
                                sigma=params["sigma"], **extra)
        phot_result = {}
        if len(phot_rows) and mode != "none":
            phot_result = photoz_environment(
                estimator, trc, trc_pos[phot_rows], draws_for(mode), k,
                list(spec_result), sigma=params["sigma"], **extra)
        for quantity, values in spec_result.items():
            col = f"{quantity}_{name}_k{k}{suffix}"
            med, lo, hi = (np.full(len(full), np.nan) for _ in range(3))
            med[spec_rows], lo[spec_rows], hi[spec_rows] = values, values, values
            if phot_result:
                p_lo, p_med, p_hi = phot_result[quantity]
                med[phot_rows], lo[phot_rows], hi[phot_rows] = p_med, p_lo, p_hi
            new_columns[col], new_columns[col + "_lo"], new_columns[col + "_hi"] = med, lo, hi

    p_in_window = np.full(len(full), np.nan)
    p_in_window[spec_rows] = 1.0
    if len(phot_rows):
        p_in_window[phot_rows] = draws_for(settings["mode"]).p_in_window
    new_columns["p_in_window"] = p_in_window

    full["cx"], full["cy"], full["cz"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    full["d_com"] = comoving_distance(full["z_best"], cosmo)
    for col, values in new_columns.items():
        full[col] = values
    _add_columns(full, mass_columns)
    out = full[in_window].reset_index(drop=True)

    extra_info = {"environment": env_info}
    if phot is not None:
        ind_cols, ind_info = compute_indicators(out, indicator_settings(cfg))
        _add_columns(out, ind_cols)
        spec_out = (out["z_type"] == "spec").to_numpy()
        extra_info["photometry"] = phot.info
        extra_info["mass"] = {
            **ms,
            "n_with_mass_spec": int((spec_out & out["logmstar"].notna().to_numpy()).sum()),
            "n_with_mass_phot": int((~spec_out & out["logmstar"].notna().to_numpy()).sum()),
            "n_mass_tracers": tracer_sets.get("mass", (None, None, None))[2],
            "mass_method_counts": out["mass_method"].value_counts(dropna=False)
            .rename(lambda v: str(v)).to_dict(),
            "photoz_uncertainty": "distance only (redshift draws, same seed as the "
                                  "environment); photometric errors and the 0.1 dex "
                                  "intrinsic scatter of the M/L relation are not propagated",
        }
        extra_info["indicators"] = ind_info
    if cfg.get("joins"):
        raw = (df["ra"].to_numpy(float), df["dec"].to_numpy(float))
        out, join_reports = apply_joins(out, cfg["joins"],
                                        cfg.get("_meta", {}).get("root", "."), reference=raw)
        extra_info["joins"] = join_reports
    if cfg.get("cluster") is not None:
        ps_cols, ps_info = phase_space(out, cluster_settings(cfg), cosmo)
        _add_columns(out, ps_cols)
        extra_info["phase_space"] = ps_info

    build_info = {
        "n_tracers": int(is_tracer.sum()),
        "n_spec_targets": int(len(spec_rows)),
        "n_phot_targets": int(len(phot_rows)),
        "tracer_window": tw,
        "photoz": {**settings, **params},
        **extra_info,
    }
    out_row = np.where(in_window, np.cumsum(in_window) - 1, -1)
    logm_tracer = (logm_point[is_tracer] if logm_point is not None
                   else np.full(int(is_tracer.sum()), np.nan))
    tracer_table = pd.DataFrame({
        "ra": full["ra"].to_numpy()[is_tracer], "dec": full["dec"].to_numpy()[is_tracer],
        "z_best": full["z_best"].to_numpy()[is_tracer],
        "z_type": full["z_type"].to_numpy()[is_tracer],
        "cx": xyz[is_tracer, 0], "cy": xyz[is_tracer, 1], "cz": xyz[is_tracer, 2],
        "logm_point": logm_tracer, "out_row": out_row[is_tracer],
    })
    return out, tracer_table, build_info


def tracers_path(output):
    """Path of the tracer table written next to the enriched catalogue."""
    output = Path(output)
    return output.with_name(output.stem + TRACER_SUFFIX)


def build_in_memory(cfg):
    """Run the pipeline without writing anything.

    Returns (enriched DataFrame, tracer table). The tracer table has one row
    per tracer (ra, dec, z_best, z_type, cx, cy, cz, logm_point = log mass at
    z_best or NaN, out_row = row of the tracer in the enriched DataFrame, -1
    if it lies outside redshift_window); the explorer (knn.explorer) needs it
    to recompute environments.
    """
    out, tracer_table, _ = _run(cfg)
    return out, tracer_table


def build(cfg, output=None):
    """Run the pipeline described by cfg and write the enriched parquet.

    The tracer table (see build_in_memory) is written to <output stem>_tracers.parquet.

    Steps: load catalogue, assign redshifts, resolve the photo-z scatter and
    bias, compute comoving coordinates, compute each configured environment
    estimator. Returns the enriched DataFrame; also writes it to 'output'
    (default: cfg['output']) with a provenance sidecar.

    Tracers come from tracer_window (default: redshift_window); the rows of
    the output and the targets are the galaxies with z_best inside
    redshift_window. Spec-z targets are measured at their position, excluding
    themselves. Photo-z targets are measured at each of n_draws redshift
    draws (config 'photoz'); only draws inside redshift_window are kept. For
    every estimator quantity <col> the column holds the median over the kept
    draws for photo-z rows, <col>_lo and <col>_hi the 16th and 84th
    percentiles (all equal to the value for spec-z rows), and p_in_window is
    the fraction of draws inside the window (1.0 for spec-z rows).

    If the config has a 'photometry' section the steps of phase 3 follow, in
    this order: stellar masses (knn.mass; photo-z rows carry the median and
    the 16th and 84th percentiles over the same redshift draws as the
    environment), mass-weighted environment entries (weight: mass; their
    tracers are the tracers that have a mass, columns '<q>_<name>_k<k>_mass'),
    colours and indicators (knn.indicators), 'joins' (knn.joins) and
    'cluster' phase space (knn.phasespace). An environment entry with
    photoz_mode 'none' leaves its photo-z rows NaN.
    """
    out, tracer_table, build_info = _run(cfg)
    out_path = Path(output) if output is not None else resolve_path(cfg, cfg["output"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    tracer_file = tracers_path(out_path)
    tracer_table.to_parquet(tracer_file, index=False)
    build_info["tracers_file"] = tracer_file.name
    write_provenance(cfg, out_path, build_info)
    return out
