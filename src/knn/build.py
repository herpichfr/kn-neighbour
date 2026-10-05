"""Build the enriched catalogue."""

import datetime
import json
import subprocess
from pathlib import Path

import numpy as np

from . import __version__
from .config import photoz_settings, resolve_path, tracer_window
from .environment import Points, get_estimator
from .geometry import comoving_distance, comoving_xyz, make_cosmology
from .io import load_catalogue
from .photoz_env import make_draws, photoz_environment
from .redshift import assign_redshifts, resolve_photoz_params

ESTIMATOR_PARAMS = ("dv", "n_sigma")


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


def build(cfg, output=None):
    """Run the pipeline described by cfg and write the enriched parquet.

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
    """
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

    new_columns = {}
    for entry in cfg["environment"]:
        name, k = entry["name"], entry["k"]
        estimator = get_estimator(name)
        extra = {key: entry[key] for key in ESTIMATOR_PARAMS if key in entry}
        mode = entry.get("photoz_mode", settings["mode"])
        spec_result = estimator(tracers, spec_targets, k, self_index=tracer_pos[spec_rows],
                                sigma=params["sigma"], **extra)
        phot_result = {}
        if len(phot_rows):
            phot_result = photoz_environment(
                estimator, tracers, tracer_pos[phot_rows], draws_for(mode), k,
                list(spec_result), sigma=params["sigma"], **extra)
        for quantity, values in spec_result.items():
            col = f"{quantity}_{name}_k{k}"
            med, lo, hi = (np.full(len(full), np.nan) for _ in range(3))
            med[spec_rows], lo[spec_rows], hi[spec_rows] = values, values, values
            if len(phot_rows):
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
    out = full[in_window].reset_index(drop=True)

    out_path = Path(output) if output is not None else resolve_path(cfg, cfg["output"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    build_info = {
        "n_tracers": int(is_tracer.sum()),
        "n_spec_targets": int(len(spec_rows)),
        "n_phot_targets": int(len(phot_rows)),
        "tracer_window": tw,
        "photoz": {**settings, **params},
    }
    write_provenance(cfg, out_path, build_info)
    return out
