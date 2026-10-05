"""Validate photo-z environments against the spec-z environment."""

import datetime
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from . import __version__
from .build import ESTIMATOR_PARAMS
from .config import photoz_settings, resolve_path, tracer_window
from .environment import Points, get_estimator
from .geometry import comoving_xyz, make_cosmology
from .io import load_catalogue
from .photoz_env import make_draws, photoz_environment
from .redshift import (NMAD_FACTOR, PHOTOZ_MODES, assign_redshifts, overlap_sample,
                       photoz_point_stats, pit, pit_gauss, resolve_photoz_params)

OUTPUT_NAME = "photoz_validation.json"
PIT_BINS = 10
TERCILES = ("low", "mid", "high")
MIN_METRIC_N = 10


def _spearman(a, b):
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(spearmanr(a, b).statistic)


def _nmad(d):
    return float(NMAD_FACTOR * np.median(np.abs(d - np.median(d))))


def _pit_histogram(p):
    p = p[np.isfinite(p)]
    counts, _ = np.histogram(p, bins=PIT_BINS, range=(0.0, 1.0))
    return {"n": int(len(p)), "counts": counts.tolist()}


def _metrics(true, est, lo, hi, edges, with_coverage):
    """Metrics of est against true for one set of galaxies."""
    n = len(true)
    out = {"n": int(n)}
    if n < MIN_METRIC_N:
        return out
    d = est - true
    out["spearman"] = _spearman(true, est)
    out["quartile_acc"] = float(np.mean(np.digitize(est, edges) == np.digitize(true, edges)))
    own = np.quantile(est, [0.25, 0.5, 0.75])
    out["quartile_acc_rank"] = float(np.mean(np.digitize(est, own) == np.digitize(true, edges)))
    out["offset"] = float(np.median(d))
    out["nmad"] = _nmad(d)
    out["coverage"] = float(np.mean((lo <= true) & (true <= hi))) if with_coverage else None
    return out


def _tercile_metrics(true, est, odds, edges):
    """Spearman and quartile accuracy in odds terciles (global true quartiles)."""
    ok = np.isfinite(odds)
    cuts = np.quantile(odds[ok], [1.0 / 3.0, 2.0 / 3.0]) if ok.any() else [np.nan, np.nan]
    group = np.where(ok, np.digitize(odds, cuts), -1)
    out = {}
    for g, label in enumerate(TERCILES):
        m = group == g
        entry = {"n": int(m.sum())}
        if m.sum() >= MIN_METRIC_N:
            entry["spearman"] = _spearman(true[m], est[m])
            entry["quartile_acc"] = float(
                np.mean(np.digitize(est[m], edges) == np.digitize(true[m], edges)))
        out[label] = entry
    return out


def validate_photoz(cfg, output_dir=None):
    """Measure how well photo-z environments recover the spec-z environment.

    Validation sample: galaxies with a spec-z inside redshift_window and a
    photo-z. Tracers: all galaxies with a spec-z inside tracer_window (the
    config's 'tracers' option is not used). Every validation galaxy is a
    tracer at its spec-z position and is excluded from its own neighbours
    by index, both for the true environment (spec-z position) and for the
    photo-z environments (point, gauss and pdf modes; draws outside the
    redshift window are dropped as in the build).

    For every configured environment estimator and photo-z mode, log1p_delta
    of the photo-z environment is compared with the true one on the common
    sample (galaxies with a finite value in all modes): Spearman rho, median
    offset, NMAD scatter, fraction in the correct quartile of the true
    values ('quartile_acc'; 'quartile_acc_rank' uses the quartiles of the
    photo-z values themselves), coverage of the true value by the 16-84
    percentile interval (draw modes) and Spearman / quartile accuracy in
    odds terciles. PIT histograms are given for the mixture (all usable
    components, and after the pdf_max_std cut) and for the gauss mode.

    Writes photoz_validation.json to output_dir (default data/derived under
    the repository root) and returns the results dictionary.
    """
    df = load_catalogue(resolve_path(cfg, cfg["catalogue"]), cfg["columns"])
    window = cfg["redshift_window"]
    tw = tracer_window(cfg)
    df = assign_redshifts(df, window)
    settings = photoz_settings(cfg)
    params = resolve_photoz_params(df, settings, window)
    if params["sigma"] is None or params["bias"] is None:
        raise ValueError(
            f"validation needs photoz sigma and bias: only {params['n_overlap']} galaxies "
            "have a spec-z in the window and a photo-z; set photoz.sigma and photoz.bias")
    cosmo = make_cosmology(cfg)

    is_spec = (df["z_type"] == "spec").to_numpy()
    tracer_rows = np.flatnonzero(is_spec & ((df["z_best"] > tw[0]) & (df["z_best"] < tw[1])).to_numpy())
    tracer_pos = np.full(len(df), -1, dtype=np.intp)
    tracer_pos[tracer_rows] = np.arange(len(tracer_rows))
    tdf = df.iloc[tracer_rows]
    tracers = Points(comoving_xyz(tdf["ra"], tdf["dec"], tdf["z_best"], cosmo),
                     tdf["ra"], tdf["dec"], tdf["z_best"])

    val_rows = np.flatnonzero(overlap_sample(df, window))
    vdf = df.iloc[val_rows].reset_index(drop=True)
    self_index = tracer_pos[val_rows]
    z_spec = vdf["z_spec"].to_numpy(float)
    true_points = Points(comoving_xyz(vdf["ra"], vdf["dec"], z_spec, cosmo),
                         vdf["ra"], vdf["dec"], z_spec)
    odds = vdf["z_phot_odds"].to_numpy(float) if "z_phot_odds" in vdf.columns else None

    bias_all, sigma_all = photoz_point_stats(vdf["z_phot"], z_spec)
    results = {
        "meta": {
            "n_validation": int(len(vdf)),
            "n_tracers": int(len(tracer_rows)),
            "redshift_window": list(window),
            "tracer_window": tw,
            "photoz": {**settings, **params},
            "note": "sigma and bias are measured on the validation sample itself "
                    "unless set in the config",
            "config_sha256": cfg.get("_meta", {}).get("sha256"),
            "knn_version": __version__,
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        },
        "point_stats": {"bias": bias_all, "sigma_nmad": sigma_all, "n": int(len(vdf))},
        "pit": {
            "mixture": _pit_histogram(pit(vdf, z_spec)),
            "mixture_pdf_max_std": _pit_histogram(pit(vdf, z_spec, settings["pdf_max_std"])),
            "gauss": _pit_histogram(
                pit_gauss(vdf["z_phot"], z_spec, params["sigma"], params["bias"])),
        },
        "environment": {},
    }
    if odds is None:
        results["meta"]["odds"] = "no z_phot_odds column: terciles skipped"

    draws = {mode: make_draws(vdf, mode, settings, params, window, cosmo)
             for mode in PHOTOZ_MODES}
    for entry in cfg["environment"]:
        if entry.get("weight", "count") == "mass":
            continue    # needs tracer stellar masses; only the count estimators are validated
        name, k = entry["name"], entry["k"]
        estimator = get_estimator(name)
        extra = {key: entry[key] for key in ESTIMATOR_PARAMS if key in entry}
        true_result = estimator(tracers, true_points, k, self_index=self_index,
                                sigma=params["sigma"], **extra)
        true = true_result["log1p_delta"]
        est = {}
        for mode in PHOTOZ_MODES:
            res = photoz_environment(estimator, tracers, self_index, draws[mode], k,
                                     list(true_result), sigma=params["sigma"], **extra)
            est[mode] = res["log1p_delta"]

        common = np.isfinite(true)
        for mode in PHOTOZ_MODES:
            common &= np.isfinite(est[mode][1])
        edges = np.quantile(true[common], [0.25, 0.5, 0.75]) if common.any() else None
        block = {
            "n_common": int(common.sum()),
            "n_valid": {mode: int(np.isfinite(est[mode][1]).sum()) for mode in PHOTOZ_MODES},
            "modes": {},
        }
        for mode in PHOTOZ_MODES:
            lo, med, hi = (a[common] for a in est[mode])
            overall = _metrics(true[common], med, lo, hi, edges, mode != "point") \
                if common.any() else {"n": 0}
            tercile = _tercile_metrics(true[common], med, odds[common], edges) \
                if (odds is not None and common.any()) else None
            block["modes"][mode] = {"overall": overall, "odds_terciles": tercile}
        results["environment"][f"{name}_k{k}"] = block

    out_dir = Path(output_dir) if output_dir is not None else resolve_path(cfg, "data/derived")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / OUTPUT_NAME
    path.write_text(json.dumps(results, indent=2) + "\n")
    results["output_path"] = str(path)
    return results


def _fmt(x, spec):
    return "   n/a" if x is None or not np.isfinite(x) else format(x, spec)


def format_summary(results):
    """Plain-text summary table of the results of validate_photoz."""
    meta, ps = results["meta"], results["point_stats"]
    ph = meta["photoz"]
    lines = [
        f"validation galaxies: {meta['n_validation']}, tracers: {meta['n_tracers']}",
        f"point estimate: bias = {ps['bias']:.4f}, sigma_NMAD = {ps['sigma_nmad']:.4f}",
        f"gauss mode: bias = {ph['bias']:.4f} ({ph['bias_source']}), "
        f"sigma = {ph['sigma']:.4f} ({ph['sigma_source']})",
        "",
        "PIT histograms (10 bins on [0, 1]):",
    ]
    for key, label in (("mixture", "mixture, all components"),
                       ("mixture_pdf_max_std", "mixture, pdf_max_std cut"),
                       ("gauss", "gauss mode")):
        h = results["pit"][key]
        lines.append(f"  {label:26s} n={h['n']:5d} {h['counts']}")
    for key, block in results["environment"].items():
        lines += ["", f"{key}: common sample n = {block['n_common']}, "
                      f"valid per mode {block['n_valid']}",
                  f"  {'mode':6s} {'sample':8s} {'n':>5s} {'rho':>6s} {'offset':>7s} "
                  f"{'nmad':>6s} {'quart':>6s} {'q_rank':>6s} {'cover':>6s}"]
        for mode, m in block["modes"].items():
            o = m["overall"]
            lines.append(
                f"  {mode:6s} {'all':8s} {o['n']:5d} {_fmt(o.get('spearman'), '6.3f')} "
                f"{_fmt(o.get('offset'), '7.3f')} {_fmt(o.get('nmad'), '6.3f')} "
                f"{_fmt(o.get('quartile_acc'), '6.3f')} "
                f"{_fmt(o.get('quartile_acc_rank'), '6.3f')} {_fmt(o.get('coverage'), '6.3f')}")
            for label, t in (m["odds_terciles"] or {}).items():
                lines.append(
                    f"  {'':6s} {'odds ' + label:8s} {t['n']:5d} "
                    f"{_fmt(t.get('spearman'), '6.3f')} {'':7s} {'':6s} "
                    f"{_fmt(t.get('quartile_acc'), '6.3f')}")
    return "\n".join(lines)
