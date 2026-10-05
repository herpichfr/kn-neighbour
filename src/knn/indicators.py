"""Galaxy-evolution indicators: colours, red sequence, quenching, W2-W3.

Sources
-------
W2 - W3 >= 1.5 (Vega) as the star-forming selection: Cluver et al. 2014, ApJ
782, 90 (arXiv:1401.0837), Sec. 4.2 ("selecting galaxies with W2-W3 >= 1.5
(see Figure 5) we choose systems that are most likely dominated by star
formation"). WISE Vega-to-AB offsets as in knn.mass.

The red-sequence procedure (seeding, asymmetric sigma clipping, quenched =
within n sigma of the sequence) is this package's own and not from a paper.
"""

import numpy as np
import pandas as pd

from .mass import WISE_VEGA_TO_AB
from .photometry import rest_frame_ur

NMAD_FACTOR = 1.4826
INDICATOR_DEFAULTS = {
    "red_sequence": {"pivot": 10.0, "mass_range": [9.5, 12.0], "bin_width": 0.4,
                     "min_per_bin": 20, "clip": [2.0, 3.0], "max_iter": 20,
                     "n_sigma_quench": 2.0},
    "w2w3_sf_threshold": 1.5,
}
PASSTHROUGH = ("class_spec", "subclass_final", "type")


def indicator_settings(cfg):
    """The 'indicators' section with defaults filled in."""
    user = cfg.get("indicators") or {}
    return {"red_sequence": {**INDICATOR_DEFAULTS["red_sequence"],
                             **(user.get("red_sequence") or {})},
            "w2w3_sf_threshold": user.get("w2w3_sf_threshold",
                                          INDICATOR_DEFAULTS["w2w3_sf_threshold"])}


def _nmad(x):
    return float(NMAD_FACTOR * np.median(np.abs(x - np.median(x))))


def fit_red_sequence(colour, logm, *, pivot=10.0, mass_range=None, bin_width=0.4,
                     min_per_bin=20, clip=(2.0, 3.0), max_iter=20):
    """Linear red sequence colour = intercept + slope (logm - pivot).

    Sample: galaxies with finite colour and logm, logm inside mass_range if
    given. Seeding: the sample is cut into logm bins of bin_width; in every
    bin with at least min_per_bin galaxies the galaxies redder than the bin
    median (the upper half) form the seed; a line is fitted to the seed.
    Iteration: with sigma the NMAD (1.4826 MAD) of the residuals of the
    current set, the new set is every galaxy of the sample with residual
    in (-clip[0] sigma, +clip[1] sigma), refitted by least squares; repeated
    until the set stops changing or max_iter. The blue side is clipped
    tighter than the red side so that the blue cloud does not drag the
    fit. sigma returned is the NMAD of the residuals of the final set about
    the final line; because the set is truncated it underestimates the
    intrinsic scatter.

    Returns a dict (intercept at the pivot, slope, sigma, pivot, n_sample,
    n_fit, n_iter, converged) or None when no bin has enough galaxies.
    """
    colour = np.asarray(colour, dtype=float)
    logm = np.asarray(logm, dtype=float)
    ok = np.isfinite(colour) & np.isfinite(logm)
    if mass_range is not None:
        ok &= (logm >= mass_range[0]) & (logm <= mass_range[1])
    y, x = colour[ok], logm[ok] - pivot
    n = len(y)
    if n < 3:
        return None
    edges = np.arange(logm[ok].min(), logm[ok].max() + bin_width, bin_width)
    which = np.digitize(logm[ok], edges)
    seed = np.zeros(n, dtype=bool)
    for b in np.unique(which):
        in_bin = which == b
        if in_bin.sum() >= min_per_bin:
            seed |= in_bin & (y >= np.median(y[in_bin]))
    if seed.sum() < 3 or np.ptp(x[seed]) == 0:
        return None
    mask = seed
    converged, it = False, 0
    for it in range(1, max_iter + 1):
        slope, intercept = np.polyfit(x[mask], y[mask], 1)
        resid = y - (intercept + slope * x)
        sigma = _nmad(resid[mask])
        new = (resid > -clip[0] * sigma) & (resid < clip[1] * sigma)
        if new.sum() < 3:
            break
        if (new == mask).all():
            converged = True
            break
        mask = new
    slope, intercept = np.polyfit(x[mask], y[mask], 1)
    sigma = _nmad((y - (intercept + slope * x))[mask])
    return {"intercept": float(intercept), "slope": float(slope), "sigma": float(sigma),
            "pivot": float(pivot), "n_sample": int(n), "n_fit": int(mask.sum()),
            "n_iter": int(it), "converged": bool(converged)}


def red_sequence_colour(fit, logm):
    """Colour of the fitted red sequence at logm."""
    return fit["intercept"] + fit["slope"] * (np.asarray(logm, dtype=float) - fit["pivot"])


def colours(phot, z, settings=None, rows=None):
    """Rest-frame (K-corrected if phot.kcorrection) u-r and Vega W2-W3.

    z one redshift per entry (rows as in knn.mass.compute_masses). Returns
    {'ur', 'w2w3_vega'}; NaN where a magnitude is missing. W3 is 'measured'
    when it is finite (no flux error is available to apply a S/N cut).
    """
    def take(a):
        return a if rows is None else a[rows]

    u_, r_ = take(phot.ur["u"]), take(phot.ur["r"])
    ur = rest_frame_ur(np.asarray(z, dtype=float), u_, r_, apply=phot.kcorrection)
    w2, w3 = take(phot.wise["w2"]), take(phot.wise["w3"])
    shift = 0.0 if phot.wise_vega else -(WISE_VEGA_TO_AB["w2"] - WISE_VEGA_TO_AB["w3"])
    return {"ur": ur, "w2w3_vega": w2 - w3 + shift}


def compute_indicators(frame, settings):
    """Indicator columns for the rows of frame (the enriched catalogue).

    frame needs col_gr, logmstar and z_type; the red sequence is fitted on its
    rows with z_type == 'spec' (all rows of the frame are in the redshift
    window). Returns (columns, info): columns has rs_offset (colour minus
    the red sequence at the row's mass; for photo-z rows at the median mass,
    nothing else is propagated), quenched (nullable boolean, offset >
    -n_sigma_quench sigma_RS) and sf_w2w3 (nullable boolean, Vega W2 - W3
    >= threshold); info is the fit and counts for the provenance.
    """
    rs = settings["red_sequence"]
    is_spec = (frame["z_type"] == "spec").to_numpy()
    colour = frame["col_gr"].to_numpy(float)
    logm = frame["logmstar"].to_numpy(float)
    fit = fit_red_sequence(colour[is_spec], logm[is_spec], pivot=rs["pivot"],
                           mass_range=rs["mass_range"], bin_width=rs["bin_width"],
                           min_per_bin=rs["min_per_bin"], clip=rs["clip"],
                           max_iter=rs["max_iter"])
    n = len(frame)
    cols = {}
    info = {"red_sequence": fit, "settings": settings}
    if fit is None:
        cols["rs_offset"] = np.full(n, np.nan)
        cols["quenched"] = pd.array([pd.NA] * n, dtype="boolean")
    else:
        offset = colour - red_sequence_colour(fit, logm)
        cols["rs_offset"] = offset
        valid = np.isfinite(offset)
        quenched = pd.array(np.where(valid, offset > -rs["n_sigma_quench"] * fit["sigma"], False),
                            dtype="boolean")
        quenched[~valid] = pd.NA
        cols["quenched"] = quenched
        spec_valid = valid & is_spec
        info["quenched_fraction_spec"] = (
            float(np.mean(offset[spec_valid] > -rs["n_sigma_quench"] * fit["sigma"]))
            if spec_valid.any() else None)
        info["n_spec_with_offset"] = int(spec_valid.sum())
    w2w3 = frame["col_w2w3_vega"].to_numpy(float) if "col_w2w3_vega" in frame else np.full(n, np.nan)
    sf = pd.array(np.where(np.isfinite(w2w3), w2w3 >= settings["w2w3_sf_threshold"], False),
                  dtype="boolean")
    sf[~np.isfinite(w2w3)] = pd.NA
    cols["sf_w2w3"] = sf
    info["passthrough_present"] = [c for c in PASSTHROUGH if c in frame.columns]
    return cols, info
