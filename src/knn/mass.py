"""Luminosity distance, absolute magnitudes and stellar masses.

Sources of every number in this module
--------------------------------------
Taylor et al. 2011, MNRAS 418, 1587 (arXiv:1108.0635), GAMA:
    log10(M*/L_i) = -0.68 + 0.70 (g - i)             (their Eq. 7)
with the rest-frame g - i colour, Chabrier 2003 IMF, L_i in solar units
(absolute AB magnitude of the Sun in the i band M_i,sun = 4.58, Sec. 5.3),
scatter about 0.1 dex. Eq. 8 of the arXiv v2 text writes the combination as
log M* = -0.68 + 0.70 (g - i) - 0.4 M_i; that omits the +0.4 M_i,sun =
+1.832 term (it would give masses 1.8 dex too low), so Eq. 7 with M_i,sun is
used here, which is the self-consistent form.

Cluver et al. 2014, ApJ 782, 90 (arXiv:1401.0837), GAMA-WISE, Sec. 4.2, Eq. 2
(entire sample, z < 0.12, W1 and W2 Vega magnitudes):
    log10(M*/L_W1) = -1.96 (W1 - W2) - 0.03
with L_W1 = 10^(-0.4 (M - M_sun)) and M the absolute W1 Vega magnitude,
M_sun = 3.24. (Their Eq. 1, resolved low-z galaxies, has -2.54 and -0.17.)
Cluver et al. exclude W1 - W2 >= 0.8 as AGN-dominated (Stern et al. 2012).

WISE Vega to AB offsets, m_AB = m_Vega + dm: W1 2.699, W2 3.339, W3 5.174,
W4 6.620 (WISE All-Sky Release Explanatory Supplement, Sec. 4.4h, Table 3;
https://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec4_4h.html).

The colour limits used to reject implausible colours (default g-i in
[-0.25, 2.25], g-r in [-0.25, 1.5], W1-W2 in [-0.2, 0.8]) are not from a
paper: the g-i and g-r limits keep the Chilingarian K-correction polynomials
within about 0.15 mag at z = 0.04; the W1-W2 upper limit is the Cluver
AGN cut and the lower limit is the lower edge of their Fig. 6 axis.
"""

import numpy as np
import astropy.units as u

from .photometry import rest_frame_optical

M_SUN_I_AB = 4.58
TAYLOR_INTERCEPT, TAYLOR_SLOPE = -0.68, 0.70
M_SUN_W1_VEGA = 3.24
CLUVER_INTERCEPT, CLUVER_SLOPE = -0.03, -1.96
WISE_VEGA_TO_AB = {"w1": 2.699, "w2": 3.339, "w3": 5.174, "w4": 6.620}

METHODS = ("taylor", "w1")
MASS_DEFAULTS = {
    "method": "taylor",
    "fallback": None,
    "colour_limits": {"g_i": [-0.25, 2.25], "g_r": [-0.25, 1.5], "w1_w2": [-0.2, 0.8]},
}
# quantities with an uncertainty range (_lo, _hi) in the enriched catalogue
QUANTITIES = ("logmstar", "logmstar_taylor", "logmstar_w1", "M_r", "M_i")


def mass_settings(cfg):
    """The 'mass' section with defaults filled in (colour_limits merged)."""
    user = cfg.get("mass") or {}
    out = {**MASS_DEFAULTS, **user}
    out["colour_limits"] = {**MASS_DEFAULTS["colour_limits"], **(user.get("colour_limits") or {})}
    return out


def luminosity_distance_mpc(z, cosmo):
    """Luminosity distance in Mpc; NaN for z <= 0 or NaN."""
    z = np.asarray(z, dtype=float)
    out = np.full(z.shape, np.nan)
    ok = np.isfinite(z) & (z > 0)
    out[ok] = np.asarray(cosmo.luminosity_distance(z[ok]).to_value(u.Mpc))
    return out


def distance_modulus(z, cosmo):
    """Distance modulus 5 log10(D_L / 10 pc); NaN for z <= 0 or NaN."""
    with np.errstate(invalid="ignore"):
        return 5.0 * np.log10(luminosity_distance_mpc(z, cosmo)) + 25.0


def taylor_logmstar(gi_rest, abs_mag_i):
    """log10 M*/Msun of Taylor et al. 2011 from rest-frame g-i and M_i (AB)."""
    return (TAYLOR_INTERCEPT + TAYLOR_SLOPE * np.asarray(gi_rest, dtype=float)
            + 0.4 * (M_SUN_I_AB - np.asarray(abs_mag_i, dtype=float)))


def cluver_logmstar(w1_w2_vega, abs_mag_w1_vega):
    """log10 M*/Msun of Cluver et al. 2014 (Eq. 2) from W1-W2 and M_W1, both Vega."""
    return (CLUVER_INTERCEPT + CLUVER_SLOPE * np.asarray(w1_w2_vega, dtype=float)
            + 0.4 * (M_SUN_W1_VEGA - np.asarray(abs_mag_w1_vega, dtype=float)))


def _within(x, limits):
    lo, hi = limits
    with np.errstate(invalid="ignore"):
        return (x >= lo) & (x <= hi)


def compute_masses(phot, z, cosmo, settings, rows=None):
    """Absolute magnitudes and stellar masses at redshifts z.

    phot is a Photometry; z holds one redshift per output entry; rows, if
    given, are the indices into the photometry arrays of each entry (used for
    redshift draws: rows = galaxy index of each draw), else entry j is row j.
    settings is mass_settings(cfg).

    Magnitudes are extinction-corrected (already, in phot) and K-corrected if
    phot.kcorrection. A galaxy with a colour outside settings['colour_limits']
    gets NaN for everything that needs that colour (g-i: gi, M_i,
    logmstar_taylor; g-r: gr, M_r; W1-W2: logmstar_w1).

    Returns a dict of arrays: gi, gr (rest-frame colours), M_r, M_i (AB),
    M_w1 (Vega), logmstar_taylor, logmstar_w1.
    """
    z = np.asarray(z, dtype=float)

    def take(a):
        return a if rows is None else a[rows]

    limits = settings["colour_limits"]
    g, r, i = (take(phot.opt[b]) for b in ("g", "r", "i"))
    rest = rest_frame_optical(z, g, r, i, apply=phot.kcorrection)
    dm = distance_modulus(z, cosmo)
    ok_gi = _within(g - i, limits["g_i"])
    ok_gr = _within(g - r, limits["g_r"])
    nan = np.nan
    gi = np.where(ok_gi, rest["gi"], nan)
    gr = np.where(ok_gr, rest["gr"], nan)
    m_i = np.where(ok_gi, rest["i"] - dm, nan)
    m_r = np.where(ok_gr, rest["r"] - dm, nan)

    w1, w2 = take(phot.wise["w1"]), take(phot.wise["w2"])
    off = {b: 0.0 if phot.wise_vega else WISE_VEGA_TO_AB[b] for b in ("w1", "w2")}
    w1_v, w2_v = w1 - off["w1"], w2 - off["w2"]
    c12 = w1_v - w2_v
    ok_w = _within(c12, limits["w1_w2"])
    m_w1 = np.where(ok_w, w1_v - dm, nan)

    return {"gi": gi, "gr": gr, "M_r": m_r, "M_i": m_i, "M_w1": m_w1,
            "logmstar_taylor": taylor_logmstar(gi, m_i),
            "logmstar_w1": np.where(ok_w, cluver_logmstar(c12, m_w1), nan)}


def select_mass(values, method="taylor", fallback=None):
    """The default stellar mass: values[f'logmstar_{method}'], filled from the
    fallback method where it is NaN.

    Returns (logmstar, used) with used an object array 'taylor' / 'w1' / None.
    """
    for name in (method, fallback):
        if name is not None and name not in METHODS:
            raise ValueError(f"unknown mass method {name!r}; choose from {METHODS}")
    primary = np.asarray(values[f"logmstar_{method}"], dtype=float)
    logm = primary.copy()
    used = np.full(logm.shape, None, dtype=object)
    used[np.isfinite(primary)] = method
    if fallback is not None and fallback != method:
        second = np.asarray(values[f"logmstar_{fallback}"], dtype=float)
        use = ~np.isfinite(primary) & np.isfinite(second)
        logm[use] = second[use]
        used[use] = fallback
    return logm, used
