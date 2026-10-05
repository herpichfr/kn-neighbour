"""Photometry for the mass and indicator steps.

Magnitude sets, missing-value handling, Galactic extinction and
K-corrections. All magnitudes are AB unless said otherwise.

Sources of every number in this module
--------------------------------------
Extinction coefficients, A_band / E(B-V)_SFD:

* 'sdss' (S-PLUS u, g, r, i, z): Schlafly & Finkbeiner 2011, ApJ 737, 103
  (arXiv:1012.4804), Table 6, R_V = 3.1 column ("When used with R_V = 3.1,
  these give the coefficients to use with E(B-V)_SFD"): u 4.239, g 3.303,
  r 2.285, i 1.698, z 1.263. The S-PLUS filters are not the SDSS filters;
  the SDSS values are used as an approximation.
* 'decam' (Legacy Surveys g, r, i, z, W1-W4): Legacy Surveys DR9/DR10 catalogue
  documentation, section "Galactic Extinction Coefficients" (fetched from
  https://www.legacysurvey.org/dr9/catalogs/): DECam u 3.995, g 3.214,
  r 2.165, i 1.592, z 1.211 (computed by E. Schlafly following the appendix of
  Schlafly & Finkbeiner 2011) and WISE W1 0.184, W2 0.113, W3 0.0241,
  W4 0.00910 (Fitzpatrick 1999 curve).

K-corrections: Chilingarian, Melchior & Zolotukhin 2010, MNRAS 405, 1409 and
Chilingarian & Zolotukhin 2012, MNRAS 419, 1727. The two-variable polynomials
K(z, colour) = sum_xy c[x][y] z^x colour^y are the coefficients of the authors'
Python code (http://kcor.sai.msu.ru/getthecode/, calc_kcor.py, tables
'g_gi', 'i_gi', 'r_gr', 'r_ur', 'u_ur'), copied programmatically. They are
calibrated for the SDSS ugriz filters and z < 0.5; applying them to S-PLUS
or DECam filters is an approximation (at z < 0.04 the K-corrections are
a few hundredths of a mag). No K-correction is applied to WISE bands: no
primary source for WISE coefficients was fetched.

Missing values: S-PLUS uses 99 for "not measured"; any magnitude that is not
finite or is >= missing_above (default 90) becomes NaN.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MISSING_ABOVE = 90.0
SFD_FILE = "SFD_dust_4096_ngp.fits"

PHOTOMETRY_SETS = {
    "legacy": {
        "system": "decam",
        "bands": {"g": "mag_g", "r": "mag_r", "i": "mag_i", "z": "mag_z",
                  "w1": "mag_w1", "w2": "mag_w2", "w3": "mag_w3", "w4": "mag_w4"},
    },
    "splus_auto": {"system": "sdss",
                   "bands": {b: f"{b}_auto" for b in "ugriz"}},
    "splus_pstotal": {"system": "sdss",
                      "bands": {b: f"{b}_PStotal" for b in "ugriz"}},
    "splus_aper_6": {"system": "sdss",
                     "bands": {b: f"{b}_aper_6" for b in "ugriz"}},
}

EXTINCTION_COEFFICIENTS = {
    "sdss": {"u": 4.239, "g": 3.303, "r": 2.285, "i": 1.698, "z": 1.263},
    "decam": {"u": 3.995, "g": 3.214, "r": 2.165, "i": 1.592, "z": 1.211,
              "w1": 0.184, "w2": 0.113, "w3": 0.0241, "w4": 0.00910},
}

PHOTOMETRY_DEFAULTS = {"set": "legacy", "ur_set": "splus_auto", "wise_set": "legacy",
                       "wise_system": "ab", "missing_above": MISSING_ABOVE, "sets": None}
EXTINCTION_DEFAULTS = {"ebv": None, "sfd_dir": None, "coefficients": None}
KCORRECTION_DEFAULTS = {"apply": True}

BANDS_OPTICAL = ("g", "r", "i")
BANDS_UR = ("u", "r")
BANDS_WISE = ("w1", "w2", "w3")

# K(z, colour) polynomial coefficients c[x][y] (z^x colour^y), see module docstring
KCOR_COEFFICIENTS = {
    "g_gi": [
        [0.0, 0.0, 0.0, 0.0],
        [1.59269, -2.97991, 7.31089, -3.46913],
        [-27.5631, -9.89034, 15.4693, 6.53131],
        [161.969, -76.171, -56.1923, 0.0],
        [-204.457, 217.977, 0.0, 0.0],
        [-50.6269, 0.0, 0.0, 0.0],
    ],
    "i_gi": [
        [0.0, 0.0, 0.0, 0.0],
        [-2.21853, 3.94007, 0.678402, -1.24751],
        [-15.7929, -19.3587, 15.0137, 2.27779],
        [118.791, -40.0709, -30.6727, 0.0],
        [-134.571, 125.799, 0.0, 0.0],
        [-55.4483, 0.0, 0.0, 0.0],
    ],
    "r_gr": [
        [0.0, 0.0, 0.0, 0.0],
        [1.83285, -2.71446, 4.97336, -3.66864],
        [-19.7595, 10.5033, 18.8196, 6.07785],
        [33.6059, -120.713, -49.299, 0.0],
        [144.371, 216.453, 0.0, 0.0],
        [-295.39, 0.0, 0.0, 0.0],
    ],
    "r_ur": [
        [0.0, 0.0, 0.0, 0.0],
        [3.03458, -1.50775, 0.576228, -0.0754155],
        [-47.8362, 19.0053, -3.15116, 0.286009],
        [154.986, -35.6633, 1.09562, 0.0],
        [-188.094, 28.1876, 0.0, 0.0],
        [68.9867, 0.0, 0.0, 0.0],
    ],
    "u_ur": [
        [0.0, 0.0, 0.0, 0.0],
        [10.3686, -6.12658, 2.58748, -0.299322],
        [-138.069, 45.0511, -10.8074, 0.95854],
        [540.494, -43.7644, 3.84259, 0.0],
        [-1005.28, 10.9763, 0.0, 0.0],
        [710.482, 0.0, 0.0, 0.0],
    ],
}


def photometry_settings(cfg):
    """The 'photometry' section with defaults filled in."""
    return {**PHOTOMETRY_DEFAULTS, **(cfg.get("photometry") or {})}


def extinction_settings(cfg):
    """The 'extinction' section with defaults filled in."""
    return {**EXTINCTION_DEFAULTS, **(cfg.get("extinction") or {})}


def kcorrection_settings(cfg):
    """The 'kcorrection' section with defaults filled in."""
    return {**KCORRECTION_DEFAULTS, **(cfg.get("kcorrection") or {})}


def clean_mags(values, missing_above=MISSING_ABOVE):
    """Magnitudes as a float array with missing values set to NaN.

    A value that is not finite (NaN, +-inf) or is >= missing_above (the
    S-PLUS 99 convention) is missing.
    """
    out = np.array(values, dtype=float, copy=True)
    with np.errstate(invalid="ignore"):
        out[~np.isfinite(out) | (out >= missing_above)] = np.nan
    return out


def resolve_set(name, custom=None):
    """The definition {'system', 'bands'} of a magnitude set.

    custom is the config's 'photometry.sets': name -> {'system': 'sdss' |
    'decam', 'bands': {band: catalogue column}}; it adds to or replaces the
    built-in PHOTOMETRY_SETS.
    """
    sets = {**PHOTOMETRY_SETS, **(custom or {})}
    if name not in sets:
        raise KeyError(f"unknown photometry set '{name}'; available: {sorted(sets)}")
    return sets[name]


def extract_mags(df, set_name, bands, custom=None, missing_above=MISSING_ABOVE):
    """Cleaned magnitudes {band: array} of a set, and the set's filter system."""
    spec = resolve_set(set_name, custom)
    out = {}
    for band in bands:
        col = spec["bands"].get(band)
        if col is None:
            raise KeyError(f"photometry set '{set_name}' has no band '{band}'")
        if col not in df.columns:
            raise KeyError(f"column '{col}' (set '{set_name}', band '{band}') "
                           "not found in the catalogue")
        out[band] = clean_mags(df[col].to_numpy(float), missing_above)
    return out, spec["system"]


def sfd_ebv(ra, dec, mapdir):
    """E(B-V) from the Schlegel, Finkbeiner & Davis 1998 map with sfdmap.

    mapdir holds SFD_dust_4096_ngp.fits and SFD_dust_4096_sgp.fits. The map
    is used unscaled: the Schlafly & Finkbeiner / Legacy Surveys coefficients
    are defined per unit E(B-V)_SFD.
    """
    try:
        import sfdmap
    except ImportError as err:
        raise ImportError("extinction.ebv 'sfd' needs the sfdmap package") from err
    if mapdir is None or not (Path(mapdir) / SFD_FILE).is_file():
        raise FileNotFoundError(
            f"extinction.ebv 'sfd' needs extinction.sfd_dir to point to the SFD map "
            f"files ({SFD_FILE} not found in {mapdir!r})")
    return np.asarray(sfdmap.SFDMap(str(mapdir)).ebv(np.asarray(ra, float),
                                                     np.asarray(dec, float)), dtype=float)


def ebv_values(df, ext_cfg, root="."):
    """E(B-V) per row from the 'extinction' settings, and a description.

    ext_cfg['ebv']: None (extinction not applied), 'sfd' (SFD map in
    ext_cfg['sfd_dir'], relative to root if not absolute) or the name of a
    catalogue column. Returns (ebv or None, info).
    """
    source = ext_cfg.get("ebv")
    if source is None:
        return None, {"applied": False, "source": None,
                      "note": "extinction not applied (no E(B-V) source configured)"}
    if source == "sfd":
        mapdir = ext_cfg.get("sfd_dir")
        if mapdir is not None and not Path(mapdir).is_absolute():
            mapdir = Path(root) / mapdir
        ebv = sfd_ebv(df["ra"].to_numpy(float), df["dec"].to_numpy(float), mapdir)
        label = f"sfd ({mapdir})"
    else:
        if source not in df.columns:
            raise KeyError(f"extinction.ebv column '{source}' not found in the catalogue")
        ebv = df[source].to_numpy(float)
        label = f"column {source}"
    finite = ebv[np.isfinite(ebv)]
    return ebv, {"applied": True, "source": label,
                 "ebv_median": float(np.median(finite)) if finite.size else None,
                 "ebv_range": [float(finite.min()), float(finite.max())] if finite.size else None}


def extinction_corrected(mags, system, ebv, overrides=None):
    """Magnitudes corrected for Galactic extinction, m - A_band E(B-V).

    mags is {band: array}; ebv is an array or None (no correction). overrides
    is the config's 'extinction.coefficients': band -> A/E(B-V), replacing the
    built-in values of the system. Returns (mags, coefficients used).
    """
    coeffs = {**EXTINCTION_COEFFICIENTS[system], **(overrides or {})}
    if ebv is None:
        return dict(mags), {}
    out, used = {}, {}
    for band, m in mags.items():
        if band not in coeffs:
            raise KeyError(f"no extinction coefficient for band '{band}' ({system})")
        out[band] = m - coeffs[band] * ebv
        used[band] = coeffs[band]
    return out, used


def kcorrection(filter_name, colour_name, z, colour):
    """K-correction (mag) of a band from a colour, Chilingarian et al.

    filter_name and colour_name select a table, e.g. ('g', 'gi') or
    ('r', 'gr'); z and colour are arrays (broadcast together). K is 0 at z = 0
    and NaN where z or the colour is NaN.
    """
    key = f"{filter_name}_{colour_name}"
    if key not in KCOR_COEFFICIENTS:
        raise KeyError(f"no K-correction table '{key}'; available: {sorted(KCOR_COEFFICIENTS)}")
    z = np.asarray(z, dtype=float)
    colour = np.asarray(colour, dtype=float)
    k = np.zeros(np.broadcast(z, colour).shape)
    for x, row in enumerate(KCOR_COEFFICIENTS[key]):
        for y, coef in enumerate(row):
            if coef != 0.0:
                k = k + coef * z ** x * colour ** y
    return k


def rest_frame_optical(z, g, r, i, apply=True):
    """K-corrected g, r, i from observed (extinction-corrected) magnitudes.

    K_g and K_i come from the observed g-i colour, K_r from g-r. apply=False
    returns the observed magnitudes and zero K-corrections. Returns a dict
    with g, r, i, gi, gr (rest-frame) and kg, kr, ki.
    """
    if apply:
        kg = kcorrection("g", "gi", z, g - i)
        ki = kcorrection("i", "gi", z, g - i)
        kr = kcorrection("r", "gr", z, g - r)
    else:
        kg = ki = kr = np.zeros(np.broadcast(z, g).shape)
    g0, r0, i0 = g - kg, r - kr, i - ki
    return {"g": g0, "r": r0, "i": i0, "gi": g0 - i0, "gr": g0 - r0,
            "kg": kg, "kr": kr, "ki": ki}


def rest_frame_ur(z, u, r, apply=True):
    """Rest-frame u-r from observed (extinction-corrected) u and r.

    K_u and K_r both come from the observed u-r colour (tables 'u_ur' and
    'r_ur'). Returns the rest-frame colour.
    """
    if not apply:
        return u - r
    colour = u - r
    return colour - (kcorrection("u", "ur", z, colour) - kcorrection("r", "ur", z, colour))


@dataclass
class Photometry:
    """Cleaned, extinction-corrected magnitudes of every row of the catalogue.

    opt : g, r, i of the main set; ur : u, r of the u-r set; wise : w1, w2, w3
    (AB unless wise_vega); settings : the photometry settings; info : what
    was done, for the provenance.
    """

    opt: dict
    ur: dict
    wise: dict
    wise_vega: bool
    settings: dict
    kcorrection: bool
    info: dict = field(default_factory=dict)


def prepare_photometry(df, cfg, root="."):
    """Read the magnitude sets named in cfg from df and correct for extinction.

    Uses cfg['photometry'], cfg['extinction'] and cfg['kcorrection'] (see
    their *_settings functions). A missing optional set (u-r, WISE) is not an
    error: its magnitudes are all NaN and the info records it.
    """
    ps = photometry_settings(cfg)
    ext = extinction_settings(cfg)
    kc = kcorrection_settings(cfg)
    custom = ps.get("sets")
    miss = ps["missing_above"]
    ebv, ebv_info = ebv_values(df, ext, root)

    opt_raw, system = extract_mags(df, ps["set"], BANDS_OPTICAL, custom, miss)
    opt, opt_used = extinction_corrected(opt_raw, system, ebv, ext.get("coefficients"))
    info = {"set": ps["set"], "ur_set": ps["ur_set"], "wise_set": ps["wise_set"],
            "missing_above": miss, "extinction": ebv_info,
            "extinction_coefficients": {"optical": opt_used},
            "kcorrection": {"applied": bool(kc["apply"]),
                            "source": "Chilingarian et al. 2010, 2012 (kcor.sai.msu.ru), "
                                      "SDSS-filter polynomials g_gi, i_gi, r_gr, r_ur, u_ur; "
                                      "no K-correction for WISE" if kc["apply"] else None}}

    def optional(set_name, bands, label):
        try:
            mags, sys_ = extract_mags(df, set_name, bands, custom, miss)
        except KeyError as err:
            info[f"{label}_missing"] = str(err)
            nan = np.full(len(df), np.nan)
            return {b: nan for b in bands}, None
        corrected, used = extinction_corrected(mags, sys_, ebv, ext.get("coefficients"))
        info["extinction_coefficients"][label] = used
        return corrected, sys_

    ur, _ = optional(ps["ur_set"], BANDS_UR, "ur")
    wise, _ = optional(ps["wise_set"], BANDS_WISE, "wise")
    return Photometry(opt, ur, wise, ps["wise_system"] == "vega", ps, bool(kc["apply"]), info)
