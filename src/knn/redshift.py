"""Redshift assignment and photo-z sampling."""

import numpy as np
import pandas as pd
from scipy.stats import norm

PHOTOZ_MODES = ("point", "gauss", "pdf")
PDF_COMPONENTS = 3
MIN_OVERLAP = 30
NMAD_FACTOR = 1.4826


def assign_redshifts(df, window):
    """Add z_best, z_type and in_window to a catalogue.

    z_best is z_spec where it exists, otherwise z_phot. z_type is 'spec',
    'phot' or NaN. Spec-z takes precedence even when it lies outside the
    window: the galaxy is then out of the window, whatever its z_phot.
    in_window is True when z_lo < z_best < z_hi (open interval).

    Photo-z draws and probability distributions are handled by
    photoz_draws and pit below.
    """
    z_lo, z_hi = window
    out = df.copy()
    has_spec = out["z_spec"].notna()
    has_phot = ~has_spec & out["z_phot"].notna()

    z_best = out["z_spec"].where(has_spec, out["z_phot"])
    z_type = pd.Series(np.nan, index=out.index, dtype=object)
    z_type[has_spec] = "spec"
    z_type[has_phot] = "phot"

    out["z_best"] = z_best
    out["z_type"] = z_type
    out["in_window"] = ((z_best > z_lo) & (z_best < z_hi)).to_numpy()
    return out


def photoz_point_stats(z_phot, z_spec):
    """Bias and sigma_NMAD of the photo-z point estimate.

    d = (z_phot - z_spec) / (1 + z_spec); bias = median(d) and
    sigma_NMAD = 1.4826 * median(|d - median(d)|). Pairs with a NaN in
    either redshift are ignored. Returns (bias, sigma_nmad), both NaN when
    no pair is usable.
    """
    zp = np.asarray(z_phot, dtype=float)
    zs = np.asarray(z_spec, dtype=float)
    ok = np.isfinite(zp) & np.isfinite(zs)
    if not ok.any():
        return float("nan"), float("nan")
    d = (zp[ok] - zs[ok]) / (1.0 + zs[ok])
    bias = float(np.median(d))
    sigma = float(NMAD_FACTOR * np.median(np.abs(d - bias)))
    return bias, sigma


def overlap_sample(df, window):
    """Boolean mask of galaxies with a spec-z inside the window and a photo-z.

    df needs the canonical columns z_spec and z_phot.
    """
    z_lo, z_hi = window
    zs, zp = df["z_spec"], df["z_phot"]
    return ((zs > z_lo) & (zs < z_hi) & zp.notna()).to_numpy()


def resolve_photoz_params(df, settings, window):
    """Resolve the photo-z sigma and bias.

    A value set in settings (the config's 'photoz' section) is used as it
    is. A null value is measured on the overlap sample (spec-z in the
    window and a photo-z) with photoz_point_stats, provided the sample has
    at least MIN_OVERLAP galaxies; otherwise it stays None.

    Returns {'sigma', 'bias', 'sigma_source', 'bias_source', 'n_overlap'}
    where a source is 'config', 'empirical' or None. Both quantities are in
    units of (z_phot - z_spec) / (1 + z_spec).
    """
    mask = overlap_sample(df, window)
    n_overlap = int(mask.sum())
    measured = {}
    if n_overlap >= MIN_OVERLAP and (settings.get("sigma") is None
                                     or settings.get("bias") is None):
        bias, sigma = photoz_point_stats(df["z_phot"].to_numpy()[mask],
                                         df["z_spec"].to_numpy()[mask])
        measured = {"bias": bias, "sigma": sigma}
    out = {"n_overlap": n_overlap}
    for name in ("sigma", "bias"):
        if settings.get(name) is not None:
            out[name], out[name + "_source"] = float(settings[name]), "config"
        elif name in measured:
            out[name], out[name + "_source"] = measured[name], "empirical"
        else:
            out[name], out[name + "_source"] = None, None
    return out


def _mixture(df, max_std=None):
    """Weights, means, stds and usable flags of the photo-z Gaussian mixture.

    Reads pdf_weights_j, pdf_means_j, pdf_stds_j (j = 0, 1, 2). A component
    is usable when weight > 0, std > 0, all three are finite and, if max_std
    is given, std <= max_std. Weights of unusable components are set to 0
    and the rest renormalised per galaxy, so a row with no usable component
    has all weights 0. Returns (w, m, s, ok), each (N, 3), with m and s
    replaced by 0 and 1 where not usable.
    """
    names = [f"pdf_{kind}_{j}" for kind in ("weights", "means", "stds")
             for j in range(PDF_COMPONENTS)]
    missing = [c for c in names if c not in df.columns]
    if missing:
        raise KeyError(f"photo-z PDF columns not found in the catalogue: {missing}")
    w, m, s = (np.column_stack([df[f"pdf_{kind}_{j}"].to_numpy(float)
                                for j in range(PDF_COMPONENTS)])
               for kind in ("weights", "means", "stds"))
    with np.errstate(invalid="ignore"):
        ok = np.isfinite(w) & np.isfinite(m) & np.isfinite(s) & (w > 0) & (s > 0)
        if max_std is not None:
            ok &= s <= max_std
    w = np.where(ok, w, 0.0)
    total = w.sum(axis=1, keepdims=True)
    w = np.where(total > 0, w / np.where(total > 0, total, 1.0), 0.0)
    return w, np.where(ok, m, 0.0), np.where(ok, s, 1.0), ok


def photoz_draws(df, mode, n_draws, rng, sigma=None, bias=None, pdf_max_std=0.5):
    """Draw redshifts for galaxies with a photo-z. Returns (N, n_draws) floats.

    df needs the canonical column z_phot (and the pdf_* columns for 'pdf').

    mode 'point': z_phot, one column (n_draws and rng are not used).
    mode 'gauss': N(z_phot - bias (1 + z_phot), sigma (1 + z_phot)); sigma is
        required, bias None means 0. bias and sigma are in units of
        (z_phot - z_spec) / (1 + z_spec), see photoz_point_stats.
    mode 'pdf': draws from the renormalised Gaussian mixture, after dropping
        components with std > pdf_max_std. Rows with no usable component are
        NaN.
    """
    n = len(df)
    if mode not in PHOTOZ_MODES:
        raise ValueError(f"unknown photo-z mode {mode!r}; choose from {PHOTOZ_MODES}")
    if mode == "point":
        return df["z_phot"].to_numpy(float).reshape(n, 1)
    if n_draws < 1:
        raise ValueError("n_draws must be >= 1")

    if mode == "gauss":
        if sigma is None:
            raise ValueError(
                "photo-z mode 'gauss' needs sigma: set photoz.sigma in the config or "
                f"provide at least {MIN_OVERLAP} galaxies with both a spec-z in the "
                "window and a photo-z")
        zp = df["z_phot"].to_numpy(float)
        centre = zp - (0.0 if bias is None else bias) * (1.0 + zp)
        return centre[:, None] + (sigma * (1.0 + zp))[:, None] * rng.standard_normal((n, n_draws))

    w, m, s, _ = _mixture(df, pdf_max_std)
    cum = np.cumsum(w, axis=1)
    usable = cum[:, -1] > 0
    # the last column is exactly 1 for usable rows, trailing dropped
    # components included, so a uniform draw always finds a component
    cum = cum / np.where(usable, cum[:, -1], 1.0)[:, None]
    u = rng.random((n, n_draws))
    comp = (u[:, :, None] < cum[:, None, :]).argmax(axis=2)
    rows = np.arange(n)[:, None]
    z = m[rows, comp] + s[rows, comp] * rng.standard_normal((n, n_draws))
    z[~usable] = np.nan
    return z


def pit(df, z_true, pdf_max_std=None):
    """Probability integral transform of z_true under each galaxy's mixture.

    Returns the mixture CDF at z_true, shape (N,). A calibrated PDF gives
    values uniform on [0, 1]. Components with std > pdf_max_std are dropped
    (None keeps all usable components); rows without a usable component
    are NaN.
    """
    w, m, s, _ = _mixture(df, pdf_max_std)
    z = np.asarray(z_true, dtype=float)[:, None]
    p = (w * norm.cdf((z - m) / s)).sum(axis=1)
    p[w.sum(axis=1) == 0] = np.nan
    return p


def pit_gauss(z_phot, z_true, sigma, bias=0.0):
    """PIT of z_true under the 'gauss' mode distribution of each galaxy."""
    zp = np.asarray(z_phot, dtype=float)
    centre = zp - bias * (1.0 + zp)
    return norm.cdf((np.asarray(z_true, dtype=float) - centre) / (sigma * (1.0 + zp)))
