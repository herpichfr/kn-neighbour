"""Projected k-th neighbour surface density in a redshift slice."""

import numpy as np

from . import register

C_KMS = 299792.458


def slice_half_width(z, is_phot, dv, n_sigma, sigma):
    """Half-width in redshift of the slice around each galaxy.

    Spec-z galaxies: dv (1 + z) / c, with dv in km/s. Photo-z galaxies:
    n_sigma * sigma * (1 + z). sigma is required when any galaxy is photo-z.
    """
    z = np.asarray(z, dtype=float)
    is_phot = np.asarray(is_phot, dtype=bool)
    if is_phot.any() and sigma is None:
        raise ValueError(
            "sigma2d needs the photo-z sigma for photo-z targets: set photoz.sigma in "
            "the config or provide a spec-z / photo-z overlap sample")
    h_spec = dv * (1.0 + z) / C_KMS
    h_phot = n_sigma * (0.0 if sigma is None else sigma) * (1.0 + z)
    return np.where(is_phot, h_phot, h_spec)


def _unit_vectors(ra, dec):
    ra = np.radians(np.asarray(ra, dtype=float))
    dec = np.radians(np.asarray(dec, dtype=float))
    return np.column_stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])


def kth_angular_separation(tracers, targets, half_width, k, self_index=None):
    """Angular separation (radians) to the k-th nearest tracer in a slice.

    The slice of target i holds the tracers with |z_tracer - z_i| <=
    half_width[i]. self_index[i] is the index in tracers of target i, or -1;
    that tracer is excluded by index. Targets with fewer than k tracers in
    the slice get NaN.

    Tracers are sorted by redshift once and each slice is found with
    searchsorted; the angular distances of one target are computed at once
    for the whole slice.
    """
    n_target = len(targets)
    order = np.argsort(tracers.z, kind="stable")
    z_sorted = tracers.z[order]
    unit = _unit_vectors(tracers.ra[order], tracers.dec[order])
    position = np.empty(len(order), dtype=np.intp)
    position[order] = np.arange(len(order))

    lo = np.searchsorted(z_sorted, targets.z - half_width, side="left")
    hi = np.searchsorted(z_sorted, targets.z + half_width, side="right")
    target_unit = _unit_vectors(targets.ra, targets.dec)

    self_pos = np.full(n_target, -1, dtype=np.intp)
    if self_index is not None:
        self_index = np.asarray(self_index, dtype=np.intp)
        if self_index.shape != (n_target,):
            raise ValueError("self_index must have one entry per target")
        has_self = self_index >= 0
        self_pos[has_self] = position[self_index[has_self]]

    cos_k = np.full(n_target, np.nan)
    for i in range(n_target):
        a, b = lo[i], hi[i]
        n = b - a
        if n < k:
            continue
        cos = unit[a:b] @ target_unit[i]
        if a <= self_pos[i] < b:
            if n - 1 < k:
                continue
            cos[self_pos[i] - a] = -2.0
        cos_k[i] = np.partition(cos, n - k)[n - k]
    return np.arccos(np.clip(cos_k, -1.0, 1.0))


@register("sigma2d")
def sigma2d_columns(tracers, targets, k, self_index=None, *, sigma=None,
                    dv=1500.0, n_sigma=1.5):
    """Projected k-th neighbour surface density.

    For each target, the tracers inside a redshift slice around the target's
    z are considered. R_k = d_com * theta_k, with d_com the target's
    comoving distance (norm of xyz) and theta_k the angular separation to
    the k-th nearest tracer in the slice, and sigma_k = k / (pi R_k^2).

    The slice half-width is dv (1 + z) / c for spec-z targets (dv in km/s)
    and n_sigma * sigma * (1 + z) for targets flagged is_phot.

    Returns {'r': R_k [Mpc], 'sigma': sigma_k [Mpc^-2], 'log1p_delta': ...}
    with log1p_delta = log10(sigma_k / median(sigma_k over the tracers)).
    The tracer median is computed with the same kind of slice as the target
    (spec slice for spec-z targets, photo-z slice for photo-z targets), so
    that the wider photo-z slice, which projects more galaxies, does not
    shift log1p_delta. Targets with fewer than k tracers in the slice get
    NaN.

    NOTE: the median normalisation is provisional, as in knn3d.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    d_target = np.linalg.norm(targets.xyz, axis=1)
    half = slice_half_width(targets.z, targets.is_phot, dv, n_sigma, sigma)
    theta = kth_angular_separation(tracers, targets, half, k, self_index=self_index)
    r_k = d_target * theta
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_k = k / (np.pi * r_k ** 2)

    n_tracer = len(tracers)
    d_tracer = np.linalg.norm(tracers.xyz, axis=1)
    all_tracers = np.arange(n_tracer)

    def tracer_median(is_phot):
        half_t = slice_half_width(tracers.z, np.full(n_tracer, is_phot), dv, n_sigma, sigma)
        th = kth_angular_separation(tracers, tracers, half_t, k, self_index=all_tracers)
        with np.errstate(divide="ignore", invalid="ignore"):
            values = k / (np.pi * (d_tracer * th) ** 2)
        return np.nanmedian(values) if np.isfinite(values).any() else np.nan

    norm_spec = tracer_median(False) if (~targets.is_phot).any() else np.nan
    norm_phot = tracer_median(True) if targets.is_phot.any() else np.nan
    norm = np.where(targets.is_phot, norm_phot, norm_spec)
    with np.errstate(divide="ignore", invalid="ignore"):
        log1p_delta = np.log10(sigma_k / norm)
    return {"r": r_k, "sigma": sigma_k, "log1p_delta": log1p_delta}
