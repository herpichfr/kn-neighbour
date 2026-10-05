"""Projected k-th neighbour surface density in a redshift slice."""

import numpy as np
from scipy.spatial import cKDTree

from . import register, tracer_masses

C_KMS = 299792.458
# kth_angular_neighbours uses the candidate search below from this many targets on
FAST_MIN_TARGETS = 2000
FAST_MAX_CANDIDATES = 4096
FAST_BLOCK = 1_000_000          # candidates (targets x m) handled per block


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


def _kth_by_candidates(unit, mass_sorted, target_unit, lo, hi, self_pos, k, cos_k, mass_k, done):
    """Fill cos_k and mass_k of the targets for which the m nearest tracers suffice.

    The m angularly nearest tracers of a target (k-d tree on the unit
    vectors) are filtered to its slice [lo, hi) minus itself. When at least k
    of them remain, their k-th largest cosine is exactly the k-th of the
    whole slice: a slice tracer closer than it would rank among the m nearest
    too. Targets that do not get k are retried with 4 times more candidates
    up to FAST_MAX_CANDIDATES and then left to the caller (done stays False).
    """
    n_tracer = len(unit)
    in_slice = (hi - lo) - ((self_pos >= lo) & (self_pos < hi))
    todo = np.flatnonzero(np.isfinite(target_unit).all(axis=1) & (in_slice >= k))
    if len(todo) == 0:
        return
    tree = cKDTree(unit)
    m = min(n_tracer, 8 * k + 8)
    while len(todo):
        block = max(1, FAST_BLOCK // m)
        retry = []
        for start in range(0, len(todo), block):
            sel = todo[start:start + block]
            _, idx = tree.query(target_unit[sel], k=m, workers=-1)
            idx = np.asarray(idx).reshape(len(sel), m)
            cos = np.einsum("ijk,ik->ij", unit[idx], target_unit[sel])
            ok = ((idx >= lo[sel, None]) & (idx < hi[sel, None])
                  & (idx != self_pos[sel, None]))
            enough = ok.sum(axis=1) >= k
            retry.append(sel[~enough])
            if not enough.any():
                continue
            cos = np.where(ok, cos, -np.inf)[enough]
            idx, rows = idx[enough], sel[enough]
            if mass_sorted is None:
                cos_k[rows] = np.partition(cos, m - k, axis=1)[:, m - k]
            else:
                nearest = np.argpartition(cos, m - k, axis=1)[:, m - k:]
                cos_k[rows] = np.take_along_axis(cos, nearest, axis=1).min(axis=1)
                mass_k[rows] = mass_sorted[np.take_along_axis(idx, nearest, axis=1)].sum(axis=1)
            done[rows] = True
        todo = np.concatenate(retry)
        if m >= min(n_tracer, FAST_MAX_CANDIDATES):
            break
        m = min(n_tracer, FAST_MAX_CANDIDATES, 4 * m)


def kth_angular_neighbours(tracers, targets, half_width, k, self_index=None, tracer_mass=None,
                           fast=None):
    """Angular separation (radians) to the k-th nearest tracer in a slice, and
    the summed mass of the k nearest tracers.

    The slice of target i holds the tracers with |z_tracer - z_i| <=
    half_width[i]. self_index[i] is the index in tracers of target i, or -1;
    that tracer is excluded by index. Targets with fewer than k tracers in
    the slice get NaN. tracer_mass, (N,) masses of the tracers, or None, in
    which case the mass sum is NaN.

    Tracers are sorted by redshift once and each slice is found with
    searchsorted. Targets are first handled by a nearest-candidates search
    (_kth_by_candidates: exact, same result up to rounding of the cosines)
    when fast is True, or None and there are at least FAST_MIN_TARGETS
    targets; the remaining targets (and all of them with fast=False) are
    handled one at a time, the angular distances of a target computed at
    once for the whole slice.

    Returns (theta_k, mass_k), each (M,).
    """
    n_target = len(targets)
    order = np.argsort(tracers.z, kind="stable")
    z_sorted = tracers.z[order]
    unit = _unit_vectors(tracers.ra[order], tracers.dec[order])
    mass_sorted = None if tracer_mass is None else np.asarray(tracer_mass, dtype=float)[order]
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
    mass_k = np.full(n_target, np.nan)
    done = np.zeros(n_target, dtype=bool)
    if fast is None:
        fast = n_target >= FAST_MIN_TARGETS
    if fast:
        _kth_by_candidates(unit, mass_sorted, target_unit, lo, hi, self_pos, k, cos_k, mass_k,
                           done)
    for i in range(n_target):
        if done[i]:
            continue
        a, b = lo[i], hi[i]
        n = b - a
        if n < k:
            continue
        cos = unit[a:b] @ target_unit[i]
        if a <= self_pos[i] < b:
            if n - 1 < k:
                continue
            cos[self_pos[i] - a] = -2.0
        if mass_sorted is None:
            cos_k[i] = np.partition(cos, n - k)[n - k]
        else:
            nearest = np.argpartition(cos, n - k)[n - k:]
            cos_k[i] = cos[nearest].min()
            mass_k[i] = mass_sorted[a:b][nearest].sum()
    return np.arccos(np.clip(cos_k, -1.0, 1.0)), mass_k


def kth_angular_separation(tracers, targets, half_width, k, self_index=None):
    """Angular separation (radians) to the k-th nearest tracer in a slice.

    See kth_angular_neighbours; returns only the separations.
    """
    return kth_angular_neighbours(tracers, targets, half_width, k, self_index)[0]


@register("sigma2d")
def sigma2d_columns(tracers, targets, k, self_index=None, *, sigma=None,
                    dv=1500.0, n_sigma=1.5, weight="count"):
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

    weight='mass' (tracers need Points.mass, linear Msun): sigma is the
    summed mass of the k nearest tracers in the slice per projected area,
    M_k / (pi R_k^2) in Msun Mpc^-2, and the median is that of the same
    quantity over the tracers.

    NOTE: the median normalisation is provisional, as in knn3d.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    mass = tracer_masses(tracers, weight)
    d_target = np.linalg.norm(targets.xyz, axis=1)
    half = slice_half_width(targets.z, targets.is_phot, dv, n_sigma, sigma)
    theta, mass_k = kth_angular_neighbours(tracers, targets, half, k, self_index,
                                           tracer_mass=mass)
    r_k = d_target * theta
    numerator = float(k) if mass is None else mass_k
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_k = numerator / (np.pi * r_k ** 2)

    n_tracer = len(tracers)
    d_tracer = np.linalg.norm(tracers.xyz, axis=1)
    all_tracers = np.arange(n_tracer)

    def tracer_median(is_phot):
        half_t = slice_half_width(tracers.z, np.full(n_tracer, is_phot), dv, n_sigma, sigma)
        th, m_t = kth_angular_neighbours(tracers, tracers, half_t, k, all_tracers,
                                         tracer_mass=mass)
        num = float(k) if mass is None else m_t
        with np.errstate(divide="ignore", invalid="ignore"):
            values = num / (np.pi * (d_tracer * th) ** 2)
        return np.nanmedian(values) if np.isfinite(values).any() else np.nan

    norm_spec = tracer_median(False) if (~targets.is_phot).any() else np.nan
    norm_phot = tracer_median(True) if targets.is_phot.any() else np.nan
    norm = np.where(targets.is_phot, norm_phot, norm_spec)
    with np.errstate(divide="ignore", invalid="ignore"):
        log1p_delta = np.log10(sigma_k / norm)
    return {"r": r_k, "sigma": sigma_k, "log1p_delta": log1p_delta}
