"""Environment of photo-z galaxies from redshift draws."""

import warnings
from dataclasses import dataclass

import numpy as np

from .environment import Points
from .geometry import comoving_xyz
from .redshift import photoz_draws

PERCENTILES = (16.0, 50.0, 84.0)


@dataclass
class DrawSet:
    """Redshift draws of N galaxies, restricted to the redshift window.

    inside : (N, n_draws) bool, True for the draws kept
    gal_idx, z, ra, dec, xyz : one entry per kept draw, in the row-major
        order of 'inside'; gal_idx is the galaxy's row index
    p_in_window : (N,) fraction of the draws inside the window; NaN where
        every draw is NaN (no usable photo-z PDF)
    """

    mode: str
    inside: np.ndarray
    gal_idx: np.ndarray
    z: np.ndarray
    ra: np.ndarray
    dec: np.ndarray
    xyz: np.ndarray
    p_in_window: np.ndarray


def make_draws(df, mode, settings, params, window, cosmo):
    """Draw redshifts for the rows of df and keep those inside the window.

    settings is the 'photoz' config section (n_draws, seed, pdf_max_std),
    params holds the resolved sigma and bias (resolve_photoz_params). The
    random generator is seeded with settings['seed'], so the draws are
    reproducible. Comoving coordinates are computed for the kept draws only.
    """
    rng = np.random.default_rng(settings["seed"])
    z_all = photoz_draws(df, mode, settings["n_draws"], rng, sigma=params.get("sigma"),
                         bias=params.get("bias"), pdf_max_std=settings["pdf_max_std"])
    with np.errstate(invalid="ignore"):
        inside = (z_all > window[0]) & (z_all < window[1])
    gal_idx = np.broadcast_to(np.arange(len(df))[:, None], z_all.shape)[inside]
    ra = df["ra"].to_numpy(float)[gal_idx]
    dec = df["dec"].to_numpy(float)[gal_idx]
    z = z_all[inside]
    p_in_window = inside.mean(axis=1)
    p_in_window[np.isnan(z_all).all(axis=1)] = np.nan
    return DrawSet(mode, inside, gal_idx, z, ra, dec,
                   comoving_xyz(ra, dec, z, cosmo), p_in_window)


def summarise(values, draws):
    """Per galaxy 16th, 50th and 84th percentile of values over its kept draws.

    values has one entry per kept draw. Returns (lo, median, hi), each (N,);
    NaN for a galaxy with no kept draw.
    """
    matrix = np.full(draws.inside.shape, np.nan)
    matrix[draws.inside] = values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        lo, med, hi = np.nanpercentile(matrix, PERCENTILES, axis=1)
    return lo, med, hi


def photoz_environment(estimator, tracers, tracer_index, draws, k, quantities,
                       sigma=None, **params):
    """Run an estimator on the draws of photo-z galaxies and summarise it.

    tracer_index[g] is the index in tracers of galaxy g, or -1 if it is not
    a tracer (it is then not excluded from its own neighbours; a galaxy that
    is a tracer is excluded at every draw, by index). quantities lists the
    estimator's output names, used when no draw is inside the window.

    Returns {quantity: (lo, median, hi)}, each array of length N galaxies.
    """
    n_gal = len(draws.inside)
    if len(draws.z) == 0:
        nan = np.full(n_gal, np.nan)
        return {q: (nan, nan, nan) for q in quantities}
    targets = Points(draws.xyz, draws.ra, draws.dec, draws.z,
                     is_phot=np.ones(len(draws.z), dtype=bool))
    result = estimator(tracers, targets, k, self_index=np.asarray(tracer_index)[draws.gal_idx],
                       sigma=sigma, **params)
    return {q: summarise(v, draws) for q, v in result.items()}
