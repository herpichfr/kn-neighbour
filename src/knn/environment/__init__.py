"""Environment estimators.

Each estimator is a callable registered under a name with the same signature:

    estimator(tracers, targets, k, self_index=None, *, sigma=None, **params) -> dict

tracers and targets are Points. self_index is an optional (M,) integer array:
self_index[i] is the index in the tracers of target i, or -1 if target i is
not a tracer; that tracer is excluded from the target's neighbours. sigma is
the photo-z scatter, in units of (z_phot - z_spec) / (1 + z_spec); it is
used by estimators with a photo-z dependent slice and ignored by the
others. params are estimator specific options from the config (for
example dv and n_sigma for 'sigma2d'); an unknown option raises TypeError.

Every estimator takes weight='count' (default) or 'mass'. With 'mass' the
tracers must carry a finite stellar mass in Points.mass (linear, Msun): the
density is the summed mass of the k nearest tracers per volume (knn3d) or per
area (sigma2d) instead of k per volume or area.

The dict maps a quantity name ('r', 'rho', 'log1p_delta', ...) to an array
with one value per target. The build step names the columns
'<quantity>_<name>_k<k>', with '_mass' appended for weight='mass'.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

ESTIMATORS = {}


@dataclass
class Points:
    """A set of galaxies, as arrays of equal length.

    xyz : (N, 3) comoving Mpc
    ra, dec : degrees
    z : redshift used for the position (spec-z, photo-z or a photo-z draw)
    is_phot : (N,) bool, True where z is a photo-z value; default all False
    mass : (N,) stellar mass in Msun (linear), or None. Only the tracers of a
        mass-weighted estimator need it, and it must be finite there.
    """

    xyz: np.ndarray
    ra: np.ndarray
    dec: np.ndarray
    z: np.ndarray
    is_phot: Optional[np.ndarray] = None
    mass: Optional[np.ndarray] = None

    def __post_init__(self):
        self.xyz = np.asarray(self.xyz, dtype=float).reshape(-1, 3)
        self.ra = np.asarray(self.ra, dtype=float)
        self.dec = np.asarray(self.dec, dtype=float)
        self.z = np.asarray(self.z, dtype=float)
        n = len(self.xyz)
        if self.is_phot is None:
            self.is_phot = np.zeros(n, dtype=bool)
        self.is_phot = np.asarray(self.is_phot, dtype=bool)
        names = ["ra", "dec", "z", "is_phot"]
        if self.mass is not None:
            self.mass = np.asarray(self.mass, dtype=float)
            names.append("mass")
        for name in names:
            if getattr(self, name).shape != (n,):
                raise ValueError(f"'{name}' must have one entry per point")

    def __len__(self):
        return len(self.xyz)


WEIGHTS = ("count", "mass")


def tracer_masses(tracers, weight):
    """Linear stellar masses of the tracers for a weight, or None for 'count'.

    Raises ValueError for an unknown weight or for weight='mass' when a
    tracer has no finite mass.
    """
    if weight not in WEIGHTS:
        raise ValueError(f"unknown weight {weight!r}; choose from {WEIGHTS}")
    if weight == "count":
        return None
    mass = tracers.mass
    if mass is None or not np.isfinite(mass).all():
        raise ValueError("weight='mass' needs a finite stellar mass for every tracer "
                         "(exclude tracers without a mass from the tracer set)")
    return mass


def register(name):
    """Decorator registering an estimator under 'name'."""
    def decorator(func):
        if name in ESTIMATORS:
            raise ValueError(f"estimator '{name}' is already registered")
        ESTIMATORS[name] = func
        return func
    return decorator


def get_estimator(name):
    """Look up a registered estimator, with a clear error for unknown names."""
    try:
        return ESTIMATORS[name]
    except KeyError:
        raise KeyError(
            f"unknown environment estimator '{name}'; available: {sorted(ESTIMATORS)}"
        ) from None


from . import knn3d  # noqa: E402,F401  (registers 'knn3d')
from . import sigma2d  # noqa: E402,F401  (registers 'sigma2d')
