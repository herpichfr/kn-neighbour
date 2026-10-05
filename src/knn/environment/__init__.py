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

The dict maps a quantity name ('r', 'rho', 'log1p_delta', ...) to an array
with one value per target. The build step names the columns
'<quantity>_<name>_k<k>'.
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
    """

    xyz: np.ndarray
    ra: np.ndarray
    dec: np.ndarray
    z: np.ndarray
    is_phot: Optional[np.ndarray] = None

    def __post_init__(self):
        self.xyz = np.asarray(self.xyz, dtype=float).reshape(-1, 3)
        self.ra = np.asarray(self.ra, dtype=float)
        self.dec = np.asarray(self.dec, dtype=float)
        self.z = np.asarray(self.z, dtype=float)
        n = len(self.xyz)
        if self.is_phot is None:
            self.is_phot = np.zeros(n, dtype=bool)
        self.is_phot = np.asarray(self.is_phot, dtype=bool)
        for name in ("ra", "dec", "z", "is_phot"):
            if getattr(self, name).shape != (n,):
                raise ValueError(f"'{name}' must have one entry per point")

    def __len__(self):
        return len(self.xyz)


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
