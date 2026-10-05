"""Cosmology and comoving coordinates."""

import astropy.units as u
import numpy as np
from astropy.cosmology import FlatLambdaCDM


def make_cosmology(cfg):
    """Flat LambdaCDM cosmology from cfg['cosmology'] = {H0, Om0}."""
    c = cfg["cosmology"]
    return FlatLambdaCDM(H0=c["H0"], Om0=c["Om0"])


def comoving_distance(z, cosmo):
    """Comoving distance in Mpc, as a float array."""
    return np.asarray(cosmo.comoving_distance(np.asarray(z, dtype=float)).to_value(u.Mpc))


def comoving_xyz(ra, dec, z, cosmo):
    """Cartesian comoving coordinates in Mpc, shape (N, 3).

    ra, dec in degrees. x points to (ra, dec) = (0, 0), y to (90, 0),
    z to the north celestial pole.
    """
    ra = np.radians(np.asarray(ra, dtype=float))
    dec = np.radians(np.asarray(dec, dtype=float))
    d = comoving_distance(z, cosmo)
    return np.column_stack([d * np.cos(dec) * np.cos(ra),
                            d * np.cos(dec) * np.sin(ra),
                            d * np.sin(dec)])
