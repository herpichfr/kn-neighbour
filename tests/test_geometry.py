import astropy.units as u
import numpy as np

from knn.geometry import comoving_xyz, make_cosmology


def _cosmo():
    return make_cosmology({"cosmology": {"H0": 70, "Om0": 0.3}})


def test_norm_equals_comoving_distance():
    cosmo = _cosmo()
    rng = np.random.default_rng(1)
    ra = rng.uniform(0, 360, 50)
    dec = rng.uniform(-90, 90, 50)
    z = rng.uniform(0.005, 0.5, 50)
    xyz = comoving_xyz(ra, dec, z, cosmo)
    assert xyz.shape == (50, 3)
    expected = cosmo.comoving_distance(z).to_value(u.Mpc)
    np.testing.assert_allclose(np.linalg.norm(xyz, axis=1), expected, rtol=1e-12)


def test_axes():
    cosmo = _cosmo()
    d = cosmo.comoving_distance(0.02).to_value(u.Mpc)
    xyz = comoving_xyz([0.0, 90.0, 0.0], [0.0, 0.0, 90.0], [0.02] * 3, cosmo)
    np.testing.assert_allclose(xyz[0], [d, 0, 0], atol=1e-9)
    np.testing.assert_allclose(xyz[1], [0, d, 0], atol=1e-9)
    np.testing.assert_allclose(xyz[2], [0, 0, d], atol=1e-9)
