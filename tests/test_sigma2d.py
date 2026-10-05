import numpy as np
import pytest
from scipy.special import gamma

from knn.environment import Points, get_estimator
from knn.environment.sigma2d import C_KMS, kth_angular_separation, slice_half_width
from knn.geometry import comoving_xyz, make_cosmology

COSMO = make_cosmology({"cosmology": {"H0": 70, "Om0": 0.3}})


def _points(ra, dec, z):
    return Points(comoving_xyz(ra, dec, z, COSMO), ra, dec, z)


def test_poisson_mean_rk():
    """Mean R_k of a uniform patch in a thin shell matches Gamma(k+1/2)/Gamma(k)/sqrt(pi n)."""
    rng = np.random.default_rng(11)
    n_pts, k, width_deg = 20000, 10, 6.0
    # uniform on the sphere: ra uniform, sin(dec) uniform, small patch around (0, 0)
    ra = rng.uniform(0.0, width_deg, n_pts)
    s = np.sin(np.radians(width_deg / 2.0))
    dec = np.degrees(np.arcsin(rng.uniform(-s, s, n_pts)))
    z = 0.02 + rng.uniform(-1e-4, 1e-4, n_pts)
    area = np.radians(width_deg) * 2.0 * s            # steradian
    n_ang = n_pts / area

    exp_theta = gamma(k + 0.5) / gamma(k) / np.sqrt(np.pi * n_ang)
    margin = np.degrees(3.0 * exp_theta)
    inside = ((ra > margin) & (ra < width_deg - margin)
              & (np.abs(dec) < width_deg / 2.0 - margin))
    targets = np.flatnonzero(inside)
    assert targets.size > 1000

    pts = _points(ra, dec, z)
    sub = Points(pts.xyz[targets], ra[targets], dec[targets], z[targets])
    res = get_estimator("sigma2d")(pts, sub, k, self_index=targets, dv=3000.0)
    d = np.linalg.norm(sub.xyz, axis=1)
    mean_theta = np.mean(res["r"] / d)
    assert abs(mean_theta / exp_theta - 1.0) < 0.03
    np.testing.assert_allclose(res["sigma"], k / (np.pi * res["r"] ** 2))


def test_self_exclusion_and_slice():
    # tracers on a line in ra at dec = 0, all at the same z, plus one far in z
    ra = np.array([0.0, 0.1, 0.3, 0.3, 1.0, 0.05])
    z = np.array([0.02, 0.02, 0.02, 0.02, 0.02, 0.05])
    dec = np.zeros(6)
    pts = _points(ra, dec, z)
    half = np.full(6, 0.001)
    theta1 = kth_angular_separation(pts, pts, half, 1, self_index=np.arange(6))
    np.testing.assert_allclose(np.degrees(theta1[:5]), [0.1, 0.1, 0.0, 0.0, 0.7], atol=1e-6)
    assert np.isnan(theta1[5])                  # alone in its slice
    theta2 = kth_angular_separation(pts, pts, half, 2, self_index=np.arange(6))
    np.testing.assert_allclose(np.degrees(theta2[:5]), [0.3, 0.2, 0.2, 0.2, 0.7], atol=1e-6)
    # without self_index the target sees itself at distance 0
    theta0 = kth_angular_separation(pts, pts, half, 1)
    np.testing.assert_allclose(theta0[:5], 0.0, atol=1e-6)
    # a larger slice reaches the tracer at z = 0.05
    big = kth_angular_separation(pts, pts, np.full(6, 0.04), 1, self_index=np.arange(6))
    assert np.isfinite(big[5])


def test_registered_estimator_slice_widths():
    rng = np.random.default_rng(2)
    n = 400
    ra, dec = rng.uniform(0, 1, n), rng.uniform(0, 1, n)
    z = rng.uniform(0.01, 0.03, n)
    pts = _points(ra, dec, z)
    est = get_estimator("sigma2d")
    # photo-z targets need sigma
    phot = Points(pts.xyz[:5], ra[:5], dec[:5], z[:5], is_phot=np.ones(5, bool))
    with pytest.raises(ValueError, match="sigma"):
        est(pts, phot, 5)
    res_phot = est(pts, phot, 5, sigma=0.01)
    spec = Points(pts.xyz[:5], ra[:5], dec[:5], z[:5])
    res_spec = est(pts, spec, 5, sigma=0.01)
    # the photo-z slice (1.5 * 0.01 * (1+z) = 0.015) is wider than the spec slice
    # (1500 km/s = 0.005): more tracers project in, so R_k is smaller
    assert np.all(res_phot["r"] <= res_spec["r"])
    assert np.all(np.isfinite(res_phot["log1p_delta"]))
    # default slice widths
    np.testing.assert_allclose(
        slice_half_width(0.02, False, 1500.0, 1.5, None), 1500.0 * 1.02 / C_KMS)
    np.testing.assert_allclose(slice_half_width(0.02, True, 1500.0, 1.5, 0.01), 1.5 * 0.01 * 1.02)


def test_speed():
    import time
    rng = np.random.default_rng(3)
    n = 5000
    ra, dec = rng.uniform(180, 184, n), rng.uniform(-2, 2, n)
    z = rng.uniform(0.007, 0.037, n)
    pts = _points(ra, dec, z)
    t0 = time.time()
    res = get_estimator("sigma2d")(pts, pts, 10, self_index=np.arange(n))
    assert np.isfinite(res["log1p_delta"]).all()
    assert time.time() - t0 < 20.0
