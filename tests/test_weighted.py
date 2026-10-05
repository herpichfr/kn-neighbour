import numpy as np
import pytest

from knn.environment import Points, get_estimator
from knn.environment.knn3d import knn3d, knn3d_mass
from knn.environment.sigma2d import kth_angular_neighbours, kth_angular_separation
from knn.geometry import comoving_xyz, make_cosmology

COSMO = make_cosmology({"cosmology": {"H0": 70, "Om0": 0.3}})
VOL = 4.0 / 3.0 * np.pi

# tracers on a line, no distance ties; k = 2, self excluded:
#   x = 0:  nearest 1 (m2), 3 (m4)  -> M = 6,  r = 3
#   x = 1:  nearest 0 (m1), 3 (m4)  -> M = 5,  r = 2
#   x = 3:  nearest 1 (m2), 0 (m1)  -> M = 3,  r = 3
#   x = 7:  nearest 3 (m4), 12 (m16) -> M = 20, r = 5
#   x = 12: nearest 7 (m8), 3 (m4)  -> M = 12, r = 9
XS = np.array([0.0, 1.0, 3.0, 7.0, 12.0])
MASS = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
M_K = np.array([6.0, 5.0, 3.0, 20.0, 12.0])
R_K = np.array([3.0, 2.0, 3.0, 5.0, 9.0])


def _line_points():
    xyz = np.column_stack([XS, np.zeros(5), np.zeros(5)])
    return Points(xyz, XS, np.zeros(5), np.full(5, 0.02), mass=MASS)


def test_knn3d_mass_known_answer():
    xyz = _line_points().xyz
    r, rho = knn3d_mass(xyz, MASS, xyz, 2, self_index=np.arange(5))
    np.testing.assert_allclose(r, R_K)
    np.testing.assert_allclose(rho, M_K / (VOL * R_K ** 3))
    # with unit masses the mass density is the count density
    r1, rho1 = knn3d_mass(xyz, np.ones(5), xyz, 2, self_index=np.arange(5))
    r0, rho0 = knn3d(xyz, xyz, 2, self_index=np.arange(5))
    np.testing.assert_allclose(rho1, rho0)
    np.testing.assert_allclose(r1, r0)


def test_knn3d_weighted_estimator_and_normalisation():
    tr = _line_points()
    est = get_estimator("knn3d")
    res = est(tr, tr, 2, self_index=np.arange(5), weight="mass")
    rho = M_K / (VOL * R_K ** 3)
    np.testing.assert_allclose(res["r"], R_K)
    np.testing.assert_allclose(res["rho"], rho)
    # median of the tracers' own mass densities is the x = 7 value
    np.testing.assert_allclose(np.median(rho), rho[3])
    np.testing.assert_allclose(res["log1p_delta"], np.log10(rho / rho[3]))
    # the count version is unchanged by the new parameter
    a = est(tr, tr, 2, self_index=np.arange(5))
    b = est(tr, tr, 2, self_index=np.arange(5), weight="count")
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])


def test_weighted_estimators_need_masses():
    tr = _line_points()
    no_mass = Points(tr.xyz, tr.ra, tr.dec, tr.z)
    bad = Points(tr.xyz, tr.ra, tr.dec, tr.z, mass=np.array([1.0, np.nan, 1.0, 1.0, 1.0]))
    for name in ("knn3d", "sigma2d"):
        est = get_estimator(name)
        with pytest.raises(ValueError, match="mass"):
            est(no_mass, no_mass, 2, self_index=np.arange(5), weight="mass")
        with pytest.raises(ValueError, match="mass"):
            est(bad, bad, 2, self_index=np.arange(5), weight="mass")
        with pytest.raises(ValueError, match="weight"):
            est(tr, tr, 2, self_index=np.arange(5), weight="luminosity")


def test_sigma2d_mass_known_answer():
    # tracers on the equator at one redshift, separations equal to the ra differences
    ra = np.array([0.0, 0.1, 0.3, 0.55, 1.1])
    z = np.full(5, 0.02)
    xyz = comoving_xyz(ra, np.zeros(5), z, COSMO)
    tr = Points(xyz, ra, np.zeros(5), z, mass=MASS)
    half = np.full(5, 0.001)
    theta, msum = kth_angular_neighbours(tr, tr, half, 2, self_index=np.arange(5),
                                         tracer_mass=MASS)
    # ra 0: 0.1 (m2), 0.3 (m4) -> 6, 0.3 deg;  0.1: 0 (m1), 0.3 (m4) -> 5, 0.2 deg
    # 0.3: 0.1 (m2), 0.55 (m8) -> 10, 0.25 deg;  0.55: 0.3 (m4), 0.1 (m2) -> 6, 0.45 deg
    # 1.1: 0.55 (m8), 0.3 (m4) -> 12, 0.8 deg
    np.testing.assert_allclose(np.degrees(theta), [0.3, 0.2, 0.25, 0.45, 0.8], atol=1e-9)
    np.testing.assert_allclose(msum, [6.0, 5.0, 10.0, 6.0, 12.0])
    np.testing.assert_allclose(
        np.degrees(kth_angular_separation(tr, tr, half, 2, self_index=np.arange(5))),
        [0.3, 0.2, 0.25, 0.45, 0.8], atol=1e-9)
    # without masses the mass sum is NaN and nothing else changes
    theta0, msum0 = kth_angular_neighbours(tr, tr, half, 2, self_index=np.arange(5))
    np.testing.assert_allclose(theta0, theta)
    assert np.isnan(msum0).all()

    est = get_estimator("sigma2d")
    res = est(tr, tr, 2, self_index=np.arange(5), dv=100.0, weight="mass")
    d = np.linalg.norm(xyz, axis=1)
    r = d * theta
    sigma = np.array([6.0, 5.0, 10.0, 6.0, 12.0]) / (np.pi * r ** 2)
    np.testing.assert_allclose(res["r"], r)
    np.testing.assert_allclose(res["sigma"], sigma)
    np.testing.assert_allclose(res["log1p_delta"], np.log10(sigma / np.median(sigma)))
    # unit masses give the count version
    unit = Points(xyz, ra, np.zeros(5), z, mass=np.ones(5))
    res_u = est(unit, tr, 2, self_index=np.arange(5), dv=100.0, weight="mass")
    res_c = est(tr, tr, 2, self_index=np.arange(5), dv=100.0)
    np.testing.assert_allclose(res_u["sigma"], res_c["sigma"])
    np.testing.assert_allclose(res_u["log1p_delta"], res_c["log1p_delta"])


def test_sigma2d_mass_slice_excludes_other_redshifts():
    ra = np.array([0.0, 0.1, 0.2, 0.05])
    z = np.array([0.02, 0.02, 0.02, 0.05])      # the last tracer is far in z
    mass = np.array([1.0, 2.0, 4.0, 1000.0])
    xyz = comoving_xyz(ra, np.zeros(4), z, COSMO)
    tr = Points(xyz, ra, np.zeros(4), z, mass=mass)
    theta, msum = kth_angular_neighbours(tr, tr, np.full(4, 0.001), 2, self_index=np.arange(4),
                                         tracer_mass=mass)
    np.testing.assert_allclose(msum[:3], [6.0, 5.0, 3.0])    # the 1000 Msun tracer never counts
    assert np.isnan(msum[3])
