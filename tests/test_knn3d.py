import numpy as np
from scipy.special import gamma

from knn.environment.knn3d import knn3d


def test_poisson_mean_rk():
    """Mean r_k for a uniform random field matches the analytic Poisson value."""
    rng = np.random.default_rng(12345)
    n_pts, box, k = 20000, 1.0, 10
    pts = rng.uniform(0.0, box, size=(n_pts, 3))
    density = n_pts / box ** 3

    expected = gamma(k + 1.0 / 3.0) / gamma(k) * (3.0 / (4.0 * np.pi * density)) ** (1.0 / 3.0)

    # only targets farther than 3 expected r_k from every face
    margin = 3.0 * expected
    inside = np.all((pts > margin) & (pts < box - margin), axis=1)
    assert inside.sum() > 1000
    targets = np.flatnonzero(inside)

    r_k, rho_k = knn3d(pts, pts[targets], k, self_index=targets)
    assert abs(r_k.mean() / expected - 1.0) < 0.02
    np.testing.assert_allclose(rho_k, k / (4.0 / 3.0 * np.pi * r_k ** 3))

    # same estimate for non-tracer targets, no self exclusion
    random_targets = rng.uniform(margin, box - margin, size=(5000, 3))
    r_k2, _ = knn3d(pts, random_targets, k)
    assert abs(r_k2.mean() / expected - 1.0) < 0.02


def test_self_exclusion_with_coincident_points():
    # A=0, B=1, C=D=3 (coincident), E=10 along x
    xs = np.array([0.0, 1.0, 3.0, 3.0, 10.0])
    pts = np.column_stack([xs, np.zeros(5), np.zeros(5)])
    self_index = np.arange(5)

    r1, rho1 = knn3d(pts, pts, 1, self_index=self_index)
    # C and D each see the other at distance 0
    np.testing.assert_allclose(r1, [1.0, 1.0, 0.0, 0.0, 7.0])
    assert np.isinf(rho1[2]) and np.isinf(rho1[3])
    np.testing.assert_allclose(rho1[0], 1.0 / (4.0 / 3.0 * np.pi))

    r2, _ = knn3d(pts, pts, 2, self_index=self_index)
    np.testing.assert_allclose(r2, [3.0, 2.0, 2.0, 2.0, 7.0])

    r3, _ = knn3d(pts, pts, 3, self_index=self_index)
    np.testing.assert_allclose(r3, [3.0, 2.0, 3.0, 3.0, 9.0])


def test_without_self_index_target_sees_itself():
    xs = np.array([0.0, 1.0, 3.0])
    pts = np.column_stack([xs, np.zeros(3), np.zeros(3)])
    r1, _ = knn3d(pts, pts, 1)
    np.testing.assert_allclose(r1, 0.0)


def test_mixed_self_index():
    xs = np.array([0.0, 1.0, 3.0])
    pts = np.column_stack([xs, np.zeros(3), np.zeros(3)])
    targets = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    # first target is tracer 0, second is not a tracer
    r1, _ = knn3d(pts, targets, 1, self_index=np.array([0, -1]))
    np.testing.assert_allclose(r1, [1.0, 0.5])


def test_more_coincident_points_than_k_plus_one():
    # five coincident tracers, k=1: the self hit may not be among the 2 returned
    pts = np.zeros((5, 3))
    pts[:, 0] = [2.0, 2.0, 2.0, 2.0, 2.0]
    r1, _ = knn3d(pts, pts, 1, self_index=np.arange(5))
    np.testing.assert_allclose(r1, 0.0)
