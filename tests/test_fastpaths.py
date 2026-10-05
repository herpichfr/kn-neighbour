"""The speed-ups used by the explorer must not change the results."""

import warnings

import numpy as np
import pytest

from knn.environment import Points
from knn.environment.sigma2d import kth_angular_neighbours
from knn.photoz_env import nanpercentiles


def _points(n, seed=1):
    rng = np.random.default_rng(seed)
    ra, dec = rng.uniform(0, 3, n), rng.uniform(0, 3, n)
    z = rng.uniform(0.01, 0.03, n)
    return Points(np.zeros((n, 3)), ra, dec, z), 10 ** rng.normal(9, 1, n), rng


@pytest.mark.parametrize("half", [0.0017, 0.0099, 0.00005])      # thin slices need retries
@pytest.mark.parametrize("weighted", [False, True])
def test_candidate_search_equals_the_brute_force(half, weighted):
    pts, mass, rng = _points(3000)
    n = len(pts)
    half_width = np.where(rng.random(n) < 0.5, half, 2 * half)
    own = np.arange(n)
    m = mass if weighted else None
    slow = kth_angular_neighbours(pts, pts, half_width, 7, own, m, fast=False)
    fast = kth_angular_neighbours(pts, pts, half_width, 7, own, m, fast=True)
    assert np.array_equal(np.isnan(slow[0]), np.isnan(fast[0]))
    np.testing.assert_allclose(fast[0], slow[0], rtol=1e-8, equal_nan=True)
    if weighted:
        np.testing.assert_allclose(fast[1], slow[1], rtol=1e-12, equal_nan=True)


def test_candidate_search_without_self_index_and_few_tracers():
    pts, _, _ = _points(400, seed=3)
    targets = Points(np.zeros((50, 3)), pts.ra[:50] + 0.01, pts.dec[:50], pts.z[:50])
    half = np.full(50, 0.004)
    a = kth_angular_neighbours(pts, targets, half, 5, None, fast=False)[0]
    b = kth_angular_neighbours(pts, targets, half, 5, None, fast=True)[0]
    np.testing.assert_allclose(b, a, rtol=1e-8, equal_nan=True)
    # k larger than any slice: all NaN either way
    assert np.isnan(kth_angular_neighbours(pts, targets, half, 300, None, fast=True)[0]).all()


def test_fast_path_is_used_from_the_threshold(monkeypatch):
    import knn.environment.sigma2d as s2
    calls = []
    real = s2._kth_by_candidates
    monkeypatch.setattr(s2, "_kth_by_candidates", lambda *a: calls.append(1) or real(*a))
    pts, _, _ = _points(300)
    kth_angular_neighbours(pts, pts, np.full(300, 0.005), 5, np.arange(300))
    assert not calls
    monkeypatch.setattr(s2, "FAST_MIN_TARGETS", 100)
    kth_angular_neighbours(pts, pts, np.full(300, 0.005), 5, np.arange(300))
    assert calls


@pytest.mark.parametrize("n_draws", [1, 2, 7, 200])
def test_nanpercentiles_equals_numpy(n_draws):
    rng = np.random.default_rng(0)
    m = rng.normal(size=(400, n_draws))
    m[rng.random(m.shape) < 0.4] = np.nan
    m[3] = np.nan
    m[4] = 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = np.nanpercentile(m, (16.0, 50.0, 84.0), axis=1)
    got = nanpercentiles(m)
    np.testing.assert_array_equal(got, ref)
