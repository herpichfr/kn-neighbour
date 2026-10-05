import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from knn.redshift import (assign_redshifts, photoz_draws, photoz_point_stats, pit,
                          pit_gauss)


def test_assign_redshifts():
    df = pd.DataFrame({
        "z_spec": [0.02, 0.05, np.nan, np.nan, 0.02],
        "z_phot": [0.03, 0.02, 0.02, np.nan, np.nan],
    })
    out = assign_redshifts(df, [0.007, 0.037])
    np.testing.assert_allclose(out["z_best"], [0.02, 0.05, 0.02, np.nan, 0.02])
    assert list(out["z_type"][[0, 1, 2, 4]]) == ["spec", "spec", "phot", "spec"]
    assert pd.isna(out["z_type"][3])
    # spec-z outside the window excludes the galaxy even if z_phot is inside
    assert list(out["in_window"]) == [True, False, True, False, True]
    assert out["in_window"].dtype == bool


def _mixture_frame(weights, means, stds, z_phot=0.02):
    cols = {"z_phot": [z_phot]}
    for j in range(3):
        cols[f"pdf_weights_{j}"] = [weights[j]]
        cols[f"pdf_means_{j}"] = [means[j]]
        cols[f"pdf_stds_{j}"] = [stds[j]]
    return pd.DataFrame(cols)


def test_photoz_draws_point_shape():
    df = pd.DataFrame({"z_phot": [0.01, 0.02, 0.03]})
    z = photoz_draws(df, "point", 200, np.random.default_rng(0))
    assert z.shape == (3, 1)
    np.testing.assert_array_equal(z[:, 0], df["z_phot"])


def test_photoz_draws_gauss_moments():
    zp = 0.03
    df = pd.DataFrame({"z_phot": [zp, 0.01]})
    sigma, bias = 0.012, -0.004
    z = photoz_draws(df, "gauss", 200000, np.random.default_rng(1), sigma=sigma, bias=bias)
    assert z.shape == (2, 200000)
    np.testing.assert_allclose(z[0].mean(), zp - bias * (1 + zp), atol=5e-5)
    np.testing.assert_allclose(z[0].std(), sigma * (1 + zp), rtol=0.01)
    np.testing.assert_allclose(z[1].std(), sigma * 1.01, rtol=0.01)
    with pytest.raises(ValueError, match="sigma"):
        photoz_draws(df, "gauss", 10, np.random.default_rng(1))


def test_photoz_draws_mixture_moments():
    w, m, s = [0.3, 0.7, 0.0], [0.02, 0.03, 0.5], [0.004, 0.002, 0.001]
    df = _mixture_frame(w, m, s)
    z = photoz_draws(df, "pdf", 400000, np.random.default_rng(2))[0]
    mean = 0.3 * 0.02 + 0.7 * 0.03
    var = 0.3 * (0.004 ** 2 + 0.02 ** 2) + 0.7 * (0.002 ** 2 + 0.03 ** 2) - mean ** 2
    np.testing.assert_allclose(z.mean(), mean, atol=2e-5)
    np.testing.assert_allclose(z.std(), np.sqrt(var), rtol=0.01)
    expected = 0.3 * norm.cdf((0.025 - 0.02) / 0.004) + 0.7 * norm.cdf((0.025 - 0.03) / 0.002)
    assert abs((z < 0.025).mean() - expected) < 0.005


def test_photoz_draws_drops_wide_component():
    # weights sum to 0.97: renormalised; the wide component is dropped
    df = _mixture_frame([0.5, 0.47, 0.0], [0.02, 0.5, 0.0], [0.003, 3.0, 1.0])
    z = photoz_draws(df, "pdf", 20000, np.random.default_rng(3), pdf_max_std=0.5)[0]
    np.testing.assert_allclose(z.mean(), 0.02, atol=1e-4)
    np.testing.assert_allclose(z.std(), 0.003, rtol=0.03)
    # a single wide component: no usable component, NaN row
    df = _mixture_frame([1.0, 0.0, 0.0], [0.02, 0.0, 0.0], [258.0, 1.0, 1.0])
    z = photoz_draws(df, "pdf", 50, np.random.default_rng(3), pdf_max_std=0.5)
    assert z.shape == (1, 50) and np.isnan(z).all()
    assert np.isnan(pit(df, [0.02], pdf_max_std=0.5)).all()


def test_pit_uniform_for_calibrated_mixture():
    rng = np.random.default_rng(4)
    n = 5000
    df = pd.DataFrame({"z_phot": np.full(n, 0.02)})
    for j, (w, m, s) in enumerate([(0.4, 0.02, 0.004), (0.6, 0.03, 0.006), (0.0, 0.0, 1.0)]):
        df[f"pdf_weights_{j}"], df[f"pdf_means_{j}"], df[f"pdf_stds_{j}"] = w, m, s
    z_true = photoz_draws(df, "pdf", 1, rng)[:, 0]
    counts, _ = np.histogram(pit(df, z_true), bins=5, range=(0, 1))
    assert counts.min() > 0.85 * n / 5 and counts.max() < 1.15 * n / 5
    p = pit_gauss(np.full(n, 0.02), 0.02 + 0.01 * rng.standard_normal(n), 0.01 / 1.02, 0.0)
    counts, _ = np.histogram(p, bins=5, range=(0, 1))
    assert counts.min() > 0.85 * n / 5 and counts.max() < 1.15 * n / 5


def test_photoz_point_stats():
    rng = np.random.default_rng(5)
    zs = rng.uniform(0.01, 0.03, 20000)
    zp = zs + (1 + zs) * (-0.004 + 0.013 * rng.standard_normal(zs.size))
    zp[:10] = np.nan
    bias, sigma = photoz_point_stats(zp, zs)
    assert abs(bias + 0.004) < 5e-4
    assert abs(sigma - 0.013) < 5e-4
    assert all(np.isnan(photoz_point_stats([np.nan], [0.02])))
