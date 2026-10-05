import numpy as np
import pandas as pd

from knn.indicators import (compute_indicators, fit_red_sequence, indicator_settings,
                            red_sequence_colour)


def _synthetic(seed=0, n_red=700, n_blue=500, slope=0.06, intercept=0.70, sigma=0.03):
    rng = np.random.default_rng(seed)
    m_red = rng.uniform(9.6, 11.5, n_red)
    c_red = intercept + slope * (m_red - 10.0) + rng.normal(0, sigma, n_red)
    m_blue = rng.uniform(9.6, 11.5, n_blue)
    c_blue = 0.38 + rng.normal(0, 0.07, n_blue)
    return np.concatenate([c_red, c_blue]), np.concatenate([m_red, m_blue])


def test_red_sequence_recovers_slope_and_intercept():
    colour, logm = _synthetic()
    fit = fit_red_sequence(colour, logm, mass_range=[9.5, 12.0])
    assert fit["converged"]
    assert abs(fit["slope"] - 0.06) < 0.02
    assert abs(fit["intercept"] - 0.70) < 0.01
    # the clipped set underestimates the 0.03 scatter somewhat, but not by much
    assert 0.02 < fit["sigma"] < 0.035
    assert fit["n_fit"] > 0.8 * 700 and fit["n_fit"] < 760     # the blue cloud is excluded
    assert fit["n_sample"] == 1200
    np.testing.assert_allclose(red_sequence_colour(fit, 10.0), fit["intercept"])


def test_red_sequence_ignores_nan_and_mass_range():
    colour, logm = _synthetic(seed=1)
    colour = colour.copy()
    colour[:10] = np.nan
    fit = fit_red_sequence(colour, logm, mass_range=[10.0, 11.0])
    assert fit["n_sample"] < 1190 and abs(fit["intercept"] - 0.70) < 0.01
    assert fit_red_sequence(colour[:2], logm[:2]) is None
    # too few galaxies per bin for a seed
    assert fit_red_sequence(colour[:30], logm[:30], min_per_bin=100) is None


def test_compute_indicators():
    colour, logm = _synthetic(seed=2)
    n = len(colour)
    frame = pd.DataFrame({
        "z_type": ["spec"] * n, "col_gr": colour, "logmstar": logm,
        "col_w2w3_vega": np.where(np.arange(n) % 3 == 0, 2.0, np.nan),
        "class_spec": "GALAXY"})
    # a photo-z row, far below the sequence, and one without a mass
    extra = pd.DataFrame({"z_type": ["phot", "phot"], "col_gr": [0.2, 0.7],
                          "logmstar": [10.0, np.nan], "col_w2w3_vega": [1.0, np.nan]})
    frame = pd.concat([frame, extra], ignore_index=True)
    cols, info = compute_indicators(frame, indicator_settings({}))
    fit = info["red_sequence"]
    assert abs(fit["intercept"] - 0.70) < 0.02
    off = cols["rs_offset"]
    np.testing.assert_allclose(off[:n], colour - red_sequence_colour(fit, logm))
    assert np.isclose(off[n], 0.2 - fit["intercept"])          # photo-z row at its median mass
    assert np.isnan(off[n + 1])
    q = cols["quenched"]
    assert str(q.dtype) == "boolean"
    assert q[n] is False or q[n] == False                      # noqa: E712
    assert q[n + 1] is pd.NA
    is_red = np.arange(n) < 700
    assert q[:n][is_red].astype(float).mean() > 0.95           # -2 sigma keeps the red galaxies
    assert q[:n][~is_red].astype(float).mean() < 0.02          # and drops the blue cloud
    # offset > -n sigma, n from the settings
    s3 = indicator_settings({"indicators": {"red_sequence": {"n_sigma_quench": 20.0}}})
    cols3, _ = compute_indicators(frame, s3)
    assert cols3["quenched"][n]
    # W2 - W3 >= 1.5 flags star formation; no colour -> NA
    sf = cols["sf_w2w3"]
    assert sf[0] and not sf[n] and sf[1] is pd.NA
    assert info["passthrough_present"] == ["class_spec"]
    assert 0.0 < info["quenched_fraction_spec"] < 1.0
