import numpy as np
import pytest
from astropy.cosmology import FlatLambdaCDM

from knn.mass import (M_SUN_I_AB, M_SUN_W1_VEGA, WISE_VEGA_TO_AB, cluver_logmstar,
                      compute_masses, distance_modulus, luminosity_distance_mpc, mass_settings,
                      select_mass, taylor_logmstar)
from knn.photometry import Photometry, kcorrection

COSMO = FlatLambdaCDM(H0=70, Om0=0.3)
SETTINGS = mass_settings({})


def test_taylor_hand_computed():
    # log M = -0.68 + 0.70 (g - i) + 0.4 (4.58 - M_i)
    assert np.isclose(taylor_logmstar(1.0, -20.0), -0.68 + 0.70 + 0.4 * 24.58)
    assert np.isclose(taylor_logmstar(1.0, -20.0), 9.852)
    # M_i = M_sun gives log M/L = -0.68 + 0.7 (g-i) in solar units
    assert np.isclose(taylor_logmstar(0.5, M_SUN_I_AB), -0.68 + 0.35)
    # a 0.4 mag fainter galaxy is 0.16 dex lighter
    assert np.isclose(taylor_logmstar(0.8, -19.6) - taylor_logmstar(0.8, -20.0), -0.16)


def test_cluver_hand_computed():
    # log M = -0.03 - 1.96 (W1 - W2) + 0.4 (3.24 - M_W1), Vega
    assert np.isclose(cluver_logmstar(0.1, -23.0), -0.03 - 0.196 + 0.4 * 26.24)
    assert np.isclose(cluver_logmstar(0.0, M_SUN_W1_VEGA), -0.03)


def test_distance_modulus_matches_astropy():
    z = np.array([0.01, 0.02, 0.037])
    np.testing.assert_allclose(distance_modulus(z, COSMO), COSMO.distmod(z).value, rtol=1e-12)
    np.testing.assert_allclose(luminosity_distance_mpc(z, COSMO),
                               COSMO.luminosity_distance(z).value)
    out = distance_modulus(np.array([np.nan, 0.0, -0.1, 0.02]), COSMO)
    assert np.isnan(out[:3]).all() and np.isfinite(out[3])


def _phot(g, r, i, w1=None, w2=None, kcorr=False, wise_vega=False):
    n = len(g)
    nan = np.full(n, np.nan)
    return Photometry(opt={"g": np.asarray(g, float), "r": np.asarray(r, float),
                           "i": np.asarray(i, float)},
                      ur={"u": nan, "r": nan},
                      wise={"w1": nan if w1 is None else np.asarray(w1, float),
                            "w2": nan if w2 is None else np.asarray(w2, float), "w3": nan},
                      wise_vega=wise_vega, settings={}, kcorrection=kcorr)


def test_absolute_magnitudes_and_masses_hand_computed():
    z = np.array([0.02, 0.03])
    dm = COSMO.distmod(z).value
    g, r, i = np.array([17.8, 18.5]), np.array([17.2, 17.9]), np.array([16.8, 17.5])
    res = compute_masses(_phot(g, r, i), z, COSMO, SETTINGS)
    np.testing.assert_allclose(res["M_i"], i - dm)
    np.testing.assert_allclose(res["M_r"], r - dm)
    np.testing.assert_allclose(res["gi"], g - i)
    np.testing.assert_allclose(res["logmstar_taylor"],
                               -0.68 + 0.70 * (g - i) + 0.4 * (4.58 - (i - dm)))
    assert np.isnan(res["logmstar_w1"]).all()
    # K-corrections enter the rest-frame magnitudes and the colour
    res_k = compute_masses(_phot(g, r, i, kcorr=True), z, COSMO, SETTINGS)
    ki = kcorrection("i", "gi", z, g - i)
    kg = kcorrection("g", "gi", z, g - i)
    np.testing.assert_allclose(res_k["M_i"], i - dm - ki)
    np.testing.assert_allclose(res_k["gi"], (g - kg) - (i - ki))


def test_w1_mass_with_vega_conversion():
    z = np.array([0.02])
    dm = COSMO.distmod(z).value
    w1_ab, w2_ab = np.array([15.0]), np.array([15.4])
    ph = _phot([17.0], [16.5], [16.2], w1=w1_ab, w2=w2_ab)
    res = compute_masses(ph, z, COSMO, SETTINGS)
    w1_v = 15.0 - WISE_VEGA_TO_AB["w1"]
    c = (15.0 - WISE_VEGA_TO_AB["w1"]) - (15.4 - WISE_VEGA_TO_AB["w2"])      # = -0.4 + 0.64
    assert np.isclose(c, 0.24)
    np.testing.assert_allclose(res["M_w1"], w1_v - dm)
    np.testing.assert_allclose(res["logmstar_w1"],
                               -0.03 - 1.96 * c + 0.4 * (3.24 - (w1_v - dm)))
    # already-Vega magnitudes are not shifted
    ph_v = _phot([17.0], [16.5], [16.2], w1=w1_ab - 2.699, w2=w2_ab - 3.339, wise_vega=True)
    res_v = compute_masses(ph_v, z, COSMO, SETTINGS)
    np.testing.assert_allclose(res_v["logmstar_w1"], res["logmstar_w1"])


def test_missing_values_and_colour_limits():
    z = np.full(4, 0.02)
    g = np.array([17.8, np.nan, 17.8, 17.8])
    i = np.array([16.8, 16.8, np.nan, 12.0])     # last: g - i = 5.8, out of range
    r = np.array([17.2, 17.2, 17.2, 17.2])
    res = compute_masses(_phot(g, r, i), z, COSMO, SETTINGS)
    assert np.isfinite(res["logmstar_taylor"][0])
    assert np.isnan(res["logmstar_taylor"][1:]).all()
    assert np.isnan(res["M_i"][3]) and np.isfinite(res["M_r"][3])
    # W1 - W2 (Vega) outside [-0.2, 0.8] is rejected
    ph = _phot([17.0] * 3, [16.5] * 3, [16.2] * 3, w1=[15.0, 15.0, 15.0],
               w2=[15.6, 14.0, np.nan])   # W1-W2 Vega = -0.6+0.64=0.04, 1.64, nan
    res = compute_masses(ph, np.full(3, 0.02), COSMO, SETTINGS)
    assert np.isfinite(res["logmstar_w1"][0])
    assert np.isnan(res["logmstar_w1"][1:]).all()


def test_rows_index_draws():
    z = np.array([0.02, 0.021, 0.03])
    rows = np.array([1, 1, 0])
    ph = _phot([17.0, 18.0], [16.5, 17.5], [16.2, 17.2])
    res = compute_masses(ph, z, COSMO, SETTINGS, rows=rows)
    dm = COSMO.distmod(z).value
    np.testing.assert_allclose(res["M_i"], np.array([17.2, 17.2, 16.2]) - dm)


def test_select_mass_and_fallback():
    vals = {"logmstar_taylor": np.array([9.0, np.nan, np.nan]),
            "logmstar_w1": np.array([9.5, 9.6, np.nan])}
    logm, used = select_mass(vals)
    np.testing.assert_array_equal(logm, [9.0, np.nan, np.nan])
    assert list(used) == ["taylor", None, None]
    logm, used = select_mass(vals, "taylor", "w1")
    np.testing.assert_array_equal(logm, [9.0, 9.6, np.nan])
    assert list(used) == ["taylor", "w1", None]
    logm, _ = select_mass(vals, "w1")
    np.testing.assert_array_equal(logm, [9.5, 9.6, np.nan])
    with pytest.raises(ValueError):
        select_mass(vals, "magic")
