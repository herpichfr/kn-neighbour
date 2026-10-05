from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from knn.photometry import (EXTINCTION_COEFFICIENTS, KCOR_COEFFICIENTS, clean_mags,
                            ebv_values, extinction_corrected, extract_mags, kcorrection,
                            prepare_photometry, rest_frame_optical, rest_frame_ur,
                            sfd_ebv)

SFD_DIR = Path("/home/herpich/Documents/sfddata-master")


def test_clean_mags_missing_values():
    raw = np.array([18.5, 99.0, np.nan, np.inf, -np.inf, 90.0, 89.9, 25.0], dtype=np.float32)
    out = clean_mags(raw)
    np.testing.assert_allclose(out[[0, 6, 7]], [18.5, 89.9, 25.0], rtol=1e-6)
    assert np.isnan(out[[1, 2, 3, 4, 5]]).all()
    assert out.dtype == float
    # the threshold is configurable
    assert np.isnan(clean_mags([20.0, 21.0], missing_above=20.5)[1])
    # the input is not modified
    assert raw[1] == 99.0


def test_extract_mags_sets_and_errors():
    df = pd.DataFrame({"mag_g": [18.0, np.inf], "mag_r": [17.0, 17.5], "mag_i": [16.5, 99.0]})
    mags, system = extract_mags(df, "legacy", ("g", "r", "i"))
    assert system == "decam"
    assert np.isnan(mags["g"][1]) and np.isnan(mags["i"][1]) and mags["r"][1] == 17.5
    with pytest.raises(KeyError, match="unknown photometry set"):
        extract_mags(df, "nope", ("g",))
    with pytest.raises(KeyError, match="not found"):
        extract_mags(df, "splus_auto", ("g",))
    # a custom set
    custom = {"mine": {"system": "sdss", "bands": {"g": "mag_g"}}}
    m, s = extract_mags(df, "mine", ("g",), custom=custom)
    assert s == "sdss" and m["g"][0] == 18.0
    with pytest.raises(KeyError, match="no band"):
        extract_mags(df, "mine", ("r",), custom=custom)


def test_extinction_correction():
    mags = {"g": np.array([18.0, np.nan]), "i": np.array([17.0, 17.0])}
    ebv = np.array([0.1, 0.1])
    out, used = extinction_corrected(mags, "decam", ebv)
    np.testing.assert_allclose(out["g"][0], 18.0 - 3.214 * 0.1)
    np.testing.assert_allclose(out["i"], 17.0 - 1.592 * 0.1)
    assert np.isnan(out["g"][1])
    assert used == {"g": 3.214, "i": 1.592}
    # S-PLUS (SDSS coefficients, Schlafly & Finkbeiner 2011 Table 6)
    out, _ = extinction_corrected(mags, "sdss", ebv)
    np.testing.assert_allclose(out["g"][0], 18.0 - 0.3303)
    # override and no extinction
    out, _ = extinction_corrected(mags, "sdss", ebv, overrides={"g": 1.0})
    np.testing.assert_allclose(out["g"][0], 17.9)
    out, used = extinction_corrected(mags, "sdss", None)
    np.testing.assert_array_equal(out["i"], mags["i"])
    assert used == {}
    with pytest.raises(KeyError, match="coefficient"):
        extinction_corrected({"y": np.array([1.0])}, "sdss", ebv[:1])
    assert EXTINCTION_COEFFICIENTS["decam"]["w1"] == 0.184


def test_ebv_sources(tmp_path):
    df = pd.DataFrame({"ra": [180.0], "dec": [1.0], "ebv_col": [0.05]})
    ebv, info = ebv_values(df, {"ebv": None})
    assert ebv is None and info["applied"] is False and "not applied" in info["note"]
    ebv, info = ebv_values(df, {"ebv": "ebv_col"})
    np.testing.assert_array_equal(ebv, [0.05])
    assert info["applied"] and info["ebv_median"] == 0.05
    with pytest.raises(KeyError, match="nope"):
        ebv_values(df, {"ebv": "nope"})
    with pytest.raises(FileNotFoundError, match="sfd_dir"):
        ebv_values(df, {"ebv": "sfd", "sfd_dir": str(tmp_path)})


@pytest.mark.skipif(not (SFD_DIR / "SFD_dust_4096_ngp.fits").is_file(),
                    reason="SFD map files are not available")
def test_sfd_map_value_at_mkw4():
    pytest.importorskip("sfdmap")
    ebv = sfd_ebv([180.99], [1.89], SFD_DIR)
    assert 0.01 < ebv[0] < 0.04


def test_kcorrection_against_authors_code():
    # values from calc_kcor.py of Chilingarian et al. (http://kcor.sai.msu.ru), run once
    np.testing.assert_allclose(kcorrection("r", "gr", 0.1, 0.8), 0.06886788400000003)
    np.testing.assert_allclose(kcorrection("g", "gi", 0.03, 1.1), 0.06668077241133001)
    np.testing.assert_allclose(kcorrection("i", "gi", 0.03, 1.1), 0.02499600040731)
    np.testing.assert_allclose(kcorrection("u", "ur", 0.05, 2.0), 0.14733717437499996)
    np.testing.assert_allclose(kcorrection("r", "ur", 0.05, 2.0), 0.045881530843750004)
    # vectorised, zero at z = 0, NaN for a NaN colour
    k = kcorrection("g", "gi", np.array([0.0, 0.03, 0.03]), np.array([1.0, 1.1, np.nan]))
    assert k[0] == 0.0 and np.isclose(k[1], 0.06668077241133001) and np.isnan(k[2])
    with pytest.raises(KeyError):
        kcorrection("g", "gr", 0.1, 1.0)
    assert set(KCOR_COEFFICIENTS) == {"g_gi", "i_gi", "r_gr", "r_ur", "u_ur"}


def test_rest_frame():
    z, g, r, i = 0.03, np.array([18.0]), np.array([17.2]), np.array([16.8])
    rest = rest_frame_optical(z, g, r, i)
    np.testing.assert_allclose(rest["kg"], kcorrection("g", "gi", z, g - i))
    np.testing.assert_allclose(rest["kr"], kcorrection("r", "gr", z, g - r))
    np.testing.assert_allclose(rest["gi"], (g - rest["kg"]) - (i - rest["ki"]))
    off = rest_frame_optical(z, g, r, i, apply=False)
    np.testing.assert_array_equal(off["gi"], g - i)
    np.testing.assert_array_equal(off["kg"], 0.0)
    u, r2 = np.array([19.5]), np.array([17.5])
    ur = rest_frame_ur(z, u, r2)
    np.testing.assert_allclose(
        ur, 2.0 - (kcorrection("u", "ur", z, 2.0) - kcorrection("r", "ur", z, 2.0)))
    np.testing.assert_array_equal(rest_frame_ur(z, u, r2, apply=False), u - r2)


def _cfg(**extra):
    cfg = {"photometry": {"set": "legacy", "ur_set": "splus_auto", "wise_set": "legacy"},
           "extinction": {"ebv": "ebv"}, "kcorrection": {"apply": False}}
    cfg.update(extra)
    return cfg


def test_prepare_photometry():
    df = pd.DataFrame({
        "ebv": [0.1, 0.1], "ra": [1.0, 2.0], "dec": [0.0, 0.0],
        "mag_g": [18.0, 99.0], "mag_r": [17.0, 17.0], "mag_i": [16.5, 16.5],
        "mag_w1": [15.0, 15.0], "mag_w2": [15.5, np.inf], "mag_w3": [13.0, 13.0],
        "u_auto": [19.0, 99.0], "r_auto": [17.0, 17.0],
    })
    phot = prepare_photometry(df, _cfg())
    np.testing.assert_allclose(phot.opt["g"][0], 18.0 - 3.214 * 0.1)
    assert np.isnan(phot.opt["g"][1])                   # 99 -> missing
    assert np.isnan(phot.wise["w2"][1])                 # inf -> missing
    np.testing.assert_allclose(phot.ur["u"][0], 19.0 - 4.239 * 0.1)
    assert np.isnan(phot.ur["u"][1])
    assert phot.info["extinction"]["applied"] and phot.info["kcorrection"]["applied"] is False
    assert phot.info["extinction_coefficients"]["wise"]["w1"] == 0.184
    # no extinction: values untouched, recorded
    phot = prepare_photometry(df, _cfg(extinction={"ebv": None}))
    assert phot.opt["r"][0] == 17.0 and phot.info["extinction"]["applied"] is False
    # no S-PLUS columns: u-r is all NaN, not an error
    phot = prepare_photometry(df.drop(columns=["u_auto"]), _cfg())
    assert np.isnan(phot.ur["u"]).all() and "ur_missing" in phot.info
