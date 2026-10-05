import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM

from knn.phasespace import (C_KMS, cluster_settings, los_velocity_kms, phase_space,
                            projected_radius_mpc)

COSMO = FlatLambdaCDM(H0=70, Om0=0.3)
FULL = {"ra": 180.0, "dec": 0.0, "z": 0.02, "r200_mpc": 0.9, "sigma_v": 300.0}


def _df():
    return pd.DataFrame({
        "ra": [180.0, 181.0, 180.0, np.nan], "dec": [0.0, 0.0, 2.0, 0.0],
        "z_best": [0.02, 0.0233, 0.0167, 0.02], "z_type": ["spec", "spec", "phot", "spec"],
        "radius_Mpc": [0.0, 1.5, 3.0, np.nan], "v_offset": [1.0, 2.0, np.nan, 4.0]})


def test_projected_radius_and_velocity():
    r = projected_radius_mpc([180.0, 181.0, 180.0, np.nan], [0.0, 0.0, 2.0, 0.0], FULL, COSMO)
    d_a = COSMO.angular_diameter_distance(0.02).value
    np.testing.assert_allclose(r[:3], d_a * np.radians([0.0, 1.0, 2.0]), rtol=1e-12)
    assert np.isnan(r[3])
    # separations are on the sphere, not ra differences, away from the equator
    r2 = projected_radius_mpc([181.0], [60.0], {**FULL, "dec": 60.0}, COSMO)
    sep = SkyCoord(181.0 * u.deg, 60.0 * u.deg).separation(SkyCoord(180.0 * u.deg, 60.0 * u.deg))
    np.testing.assert_allclose(r2, d_a * sep.rad)
    assert np.isclose(r2[0], d_a * np.radians(0.5), rtol=1e-3)
    np.testing.assert_allclose(los_velocity_kms([0.02, 0.03], 0.02),
                               [0.0, C_KMS * 0.01 / 1.02])


def test_phase_space_with_cluster_parameters():
    cols, info = phase_space(_df(), cluster_settings({"cluster": FULL}), COSMO)
    d_a = COSMO.angular_diameter_distance(0.02).value
    np.testing.assert_allclose(cols["ps_R_mpc"][:3], d_a * np.radians([0.0, 1.0, 2.0]))
    np.testing.assert_allclose(cols["ps_dv_kms"][:2],
                               C_KMS * (np.array([0.02, 0.0233]) - 0.02) / 1.02)
    assert np.isnan(cols["ps_dv_kms"][2])          # photo-z row: no velocity offset
    assert np.isnan(cols["ps_R_mpc"][3])
    np.testing.assert_allclose(cols["ps_R_over_r200"], cols["ps_R_mpc"] / 0.9)
    np.testing.assert_allclose(cols["ps_absdv_over_sigma"], np.abs(cols["ps_dv_kms"]) / 300.0)
    assert "computed" in info["R_source"] and "computed" in info["dv_source"]


def test_phase_space_falls_back_to_catalogue_columns():
    cols, info = phase_space(_df(), cluster_settings({"cluster": {"r200_mpc": 1.5}}), COSMO)
    np.testing.assert_array_equal(cols["ps_R_mpc"], _df()["radius_Mpc"])
    np.testing.assert_array_equal(cols["ps_dv_kms"], _df()["v_offset"])
    np.testing.assert_allclose(cols["ps_R_over_r200"][:3], [0.0, 1.0, 2.0])
    assert "ps_absdv_over_sigma" not in cols
    assert "catalogue" in info["R_source"]
    # z only: velocity computed, radius from the catalogue
    cols, info = phase_space(_df(), cluster_settings({"cluster": {"z": 0.02}}), COSMO)
    assert "computed" in info["dv_source"] and "catalogue" in info["R_source"]
    # nothing to compute from
    cols, _ = phase_space(_df().drop(columns=["radius_Mpc", "v_offset"]),
                          cluster_settings({"cluster": {}}), COSMO)
    assert cols == {}
