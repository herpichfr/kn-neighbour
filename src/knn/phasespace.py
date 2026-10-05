"""Cluster-centric projected phase space."""

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord

C_KMS = 299792.458
CLUSTER_KEYS = ("ra", "dec", "z", "r200_mpc", "sigma_v")
CATALOGUE_RADIUS = "radius_Mpc"
CATALOGUE_VELOCITY = "v_offset"


def cluster_settings(cfg):
    """The 'cluster' section with every key present (None where not given)."""
    user = cfg.get("cluster") or {}
    return {k: user.get(k) for k in CLUSTER_KEYS}


def projected_radius_mpc(ra, dec, cluster, cosmo):
    """Proper projected distance (Mpc) of each galaxy from the cluster centre.

    R = D_A(z_cl) theta, with theta the angular separation from
    (cluster['ra'], cluster['dec']) and D_A the angular diameter distance at
    the cluster redshift (every galaxy is assumed to lie at the cluster
    distance). NaN for rows without coordinates.
    """
    ra, dec = np.asarray(ra, float), np.asarray(dec, float)
    out = np.full(ra.shape, np.nan)
    ok = np.isfinite(ra) & np.isfinite(dec)
    centre = SkyCoord(cluster["ra"] * u.deg, cluster["dec"] * u.deg)
    theta = SkyCoord(ra[ok] * u.deg, dec[ok] * u.deg).separation(centre).to_value(u.rad)
    d_a = float(cosmo.angular_diameter_distance(cluster["z"]).to_value(u.Mpc))
    out[ok] = d_a * theta
    return out


def los_velocity_kms(z, z_cluster):
    """Line-of-sight velocity offset c (z - z_cl) / (1 + z_cl) in km/s."""
    return C_KMS * (np.asarray(z, dtype=float) - z_cluster) / (1.0 + z_cluster)


def phase_space(df, cluster, cosmo):
    """Phase-space columns for the rows of df.

    cluster is cluster_settings(cfg). Columns:

    ps_R_mpc : projected radius. With cluster ra, dec and z given it is
        computed (projected_radius_mpc, all rows); otherwise the catalogue's
        radius_Mpc is used if present.
    ps_dv_kms : line-of-sight velocity offset. With cluster z given it is
        c (z - z_cl) / (1 + z_cl) for rows with a spec-z (z_type == 'spec')
        and NaN for photo-z rows; otherwise the catalogue's v_offset if present.
    ps_R_over_r200 : ps_R_mpc / r200_mpc, if r200_mpc is given.
    ps_absdv_over_sigma : |ps_dv_kms| / sigma_v, if sigma_v is given.

    Returns (columns, info); info records where each column came from.
    """
    n = len(df)
    cols, info = {}, {}
    have_centre = (cluster["ra"] is not None and cluster["dec"] is not None
                   and cluster["z"] is not None)
    if have_centre:
        cols["ps_R_mpc"] = projected_radius_mpc(df["ra"], df["dec"], cluster, cosmo)
        info["R_source"] = "computed from the cluster centre (config)"
    elif CATALOGUE_RADIUS in df.columns:
        cols["ps_R_mpc"] = df[CATALOGUE_RADIUS].to_numpy(float)
        info["R_source"] = f"catalogue column {CATALOGUE_RADIUS}"
    if cluster["z"] is not None:
        spec = (df["z_type"] == "spec").to_numpy()
        cols["ps_dv_kms"] = np.where(spec, los_velocity_kms(df["z_best"].to_numpy(float),
                                                            cluster["z"]), np.nan)
        info["dv_source"] = "computed from the cluster redshift (config)"
    elif CATALOGUE_VELOCITY in df.columns:
        cols["ps_dv_kms"] = df[CATALOGUE_VELOCITY].to_numpy(float)
        info["dv_source"] = f"catalogue column {CATALOGUE_VELOCITY}"
    if cluster["r200_mpc"] is not None and "ps_R_mpc" in cols:
        cols["ps_R_over_r200"] = cols["ps_R_mpc"] / cluster["r200_mpc"]
    if cluster["sigma_v"] is not None and "ps_dv_kms" in cols:
        cols["ps_absdv_over_sigma"] = np.abs(cols["ps_dv_kms"]) / cluster["sigma_v"]
    info["cluster"] = dict(cluster)
    info["columns"] = sorted(cols)
    return cols, info
