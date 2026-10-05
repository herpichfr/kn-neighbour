"""Synthetic catalogues for the explorer tests (no data files needed)."""

import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM

from knn.explorer import Explorer
from knn.geometry import comoving_xyz

COSMO = FlatLambdaCDM(H0=70, Om0=0.3)


def synthetic_frames(n=100_000, phot_fraction=0.15, n_draws=20, seed=5, window=(0.007, 0.037)):
    """(cfg, df, tracers) of a random catalogue with the columns of the enriched table.

    Galaxies are uniform on a 6 x 6 deg field and in z; spec-z galaxies are the
    tracers; masses, M_r and the quenched flag are random. No build is run.
    """
    rng = np.random.default_rng(seed)
    z = rng.uniform(window[0] + 0.001, window[1] - 0.001, n)
    ra = rng.uniform(178.0, 184.0, n)
    dec = rng.uniform(-3.0, 3.0, n)
    is_phot = rng.random(n) < phot_fraction
    xyz = comoving_xyz(ra, dec, z, COSMO)
    logm = rng.normal(9.0, 1.0, n)
    quenched = pd.array(rng.random(n) < 1.0 / (1.0 + np.exp(-(logm - 10.0))), dtype="boolean")
    df = pd.DataFrame({
        "ra": ra, "dec": dec, "z_best": z, "z_type": np.where(is_phot, "phot", "spec"),
        "z_spec": np.where(is_phot, np.nan, z),
        "z_phot": np.where(is_phot, z + rng.normal(0.0, 0.006, n), z + rng.normal(0, 0.005, n)),
        "z_phot_odds": rng.uniform(0.3, 1.0, n), "ID_final": np.arange(n, dtype=float),
        "cx": xyz[:, 0], "cy": xyz[:, 1], "cz": xyz[:, 2],
        "logmstar": logm, "M_r": -16.0 - 0.8 * (logm - 8.0) + rng.normal(0, 0.3, n),
        "quenched": quenched, "p_in_window": np.where(is_phot, rng.uniform(0.5, 1.0, n), 1.0),
        "flag_member": rng.choice([-1.0, 0.0, 1.0], n),
    })
    tr = ~is_phot
    tracers = pd.DataFrame({
        "ra": ra[tr], "dec": dec[tr], "z_best": z[tr], "z_type": "spec",
        "cx": xyz[tr, 0], "cy": xyz[tr, 1], "cz": xyz[tr, 2], "logm_point": logm[tr],
        "out_row": np.flatnonzero(tr)})
    cfg = {"redshift_window": list(window), "cosmology": {"H0": 70, "Om0": 0.3},
           "photoz": {"mode": "gauss", "n_draws": n_draws, "seed": 3, "sigma": 0.006,
                      "bias": 0.0},
           "environment": [{"name": "sigma2d", "k": 10}], "tracers": "spec",
           "output": "unused.parquet"}
    return cfg, df, tracers


def synthetic_explorer(**kwargs):
    cfg, df, tracers = synthetic_frames(**kwargs)
    return Explorer(cfg, df, tracers)


N_SPEC, N_PHOT = 300, 80
BUILD_CONFIG = {
    "catalogue": "data/cat.parquet",
    "columns": {"ra": "RA", "dec": "DEC", "z_spec": "zspec", "z_phot": "zphot",
                "z_phot_odds": "pz"},
    "cosmology": {"H0": 70, "Om0": 0.3},
    "redshift_window": [0.007, 0.037],
    "tracer_window": [0.004, 0.040],
    "tracers": "spec",
    "environment": [
        {"name": "knn3d", "k": 5, "photoz_mode": "none"},
        {"name": "sigma2d", "k": 5},
        {"name": "sigma2d", "k": 5, "weight": "mass"},
    ],
    "photoz": {"mode": "gauss", "n_draws": 60, "seed": 3, "sigma": 0.01, "bias": 0.0},
    "photometry": {"set": "legacy", "ur_set": "splus_auto", "wise_set": "legacy"},
    "extinction": {"ebv": "ebv"},
    "kcorrection": {"apply": True},
    "mass": {"method": "taylor"},
    "indicators": {"red_sequence": {"mass_range": [8.0, 12.0], "min_per_bin": 10}},
    "joins": [
        {"name": "morph", "path": "data/aux.csv", "ra": "ra", "dec": "dec",
         "columns": {"ser": "morph_Ser_n"}},
        {"name": "grp_photz", "path": "data/aux.csv", "ra": "ra", "dec": "dec",
         "columns": {"gid": "grp_photz_id_group_final"}, "int_columns": ["gid"]},
        {"name": "substructure", "path": "data/aux.csv", "ra": "ra", "dec": "dec",
         "columns": {"sub": "substructure"}},
    ],
    "cluster": {"ra": 180.5, "dec": 0.5, "z": 0.02, "r200_mpc": 1.0, "sigma_v": 300.0},
    "output": "out/enriched.parquet",
}


def write_build_catalogue(root):
    """Write data/cat.parquet and data/aux.csv under root (a raw catalogue for knn.build)."""
    rng = np.random.default_rng(11)
    n = N_SPEC + N_PHOT
    zspec = np.full(n, np.nan)
    zphot = np.full(n, np.nan)
    zspec[:N_SPEC] = rng.uniform(0.012, 0.03, N_SPEC)
    zphot[N_SPEC:] = rng.uniform(0.012, 0.03, N_PHOT)
    z = np.where(np.isfinite(zspec), zspec, zphot)
    gi = rng.uniform(0.3, 1.3, n)
    i = rng.normal(-18.0, 1.0, n) + COSMO.distmod(z).value
    df = pd.DataFrame({
        "RA": rng.uniform(180, 181, n), "DEC": rng.uniform(0, 1, n),
        "zspec": zspec, "zphot": zphot, "pz": rng.uniform(0.2, 1.0, n), "ebv": 0.03,
        "mag_i": i, "mag_g": i + gi, "mag_r": i + 0.45 * gi, "mag_z": i - 0.2,
        "mag_w1": i - 0.4, "mag_w2": i - 0.4 + 0.5, "mag_w3": i + 1.0, "mag_w4": i + 2.0,
        "u_auto": i + 1.5 * gi + 0.8, "r_auto": i + 0.45 * gi,
        "class_spec": "GALAXY", "flag_member": rng.choice([-1.0, 0.0, 1.0], n),
    })
    (root / "data").mkdir(parents=True, exist_ok=True)
    df.to_parquet(root / "data" / "cat.parquet", index=False)
    m = 120                                          # galaxies with a match in the aux table
    aux = pd.DataFrame({
        "ra": df.RA[:m].to_numpy(), "dec": df.DEC[:m].to_numpy(),
        "sub": rng.choice(["S1", "S2", "S3"], m), "gid": rng.integers(-1, 5, m).astype(float),
        "ser": rng.uniform(0.5, 4.0, m)})
    aux.to_csv(root / "data" / "aux.csv", index=False)
    return df


def build_synthetic_config(root):
    """Write the catalogue and a config under root and return load_config(config)."""
    import yaml

    from knn.config import load_config
    write_build_catalogue(root)
    path = root / "config" / "t.yaml"
    path.parent.mkdir(exist_ok=True)
    path.write_text(yaml.safe_dump(BUILD_CONFIG))
    return load_config(path)
