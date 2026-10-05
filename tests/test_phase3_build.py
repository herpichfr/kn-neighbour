import copy
import json

import numpy as np
import pandas as pd
import pytest
import yaml
from astropy.cosmology import FlatLambdaCDM

from knn.build import build
from knn.config import ConfigError, load_config
from knn.io import load_catalogue
from knn.photoz_env import make_draws, summarise
from knn.redshift import assign_redshifts

COSMO = FlatLambdaCDM(H0=70, Om0=0.3)
N_SPEC, N_PHOT = 300, 80

CONFIG = {
    "catalogue": "data/cat.parquet",
    "columns": {"ra": "RA", "dec": "DEC", "z_spec": "zspec", "z_phot": "zphot"},
    "cosmology": {"H0": 70, "Om0": 0.3},
    "redshift_window": [0.007, 0.037],
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
    "joins": [{"name": "toy", "path": "data/aux.csv", "ra": "ra", "dec": "dec",
               "columns": ["label", "gid"], "int_columns": ["gid"], "prefix": "aux"}],
    "cluster": {"ra": 180.5, "dec": 0.5, "z": 0.02, "r200_mpc": 1.0, "sigma_v": 300.0},
    "output": "out/enriched.parquet",
}


def _write_catalogue(root):
    rng = np.random.default_rng(11)
    n = N_SPEC + N_PHOT
    zspec = np.full(n, np.nan)
    zphot = np.full(n, np.nan)
    zspec[:N_SPEC] = rng.uniform(0.012, 0.03, N_SPEC)
    zphot[N_SPEC:] = rng.uniform(0.012, 0.03, N_PHOT)
    z = np.where(np.isfinite(zspec), zspec, zphot)
    dm = COSMO.distmod(z).value
    gi = rng.uniform(0.3, 1.3, n)
    m_i = rng.normal(-18.0, 1.0, n)
    i = m_i + dm
    df = pd.DataFrame({
        "RA": rng.uniform(180, 181, n), "DEC": rng.uniform(0, 1, n),
        "zspec": zspec, "zphot": zphot, "ebv": 0.03,
        "mag_i": i, "mag_g": i + gi, "mag_r": i + 0.45 * gi, "mag_z": i - 0.2,
        "mag_w1": i - 0.4, "mag_w2": i - 0.4 + 0.5, "mag_w3": i + 1.0, "mag_w4": i + 2.0,
        "u_auto": i + 1.5 * gi + 0.8, "r_auto": i + 0.45 * gi,
        "class_spec": "GALAXY",
    })
    df.loc[:19, "mag_g"] = 99.0                    # 20 spec-z galaxies without g
    df.loc[20:24, "mag_w2"] = np.inf               # no W1-W2 for 5 more
    (root / "data").mkdir(parents=True, exist_ok=True)
    df.to_parquet(root / "data" / "cat.parquet", index=False)
    aux = pd.DataFrame({"ra": df.RA[:50].to_numpy(), "dec": df.DEC[:50].to_numpy(),
                        "label": [f"L\\_{j}" for j in range(50)], "gid": np.arange(50.0)})
    aux.to_csv(root / "data" / "aux.csv", index=False)
    return df


def _load(root, config):
    path = root / "config" / "t.yaml"
    path.parent.mkdir(exist_ok=True)
    path.write_text(yaml.safe_dump(config))
    return load_config(path)


@pytest.fixture
def setup(tmp_path):
    df = _write_catalogue(tmp_path)
    cfg = _load(tmp_path, CONFIG)
    return tmp_path, df, cfg


def test_phase3_columns_and_environment(setup):
    root, raw, cfg = setup
    df = build(cfg)
    spec = (df["z_type"] == "spec").to_numpy()
    phot = (df["z_type"] == "phot").to_numpy()
    assert spec.sum() == N_SPEC and phot.sum() == N_PHOT

    # photoz_mode none: photo-z rows are NaN, spec-z rows are measured
    for col in ("r_knn3d_k5", "rho_knn3d_k5", "log1p_delta_knn3d_k5"):
        assert df.loc[phot, [col, col + "_lo", col + "_hi"]].isna().all().all()
        assert df.loc[spec, col].notna().all()
    # count and mass versions live in different columns
    for col in ("r_sigma2d_k5", "sigma_sigma2d_k5", "log1p_delta_sigma2d_k5",
                "r_sigma2d_k5_mass", "sigma_sigma2d_k5_mass", "log1p_delta_sigma2d_k5_mass"):
        assert col in df.columns and df.loc[phot, col].notna().all()
    assert not np.allclose(df.loc[spec, "sigma_sigma2d_k5"], df.loc[spec, "sigma_sigma2d_k5_mass"])

    # masses: spec-z lo = hi = value, photo-z lo <= median <= hi with a real width
    for q in ("logmstar", "logmstar_taylor", "logmstar_w1", "M_r", "M_i"):
        med, lo, hi = df[q], df[q + "_lo"], df[q + "_hi"]
        np.testing.assert_array_equal(lo[spec], med[spec])
        np.testing.assert_array_equal(hi[spec], med[spec])
        ok = med.notna().to_numpy()
        assert ((lo <= med) & (med <= hi))[ok].all()
        assert ((med.isna()) == (lo.isna())).all() and ((med.isna()) == (hi.isna())).all()
    width = (df["logmstar_hi"] - df["logmstar_lo"])[phot]
    assert width.notna().all() and (width > 0).all()
    assert df.loc[:19, "logmstar"].isna().all()           # 99 -> no g -> no g-i mass
    assert df.loc[:19, "M_r"].isna().all()                # K_r needs g - r
    assert df.loc[25:, "M_r"].notna().all()
    assert df.loc[20:24, "logmstar_w1"].isna().all()      # inf in W2
    assert df["logmstar_w1"].notna().sum() > 0

    for col in ("col_gi", "col_gr", "col_ur", "col_w2w3_vega", "rs_offset", "quenched",
                "sf_w2w3", "mass_method", "class_spec", "aux_label", "aux_gid",
                "join_toy_sep_arcsec", "join_toy_ambiguous", "ps_R_mpc", "ps_dv_kms",
                "ps_R_over_r200", "ps_absdv_over_sigma"):
        assert col in df.columns, col
    assert df["aux_label"].notna().sum() == 50 and df["aux_label"][0] == "L\\_0"
    assert df["ps_dv_kms"][phot].isna().all() and df["ps_dv_kms"][spec].notna().all()
    assert str(df["quenched"].dtype) == "boolean"

    info = json.loads((root / "out" / "enriched.parquet.provenance.json").read_text())["build_info"]
    envs = info["environment"]
    assert [e["weight"] for e in envs] == ["count", "count", "mass"]
    assert envs[2]["n_tracers"] < envs[1]["n_tracers"]          # tracers without a mass excluded
    assert envs[1]["n_tracers"] == N_SPEC and envs[2]["n_tracers"] == N_SPEC - 20
    assert info["mass"]["n_mass_tracers"] == N_SPEC - 20
    assert info["photometry"]["set"] == "legacy"
    assert info["photometry"]["extinction"]["applied"] is True
    assert info["photometry"]["kcorrection"]["applied"] is True
    assert info["indicators"]["red_sequence"] is not None
    assert info["joins"][0]["n_rows_matched"] == 50
    assert info["phase_space"]["columns"]


def test_mass_draws_are_the_environment_draws(setup):
    root, raw, cfg = setup
    cfg = copy.deepcopy(cfg)
    cfg["extinction"] = {"ebv": None}
    cfg["kcorrection"] = {"apply": False}
    out = build(cfg, output=root / "o2" / "e.parquet")
    window = cfg["redshift_window"]
    df = assign_redshifts(load_catalogue(root / "data" / "cat.parquet", cfg["columns"]), window)
    phot = df[(df["z_type"] == "phot") & df["in_window"]].reset_index(drop=True)
    settings = {**cfg["photoz"], "pdf_max_std": 0.5}
    draws = make_draws(phot, "gauss", settings, {"sigma": 0.01, "bias": 0.0}, window, COSMO)
    values = phot["mag_i"].to_numpy()[draws.gal_idx] - COSMO.distmod(draws.z).value
    lo, med, hi = summarise(values, draws)
    is_phot = (out["z_type"] == "phot").to_numpy()
    np.testing.assert_allclose(out.loc[is_phot, "M_i"], med)
    np.testing.assert_allclose(out.loc[is_phot, "M_i_lo"], lo)
    np.testing.assert_allclose(out.loc[is_phot, "M_i_hi"], hi)
    # spec-z rows: m - DM at the spec-z
    spec = ~is_phot & out["M_i"].notna().to_numpy()          # the first 20 have no g
    assert spec.sum() == N_SPEC - 20
    np.testing.assert_allclose(out.loc[spec, "M_i"],
                               out.loc[spec, "mag_i"] - COSMO.distmod(out.loc[spec, "z_best"]).value)
    # the same seed reproduces the build
    again = build(cfg, output=root / "o3" / "e.parquet")
    np.testing.assert_array_equal(again["logmstar_lo"].to_numpy(), out["logmstar_lo"].to_numpy())


def test_extinction_and_kcorrection_switches(setup):
    root, raw, cfg = setup
    base = copy.deepcopy(cfg)
    base["extinction"] = {"ebv": None}
    base["kcorrection"] = {"apply": False}
    a = build(base, output=root / "a" / "e.parquet")
    ext = copy.deepcopy(base)
    ext["extinction"] = {"ebv": "ebv"}
    b = build(ext, output=root / "b" / "e.parquet")
    spec = ((a["z_type"] == "spec") & a["M_i"].notna()).to_numpy()
    # E(B-V) = 0.03: A_i = 1.592 * 0.03 makes the galaxy brighter, M_i more negative
    np.testing.assert_allclose((a["M_i"] - b["M_i"])[spec], 1.592 * 0.03, atol=1e-9)
    prov = json.loads((root / "b" / "e.parquet.provenance.json").read_text())["build_info"]
    assert prov["photometry"]["extinction"]["ebv_median"] == 0.03
    prov = json.loads((root / "a" / "e.parquet.provenance.json").read_text())["build_info"]
    assert prov["photometry"]["extinction"]["applied"] is False
    assert "not applied" in prov["photometry"]["extinction"]["note"]
    assert prov["photometry"]["kcorrection"]["applied"] is False


def test_mass_method_w1_and_fallback(setup):
    root, raw, cfg = setup
    cfg = copy.deepcopy(cfg)
    cfg["mass"] = {"method": "w1"}
    df = build(cfg, output=root / "w" / "e.parquet")
    np.testing.assert_array_equal(df["logmstar"].to_numpy(), df["logmstar_w1"].to_numpy())
    cfg["mass"] = {"method": "taylor", "fallback": "w1"}
    df = build(cfg, output=root / "f" / "e.parquet")
    # the 20 galaxies without g get the W1 mass
    assert df.loc[:19, "logmstar"].notna().all()
    assert (df.loc[:19, "mass_method"] == "w1").all()


def test_no_photometry_section_skips_phase3(setup):
    root, raw, cfg = setup
    cfg = copy.deepcopy(cfg)
    for key in ("photometry", "extinction", "kcorrection", "mass", "indicators"):
        cfg.pop(key)
    cfg["environment"] = cfg["environment"][:2]
    df = build(cfg, output=root / "n" / "e.parquet")
    assert "logmstar" not in df.columns and "quenched" not in df.columns
    assert "aux_label" in df.columns and "ps_R_mpc" in df.columns


def test_config_validation_phase3(tmp_path):
    def load(**changes):
        d = copy.deepcopy(CONFIG)
        d.update(changes)
        return _load(tmp_path, d)

    load()
    no_photo = {k: v for k, v in CONFIG.items() if k != "photometry"}
    with pytest.raises(ConfigError, match="photometry"):
        _load(tmp_path, no_photo)                                  # weight: mass needs masses
    with pytest.raises(ConfigError, match="weight"):
        load(environment=[{"name": "sigma2d", "k": 5, "weight": "light"}])
    with pytest.raises(ConfigError, match="photoz_mode"):
        load(environment=[{"name": "knn3d", "k": 5, "photoz_mode": "x"}])
    with pytest.raises(ConfigError, match="photometry.set"):
        load(photometry={"set": "nope"})
    with pytest.raises(ConfigError, match="photometry.foo"):
        load(photometry={"foo": 1})
    with pytest.raises(ConfigError, match="mass.method"):
        load(mass={"method": "magic"})
    with pytest.raises(ConfigError, match="colour_limits"):
        load(mass={"colour_limits": {"g_i": [2, 1]}})
    with pytest.raises(ConfigError, match="kcorrection.apply"):
        load(kcorrection={"apply": "yes"})
    with pytest.raises(ConfigError, match="red_sequence.clip"):
        load(indicators={"red_sequence": {"clip": [1]}})
    with pytest.raises(ConfigError, match="cluster"):
        load(cluster={"ra": 1.0, "z": 0.02})
    with pytest.raises(ConfigError, match="cluster.z"):
        load(cluster={"ra": 1.0, "dec": 2.0})
    with pytest.raises(ConfigError, match="radius_arcsec"):
        load(joins=[{"name": "a", "path": "x.csv", "columns": ["c"], "radius_arcsec": 0}])
    with pytest.raises(ConfigError, match="used twice"):
        j = {"name": "a", "path": "x.csv", "columns": ["c"]}
        load(joins=[j, dict(j)])
    # null sections and all-null cluster are fine
    load(cluster={"ra": None, "dec": None, "z": None, "r200_mpc": None, "sigma_v": None},
         extinction={"ebv": None})
