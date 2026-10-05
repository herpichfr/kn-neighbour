import json

import numpy as np
import pandas as pd
import pytest
import yaml

from knn.build import build
from knn.config import ConfigError, load_config
from knn.io import load_catalogue

CONFIG = {
    "catalogue": "data/cat.parquet",
    "columns": {"ra": "RA", "dec": "DEC", "z_spec": "zspec", "z_spec_err": "zerr",
                "z_phot": "zphot", "z_phot_odds": "pz"},
    "cosmology": {"H0": 70, "Om0": 0.3},
    "redshift_window": [0.007, 0.037],
    "tracers": "spec",
    "environment": [{"name": "knn3d", "k": 5}],
    "photoz": {"mode": "gauss", "n_draws": 40, "seed": 1, "sigma": 0.01, "bias": 0.0},
    "output": "out/enriched.parquet",
}


def _write_catalogue(path):
    rng = np.random.default_rng(7)
    n_spec, n_phot, n_out, n_none, n_edge = 200, 50, 20, 30, 60
    n = n_spec + n_phot + n_out + n_none + n_edge
    zspec = np.full(n, np.nan)
    zphot = np.full(n, np.nan)
    zspec[:n_spec] = rng.uniform(0.01, 0.03, n_spec)
    zspec[n_spec - 40:n_spec] = rng.uniform(0.0355, 0.0369, 40)   # near the upper edge
    zphot[n_spec:n_spec + n_phot] = rng.uniform(0.01, 0.03, n_phot)
    sl = slice(n_spec + n_phot, n_spec + n_phot + n_out)
    zspec[sl] = 0.08                       # spec-z outside the window
    zphot[sl] = 0.02                       # photo-z inside: must still be dropped
    zspec[n - n_edge:] = rng.uniform(0.0371, 0.0374, n_edge)   # tracer window only
    df = pd.DataFrame({
        "RA": rng.uniform(180, 181, n), "DEC": rng.uniform(0, 1, n),
        "zspec": zspec, "zerr": 1e-4, "zphot": zphot, "pz": 0.9,
        "extra": np.arange(n),
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


@pytest.fixture
def cfg(tmp_path):
    _write_catalogue(tmp_path / "data" / "cat.parquet")
    cfg_path = tmp_path / "config" / "test.yaml"
    cfg_path.parent.mkdir()
    cfg_path.write_text(yaml.safe_dump(CONFIG))
    return load_config(cfg_path)


def test_build(cfg, tmp_path):
    df = build(cfg)
    assert len(df) == 250
    for col in ["ra", "dec", "z_spec", "z_phot", "z_best", "z_type", "in_window",
                "cx", "cy", "cz", "d_com", "extra",
                "r_knn3d_k5", "rho_knn3d_k5", "log1p_delta_knn3d_k5"]:
        assert col in df.columns
    spec = df["z_type"] == "spec"
    phot = df["z_type"] == "phot"
    assert spec.sum() == 200 and phot.sum() == 50
    # photo-z rows now carry the median over their redshift draws
    assert df.loc[phot, ["r_knn3d_k5", "rho_knn3d_k5", "log1p_delta_knn3d_k5"]].notna().all().all()
    assert df.loc[spec, "r_knn3d_k5"].notna().all()
    np.testing.assert_allclose(np.sqrt(df.cx ** 2 + df.cy ** 2 + df.cz ** 2), df.d_com)

    out = tmp_path / "out" / "enriched.parquet"
    assert out.is_file()
    assert len(pd.read_parquet(out)) == 250

    prov = json.loads((tmp_path / "out" / "enriched.parquet.provenance.json").read_text())
    assert prov["config"]["tracers"] == "spec"
    assert prov["config_sha256"] == cfg["_meta"]["sha256"]
    assert {"git_head", "timestamp_utc", "knn_version", "build_info"} <= set(prov)
    info = prov["build_info"]
    assert info["n_tracers"] == 200
    assert info["n_spec_targets"] == 200 and info["n_phot_targets"] == 50
    assert info["photoz"]["sigma_source"] == "config"


def test_output_override(cfg, tmp_path):
    out = tmp_path / "elsewhere" / "x.parquet"
    build(cfg, output=out)
    assert out.is_file()
    assert (tmp_path / "elsewhere" / "x.parquet.provenance.json").is_file()


def test_load_catalogue_csv(tmp_path):
    pd.DataFrame({"a": [1.0], "b": [2.0], "c": [3]}).to_csv(tmp_path / "t.csv", index=False)
    df = load_catalogue(tmp_path / "t.csv", {"x": "a"})
    assert list(df.columns) == ["x", "b", "c"]


def test_missing_config_key(tmp_path):
    bad = {k: v for k, v in CONFIG.items() if k != "cosmology"}
    p = tmp_path / "config" / "bad.yaml"
    p.parent.mkdir()
    p.write_text(yaml.safe_dump(bad))
    with pytest.raises(ConfigError, match="cosmology"):
        load_config(p)


def test_photoz_rows(cfg):
    df = build(cfg)
    spec = (df["z_type"] == "spec").to_numpy()
    phot = (df["z_type"] == "phot").to_numpy()
    for q in ("r", "rho", "log1p_delta"):
        col = f"{q}_knn3d_k5"
        med, lo, hi = df[col], df[col + "_lo"], df[col + "_hi"]
        assert np.isfinite(med[phot]).all()
        assert (lo[phot] <= med[phot]).all() and (med[phot] <= hi[phot]).all()
        assert (hi[phot] > lo[phot]).any()          # the draws spread the estimate
        np.testing.assert_array_equal(lo[spec], med[spec])
        np.testing.assert_array_equal(hi[spec], med[spec])
    p = df["p_in_window"]
    assert ((p >= 0) & (p <= 1)).all()
    assert (p[spec] == 1.0).all()
    assert (p[phot] > 0).all() and (p[phot] < 1).any()


def test_photoz_modes_and_override(cfg, tmp_path):
    cfg["environment"] = [{"name": "knn3d", "k": 5},
                          {"name": "sigma2d", "k": 5, "photoz_mode": "point"}]
    df = build(cfg)
    phot = (df["z_type"] == "phot").to_numpy()
    # sigma2d in point mode: a single draw, no spread
    col = "log1p_delta_sigma2d_k5"
    assert np.isfinite(df.loc[phot, col]).all()
    np.testing.assert_array_equal(df.loc[phot, col + "_lo"], df.loc[phot, col])
    np.testing.assert_array_equal(df.loc[phot, col + "_hi"], df.loc[phot, col])
    assert np.isfinite(df.loc[~phot, "sigma_sigma2d_k5"]).all()
    # knn3d in gauss mode: spread
    assert (df.loc[phot, "r_knn3d_k5_hi"] > df.loc[phot, "r_knn3d_k5_lo"]).any()

    cfg["photoz"]["mode"] = "point"
    cfg["environment"] = [{"name": "knn3d", "k": 5}]
    df = build(cfg, output=tmp_path / "point.parquet")
    phot = (df["z_type"] == "phot").to_numpy()
    np.testing.assert_array_equal(df.loc[phot, "r_knn3d_k5_lo"], df.loc[phot, "r_knn3d_k5"])


def test_photoz_pdf_mode(cfg, tmp_path):
    raw = pd.read_parquet(tmp_path / "data" / "cat.parquet")
    for j in range(3):
        raw[f"pdf_weights_{j}"] = [0.6, 0.4, 0.0][j]
        raw[f"pdf_means_{j}"] = raw["zphot"] + 0.002 * j
        raw[f"pdf_stds_{j}"] = [0.004, 0.006, 1.0][j]
    raw.to_parquet(tmp_path / "data" / "cat.parquet", index=False)
    cfg["photoz"]["mode"] = "pdf"
    df = build(cfg)
    phot = (df["z_type"] == "phot").to_numpy()
    assert np.isfinite(df.loc[phot, "r_knn3d_k5"]).all()
    assert (df.loc[phot, "p_in_window"] > 0).all()


def test_missing_sigma_is_an_error(cfg):
    # no spec-z / photo-z overlap in the catalogue and no sigma in the config
    cfg["photoz"] = {"mode": "gauss"}
    with pytest.raises(ValueError, match="sigma"):
        build(cfg)


def test_tracer_window(cfg, tmp_path):
    base = build(cfg)
    base_info = json.loads((tmp_path / "out" / "enriched.parquet.provenance.json").read_text())

    cfg["tracer_window"] = [0.002, 0.042]
    wide = build(cfg, output=tmp_path / "wide" / "enriched.parquet")
    wide_info = json.loads(
        (tmp_path / "wide" / "enriched.parquet.provenance.json").read_text())

    assert len(wide) == len(base)
    assert wide_info["build_info"]["n_spec_targets"] == base_info["build_info"]["n_spec_targets"]
    assert wide_info["build_info"]["n_phot_targets"] == base_info["build_info"]["n_phot_targets"]
    assert base_info["build_info"]["n_tracers"] == 200
    assert wide_info["build_info"]["n_tracers"] == 260
    # more tracers can only bring the k-th neighbour closer
    spec = (wide["z_type"] == "spec").to_numpy()
    assert (wide.loc[spec, "r_knn3d_k5"] <= base.loc[spec, "r_knn3d_k5"] + 1e-12).all()
    assert (wide.loc[spec, "r_knn3d_k5"] < base.loc[spec, "r_knn3d_k5"]).any()


def test_config_validation(tmp_path):
    def load(**changes):
        d = {**CONFIG, **changes}
        p = tmp_path / "config" / "c.yaml"
        p.parent.mkdir(exist_ok=True)
        p.write_text(yaml.safe_dump(d))
        return load_config(p)

    load(photoz={"mode": "pdf", "sigma": None, "bias": None})
    with pytest.raises(ConfigError, match="photoz.mode"):
        load(photoz={"mode": "magic"})
    with pytest.raises(ConfigError, match="photoz.n_draws"):
        load(photoz={"n_draws": 0})
    with pytest.raises(ConfigError, match="photoz.foo"):
        load(photoz={"foo": 1})
    with pytest.raises(ConfigError, match="tracer_window"):
        load(tracer_window=[0.01, 0.03])         # does not contain redshift_window
    with pytest.raises(ConfigError, match="tracer_window"):
        load(tracer_window=[0.05, 0.01])
    with pytest.raises(ConfigError, match=r"environment\[0\].dv"):
        load(environment=[{"name": "sigma2d", "k": 5, "dv": -1}])
    with pytest.raises(ConfigError, match="photoz_mode"):
        load(environment=[{"name": "knn3d", "k": 5, "photoz_mode": "x"}])
