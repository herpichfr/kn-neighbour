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
    "output": "out/enriched.parquet",
}


def _write_catalogue(path):
    rng = np.random.default_rng(7)
    n_spec, n_phot, n_out, n_none = 200, 50, 20, 30
    n = n_spec + n_phot + n_out + n_none
    zspec = np.full(n, np.nan)
    zphot = np.full(n, np.nan)
    zspec[:n_spec] = rng.uniform(0.01, 0.03, n_spec)
    zphot[n_spec:n_spec + n_phot] = rng.uniform(0.01, 0.03, n_phot)
    sl = slice(n_spec + n_phot, n_spec + n_phot + n_out)
    zspec[sl] = 0.08                       # spec-z outside the window
    zphot[sl] = 0.02                       # photo-z inside: must still be dropped
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
    assert df.loc[phot, ["r_knn3d_k5", "rho_knn3d_k5", "log1p_delta_knn3d_k5"]].isna().all().all()
    assert df.loc[spec, "r_knn3d_k5"].notna().all()
    np.testing.assert_allclose(np.sqrt(df.cx ** 2 + df.cy ** 2 + df.cz ** 2), df.d_com)

    out = tmp_path / "out" / "enriched.parquet"
    assert out.is_file()
    assert len(pd.read_parquet(out)) == 250

    prov = json.loads((tmp_path / "out" / "enriched.parquet.provenance.json").read_text())
    assert prov["config"]["tracers"] == "spec"
    assert prov["config_sha256"] == cfg["_meta"]["sha256"]
    assert {"git_head", "timestamp_utc", "knn_version"} <= set(prov)


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
