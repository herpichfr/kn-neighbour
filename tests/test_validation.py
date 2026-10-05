import json

import numpy as np
import pandas as pd
import pytest
import yaml

from knn.cli import main
from knn.config import load_config
from knn.validation import format_summary, validate_photoz

CONFIG = {
    "catalogue": "data/cat.parquet",
    "columns": {"ra": "RA", "dec": "DEC", "z_spec": "zspec", "z_phot": "zphot",
                "z_phot_odds": "odds"},
    "cosmology": {"H0": 70, "Om0": 0.3},
    "redshift_window": [0.01, 0.03],
    "tracer_window": [0.005, 0.035],
    "tracers": "spec",
    "environment": [{"name": "knn3d", "k": 5}, {"name": "sigma2d", "k": 5}],
    "photoz": {"mode": "gauss", "n_draws": 30, "seed": 3},
    "output": "out/enriched.parquet",
}


def _catalogue(path):
    rng = np.random.default_rng(21)
    n = 800
    # clustered: a Gaussian clump in ra, dec, z on top of a uniform background
    clump = rng.random(n) < 0.5
    ra = np.where(clump, rng.normal(180.5, 0.1, n), rng.uniform(180, 181, n))
    dec = np.where(clump, rng.normal(0.5, 0.1, n), rng.uniform(0, 1, n))
    z = np.where(clump, rng.normal(0.02, 0.0015, n), rng.uniform(0.004, 0.036, n))
    sigma = 0.004
    zphot = z + sigma * (1 + z) * rng.standard_normal(n)
    zphot[:100] = np.nan                     # spec-z only
    zspec = z.copy()
    zspec[700:] = np.nan                     # photo-z only
    df = pd.DataFrame({"RA": ra, "DEC": dec, "zspec": zspec, "zphot": zphot,
                       "odds": rng.uniform(0.2, 1.0, n)})
    for j in range(3):
        df[f"pdf_weights_{j}"] = [0.7, 0.3, 0.0][j]
        df[f"pdf_means_{j}"] = df["zphot"] + [0.0, 0.003, 0.0][j]
        df[f"pdf_stds_{j}"] = [sigma, 2 * sigma, 1.0][j]
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


@pytest.fixture
def cfg_path(tmp_path):
    _catalogue(tmp_path / "data" / "cat.parquet")
    p = tmp_path / "config" / "test.yaml"
    p.parent.mkdir()
    p.write_text(yaml.safe_dump(CONFIG))
    return p


def test_validation_end_to_end(cfg_path, tmp_path):
    cfg = load_config(cfg_path)
    res = validate_photoz(cfg, output_dir=tmp_path / "val")
    path = tmp_path / "val" / "photoz_validation.json"
    assert path.is_file()
    on_disk = json.loads(path.read_text())
    assert on_disk["meta"]["n_validation"] == res["meta"]["n_validation"] > 300

    # empirical scatter is close to the injected 0.004
    assert abs(res["point_stats"]["sigma_nmad"] - 0.004) < 0.001
    assert res["meta"]["photoz"]["sigma_source"] == "empirical"

    n = res["meta"]["n_validation"]
    for key in ("mixture", "mixture_pdf_max_std", "gauss"):
        assert res["pit"][key]["n"] == n
        assert sum(res["pit"][key]["counts"]) == n
        assert len(res["pit"][key]["counts"]) == 10

    assert set(res["environment"]) == {"knn3d_k5", "sigma2d_k5"}
    for block in res["environment"].values():
        assert block["n_common"] > 100
        for mode in ("point", "gauss", "pdf"):
            overall = block["modes"][mode]["overall"]
            assert np.isfinite(overall["spearman"])
            assert 0.0 <= overall["quartile_acc"] <= 1.0
            if mode == "point":
                assert overall["coverage"] is None
            else:
                assert 0.0 <= overall["coverage"] <= 1.0
            assert set(block["modes"][mode]["odds_terciles"]) == {"low", "mid", "high"}
    # the clumpy field is recovered far better than chance
    knn = res["environment"]["knn3d_k5"]["modes"]
    assert knn["gauss"]["overall"]["spearman"] > 0.3
    assert knn["gauss"]["overall"]["quartile_acc"] > 0.3
    assert "knn3d_k5" in format_summary(res)


def test_cli_validate_photoz(cfg_path, tmp_path, capsys):
    out = tmp_path / "cli_out"
    assert main(["validate-photoz", str(cfg_path), "--output-dir", str(out)]) == 0
    assert (out / "photoz_validation.json").is_file()
    assert "PIT histograms" in capsys.readouterr().out
