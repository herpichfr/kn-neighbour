from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from knn.build import build_in_memory, tracers_path
from knn.config import ConfigError, load_config, resolve_path
from knn.explorer import (ENV_CACHE_SIZE, Explorer, is_boolean_like, trend_table,
                          wilson_interval)

REPO = Path(__file__).resolve().parents[1]


# environment ---------------------------------------------------------------

def test_environment_cache_hit_is_same_object(explorer):
    a = explorer.environment("sigma2d", 5)
    assert explorer.environment("sigma2d", 5) is a
    assert explorer.environment("sigma2d", 5, "count", None) is a      # defaults spelled out
    assert explorer.environment("sigma2d", 6) is not a


def test_environment_cache_is_bounded(explorer):
    first = explorer.environment("knn3d", 3)
    for k in range(4, 4 + ENV_CACHE_SIZE + 2):
        explorer.environment("knn3d", k)
    assert len(explorer._env_cache) <= ENV_CACHE_SIZE
    assert explorer.environment("knn3d", 3) is not first               # evicted, recomputed


def test_knn3d_photoz_rows_nan_whatever_the_mode(explorer):
    for mode in (None, "gauss", "point", "pdf", "none"):
        env = explorer.environment("knn3d", 5, photoz_mode=mode)
        phot = explorer.is_phot
        assert env.loc[phot].isna().all().all()
        assert env.loc[explorer.is_spec, "log1p_delta_knn3d_k5"].notna().all()


def test_irrelevant_parameter_is_ignored_and_unknown_raises(explorer):
    a = explorer.environment("knn3d", 5, dv=900.0)
    assert explorer.environment("knn3d", 5, dv=1200.0) is a            # knn3d takes no dv
    with pytest.raises(TypeError):
        explorer.environment("sigma2d", 5, bogus=1)
    with pytest.raises(ValueError):
        explorer.environment("sigma2d", 5, weight="luminosity")
    with pytest.raises(ValueError):
        explorer.environment("sigma2d", 5, photoz_mode="nope")
    with pytest.raises(KeyError):
        explorer.environment("nonexistent", 5)


def test_default_valued_parameters_share_the_cache_entry(explorer):
    a = explorer.environment("sigma2d", 5)
    assert explorer.environment("sigma2d", 5, dv=1500.0, n_sigma=1.5) is a
    assert explorer.environment("sigma2d", 5, dv=None) is a


def test_sigma2d_parameters_change_the_result(explorer):
    base = explorer.environment("sigma2d", 5)["log1p_delta_sigma2d_k5"]
    wide = explorer.environment("sigma2d", 5, dv=4000.0)["log1p_delta_sigma2d_k5"]
    assert not np.allclose(base[explorer.is_spec], wide[explorer.is_spec])
    n_sig = explorer.environment("sigma2d", 5, n_sigma=3.0)["log1p_delta_sigma2d_k5"]
    np.testing.assert_array_equal(base[explorer.is_spec], n_sig[explorer.is_spec])   # spec: dv only
    assert not np.allclose(base[explorer.is_phot], n_sig[explorer.is_phot], equal_nan=True)


@pytest.mark.parametrize("name,k,weight", [("knn3d", 5, "count"), ("sigma2d", 5, "count"),
                                           ("sigma2d", 5, "mass")])
def test_default_parameters_reproduce_the_build(explorer, name, k, weight):
    suffix = "_mass" if weight == "mass" else ""
    env = explorer.environment(name, k, weight)
    assert len(env.columns) == 9
    for col in env.columns:
        assert col.endswith(("_lo", "_hi")) or col.endswith(f"_{name}_k{k}{suffix}")
        np.testing.assert_array_equal(env[col].to_numpy(), explorer.df[col].to_numpy())


def test_photoz_modes_use_the_same_draws_as_the_build(built, explorer):
    cfg, _ = built
    # the build with gauss/60/seed 3: an explorer evaluated in point mode must differ for
    # photo-z rows only
    gauss = explorer.environment("sigma2d", 5)
    point = explorer.environment("sigma2d", 5, photoz_mode="point")
    col = "log1p_delta_sigma2d_k5"
    np.testing.assert_array_equal(gauss.loc[explorer.is_spec, col], point.loc[explorer.is_spec, col])
    assert not np.allclose(gauss.loc[explorer.is_phot, col], point.loc[explorer.is_phot, col],
                           equal_nan=True)
    # a point draw has lo == hi == median
    p = point.loc[explorer.is_phot]
    ok = p[col].notna()
    np.testing.assert_array_equal(p.loc[ok, col], p.loc[ok, col + "_lo"])


def test_env_frame_has_env_lo_hi(explorer):
    frame = explorer.env("sigma2d", 5)
    assert list(frame.columns) == ["env", "env_lo", "env_hi"]
    np.testing.assert_array_equal(frame["env"], explorer.df["log1p_delta_sigma2d_k5"])


def test_default_environment_is_the_first_photoz_entry(explorer):
    d = explorer.default_environment()
    assert (d["name"], d["k"], d["weight"], d["photoz_mode"]) == ("sigma2d", 5, "count", "gauss")


def test_mass_weight_without_masses_raises(explorer):
    frame = explorer.tracers.copy()
    frame["logm_point"] = np.nan
    bare = Explorer(explorer.cfg, explorer.df, frame)
    with pytest.raises(ValueError, match="stellar masses"):
        bare.environment("sigma2d", 5, "mass")


# loading --------------------------------------------------------------------

def test_tracer_table_written_by_build_matches_in_memory(built):
    cfg, enriched = built
    on_disk = pd.read_parquet(tracers_path(resolve_path(cfg, cfg["output"])))
    df, tracers = build_in_memory(cfg)
    pd.testing.assert_frame_equal(on_disk, tracers)
    pd.testing.assert_frame_equal(enriched, df)
    inside = tracers["out_row"] >= 0
    np.testing.assert_array_equal(df["z_best"].to_numpy()[tracers.loc[inside, "out_row"]],
                                  tracers.loc[inside, "z_best"])
    assert (tracers["z_type"] == "spec").all()                  # tracers: spec


def test_from_config_rebuilds_when_stale_or_incomplete(built):
    cfg, _ = built
    stale = {**cfg, "_meta": {**cfg["_meta"], "sha256": "0" * 64}}
    ex = Explorer.from_config(stale)
    assert ex.source == "build"
    assert Explorer.from_config(cfg, rebuild=True).source == "build"
    tfile = tracers_path(resolve_path(cfg, cfg["output"]))
    saved = tfile.read_bytes()
    tfile.unlink()
    try:
        assert Explorer.from_config(cfg).source == "build"
    finally:
        tfile.write_bytes(saved)
    assert Explorer.from_config(cfg).source == "parquet"
    # the rebuilt explorer gives the same environment as the one read from disk
    np.testing.assert_array_equal(ex.environment("sigma2d", 5)["log1p_delta_sigma2d_k5"],
                                  Explorer.from_config(cfg).df["log1p_delta_sigma2d_k5"])


def test_config_app_section(tmp_path):
    from synthetic import BUILD_CONFIG, write_build_catalogue
    import yaml
    write_build_catalogue(tmp_path)
    (tmp_path / "config").mkdir()
    path = tmp_path / "config" / "a.yaml"
    path.write_text(yaml.safe_dump({**BUILD_CONFIG, "app": {"min_per_bin": 3}}))
    assert Explorer.from_config(load_config(path)).min_count == 3
    path.write_text(yaml.safe_dump({**BUILD_CONFIG, "app": {"min_per_bin": 0}}))
    with pytest.raises(ConfigError, match="min_per_bin"):
        load_config(path)
    path.write_text(yaml.safe_dump({**BUILD_CONFIG, "app": {"nope": 1}}))
    with pytest.raises(ConfigError, match="app.nope"):
        load_config(path)


# columns ----------------------------------------------------------------------

def test_column_lists(explorer):
    num = explorer.numeric_columns()
    assert num[0] == "env" and "logmstar" in num and "M_r" in num
    assert not any(c.endswith(("_lo", "_hi")) or c.startswith("join_") for c in num)
    assert "cx" not in num and "quenched" not in num
    assert "env" not in explorer.numeric_columns(include_env=False)
    boo = explorer.boolean_columns()
    assert "quenched" in boo and "sf_w2w3" in boo
    assert not any(c.startswith("join_") or c == "in_window" for c in boo)


def test_is_boolean_like():
    assert is_boolean_like(np.array([True, False]))
    assert is_boolean_like(pd.array([True, None], dtype="boolean"))
    assert not is_boolean_like(np.array([0.0, 1.0]))
    assert not is_boolean_like(pd.Series(["a", None]))


# select ----------------------------------------------------------------------

def test_select_without_filters_is_everything(explorer):
    assert explorer.select().all() and explorer.select({}).all()
    assert len(explorer.select()) == len(explorer.df)


def test_select_each_criterion(explorer):
    df, spec, phot = explorer.df, explorer.is_spec, explorer.is_phot
    sel = explorer.select

    z = df["z_best"].to_numpy()
    np.testing.assert_array_equal(sel({"z_range": [0.015, 0.025]}), (z >= 0.015) & (z <= 0.025))
    np.testing.assert_array_equal(sel({"z_types": ["spec"]}), spec)
    np.testing.assert_array_equal(sel({"z_types": ["phot"]}), phot)
    assert not sel({"z_types": ["__none__"]}).any()
    assert sel({"z_types": []}).all()                           # empty list = no filter

    p = df["p_in_window"].to_numpy()
    np.testing.assert_array_equal(sel({"p_in_window_min": 0.9}), spec | (p >= 0.9))
    odds = df["z_phot_odds"].to_numpy()
    np.testing.assert_array_equal(sel({"odds_min": 0.6}), ~phot | (odds >= 0.6))

    m = df["logmstar"].to_numpy()
    with np.errstate(invalid="ignore"):
        np.testing.assert_array_equal(sel({"logm_range": [8.0, 9.5]}), (m >= 8.0) & (m <= 9.5))
        mr = df["M_r"].to_numpy()
        np.testing.assert_array_equal(sel({"mr_max": -16.0}), mr <= -16.0)
    assert sel({"logm_range": [8.0, 9.5]})[np.isnan(m)].sum() == 0     # no mass: excluded

    fm = df["flag_member"].to_numpy()
    np.testing.assert_array_equal(sel({"flag_member": [1.0, 0.0]}), np.isin(fm, [1.0, 0.0]))

    sub = df["substructure"]
    np.testing.assert_array_equal(sel({"substructure": ["S1"]}), (sub == "S1").to_numpy())
    np.testing.assert_array_equal(sel({"substructure": ["(none)"]}), sub.isna().to_numpy())
    np.testing.assert_array_equal(sel({"substructure": ["S1", "(none)"]}),
                                  (sub.isna() | (sub == "S1")).to_numpy())

    grp = df["grp_photz_id_group_final"]
    np.testing.assert_array_equal(sel({"group": [2, 3]}), grp.isin([2, 3]).to_numpy(dtype=bool))
    np.testing.assert_array_equal(sel({"group": ["(none)"]}), grp.isna().to_numpy())

    q = df["quenched"]
    yes, no = sel({"quenched": "yes"}), sel({"quenched": "no"})
    np.testing.assert_array_equal(yes, (q == True).fillna(False).to_numpy(dtype=bool))  # noqa: E712
    np.testing.assert_array_equal(no, (q == False).fillna(False).to_numpy(dtype=bool))  # noqa: E712
    assert not (yes & no).any() and (yes | no).sum() == q.notna().sum()
    assert sel({"quenched": "any"}).all()

    morph = df["morph_Ser_n"].notna().to_numpy()
    assert morph.any() and not morph.all()
    np.testing.assert_array_equal(sel({"morphology": True}), morph)
    assert sel({"morphology": False}).all()


def test_select_combines_with_and_and_returns_copies(explorer):
    both = explorer.select({"z_types": ["spec"], "quenched": "yes"})
    np.testing.assert_array_equal(both, explorer.select({"z_types": ["spec"]})
                                  & explorer.select({"quenched": "yes"}))
    both[:] = False                                             # must not poison the cache
    assert explorer.select({"z_types": ["spec"], "quenched": "yes"}).any()


def test_select_unknown_filter_and_missing_column(explorer):
    with pytest.raises(ValueError, match="unknown filter"):
        explorer.select({"nonsense": 1})
    bare = Explorer(explorer.cfg, explorer.df.drop(columns=["M_r"]), explorer.tracers)
    with pytest.raises(KeyError, match="M_r"):
        bare.select({"mr_max": -15.0})


def test_counts(explorer):
    mask = explorer.select({"z_types": ["spec"]})
    assert explorer.counts(mask) == (int(explorer.is_spec.sum()), 0)


# trends ----------------------------------------------------------------------

def _step_data(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, n)
    mass = rng.uniform(8, 11, n)
    ztype = np.where(rng.random(n) < 0.7, "spec", "phot")
    return x, mass, ztype, rng


def test_trend_recovers_a_step_in_the_median():
    x, mass, ztype, rng = _step_data()
    y = np.where(x < 0.5, 1.0, 5.0) + rng.normal(0, 0.1, len(x))
    t = trend_table(x, y, None, mass, ztype, [8, 9, 10, 11], nbins=10, min_count=10)
    assert set(t["z_type"]) == {"spec", "phot"} and set(t["mass_bin"]) == {"8-9", "9-10", "10-11"}
    assert (t["kind"] == "median").all()
    low, high = t[t["x_hi"] <= 0.5], t[t["x_lo"] >= 0.5]
    assert len(low) and len(high)
    assert np.allclose(low["y"], 1.0, atol=0.05) and np.allclose(high["y"], 5.0, atol=0.05)
    assert ((t["y_lo"] <= t["y"]) & (t["y"] <= t["y_hi"]))[t["y"].notna()].all()
    # bands are the 16-84 percentiles: about +-0.1 for the noise
    assert np.allclose((high["y_hi"] - high["y_lo"]) / 2, 0.1, atol=0.03)
    assert t["n"].sum() == len(x)


def test_trend_recovers_a_step_in_the_fraction():
    x, mass, ztype, rng = _step_data(n=20000, seed=2)
    y = rng.random(len(x)) < np.where(x < 0.5, 0.2, 0.8)
    t = trend_table(x, y, None, mass, ztype, None, nbins=6)
    assert (t["kind"] == "fraction").all() and set(t["mass_bin"]) == {"all"}
    low, high = t[t["x_hi"] <= 0.5], t[t["x_lo"] >= 0.5]
    assert np.allclose(low["y"], 0.2, atol=0.05) and np.allclose(high["y"], 0.8, atol=0.05)
    assert ((t["y_lo"] <= t["y"]) & (t["y"] <= t["y_hi"])).all()
    # nullable boolean input gives the same table
    t2 = trend_table(x, pd.array(y, dtype="boolean"), None, mass, ztype, None, nbins=6)
    pd.testing.assert_frame_equal(t, t2)


def test_trend_respects_minimum_count_and_mask():
    rng = np.random.default_rng(1)
    x = np.concatenate([rng.uniform(0, 1, 100), [5.0] * 4])
    y = np.concatenate([np.zeros(100), np.ones(4)])
    t = trend_table(x, y, None, None, None, None, nbins=4, min_count=10, split_by_ztype=False)
    assert set(t["z_type"]) == {"all"}
    assert t["y"].notna().sum() == len(t) - t["y"].isna().sum()
    # bins of 26 galaxies are fine; shrink the sample and every bin falls below the count
    small = trend_table(x[:30], y[:30], None, None, None, None, nbins=4, min_count=10)
    assert small["y"].isna().all() and (small["n"] > 0).all()
    mask = np.zeros(len(x), dtype=bool)
    mask[:50] = True
    assert trend_table(x, y, mask, None, None, None, nbins=2, min_count=10)["n"].sum() == 50
    assert len(trend_table(x, y, np.zeros(len(x), dtype=bool), None, None, None)) == 0


def test_trend_ignores_nan_and_ties():
    x = np.array([1.0] * 20 + [np.nan] * 5)
    y = np.array([0.0] * 20 + [1.0] * 5)
    t = trend_table(x, y, None, None, None, None, nbins=5, min_count=5)
    assert len(t) == 1 and t.loc[0, "n"] == 20 and t.loc[0, "y"] == 0.0


def test_trend_mass_bin_edges_are_half_open_with_last_closed():
    x = np.arange(30.0)
    mass = np.array([8.0, 9.0, 10.0] * 10)
    t = trend_table(x, x, None, mass, None, [8, 9, 10], nbins=1, min_count=1,
                    split_by_ztype=False)
    n = dict(zip(t["mass_bin"], t["n"]))
    assert n == {"8-9": 10, "9-10": 20}                       # 10.0 closes the last bin


def test_explorer_trend_with_columns_and_arrays(explorer):
    mask = explorer.select({"z_types": ["spec"]})
    env = explorer.env("sigma2d", 5)["env"].to_numpy()
    a = explorer.trend(env, "quenched", mask, mass_bins=[7, 9, 11], nbins=4)
    b = explorer.trend("log1p_delta_sigma2d_k5", "quenched", mask, mass_bins=[7, 9, 11], nbins=4)
    pd.testing.assert_frame_equal(a, b)
    assert set(a["z_type"]) == {"spec"} and (a["kind"] == "fraction").all()
    assert set(explorer.trend(env, "logmstar", None, nbins=3)["kind"]) == {"median"}
    with pytest.raises(ValueError):
        explorer.trend(env[:5], "quenched", mask)


def test_wilson_interval_against_hand_values():
    # z = Phi^-1(0.84) = 0.994458; centre = (p + z^2/2n) / (1 + z^2/n),
    # half = z sqrt(p(1-p)/n + z^2/4n^2) / (1 + z^2/n), worked by hand for each case
    lo, hi = wilson_interval(0, 10)
    assert lo == pytest.approx(0.0, abs=1e-12) and hi == pytest.approx(0.0899, abs=1e-4)
    lo, hi = wilson_interval(5, 10)
    assert (lo, hi) == (pytest.approx(0.3500, abs=2e-4), pytest.approx(0.6500, abs=2e-4))
    lo, hi = wilson_interval(3, 10)
    assert (lo, hi) == (pytest.approx(0.1793, abs=2e-4), pytest.approx(0.4567, abs=2e-4))
    lo, hi = wilson_interval(10, 10)
    assert hi == pytest.approx(1.0, abs=1e-12) and lo == pytest.approx(0.9100, abs=2e-4)
    lo, hi = wilson_interval(np.array([5, 50]), np.array([10, 100]))
    assert hi[1] - lo[1] < hi[0] - lo[0]
    assert np.isnan(wilson_interval(0, 0)).all()
    lo95, hi95 = wilson_interval(5, 10, conf=0.95)            # textbook value: 0.237 - 0.763
    assert (lo95, hi95) == (pytest.approx(0.2366, abs=1e-3), pytest.approx(0.7634, abs=1e-3))


# snapshots --------------------------------------------------------------------

def test_snapshot_round_trip(explorer):
    state = {"z_range": (0.01, 0.03), "estimator": "sigma2d", "k": np.int64(7), "dv": 1500.0,
             "member": [1.0, 0.0], "mr_max": None, "density": ["on"], "nested": {"a": (1, 2)},
             "clip": np.array([2, 98])}
    text = explorer.snapshot(state)
    rec = explorer.load_snapshot(text)
    assert rec["state"] == {"z_range": [0.01, 0.03], "estimator": "sigma2d", "k": 7,
                            "dv": 1500.0, "member": [1.0, 0.0], "mr_max": None,
                            "density": ["on"], "nested": {"a": [1, 2]}, "clip": [2, 98]}
    assert rec["config_sha256"] == explorer.cfg["_meta"]["sha256"] and rec["matches_config"]
    assert rec["git_head"] is None or len(rec["git_head"]) == 40
    assert rec["knn_snapshot"] == 1 and "created_utc" in rec


def test_snapshot_detects_other_config_and_bad_text(explorer):
    rec = explorer.load_snapshot(explorer.snapshot({"k": 3}).replace(
        explorer.config_sha256, "f" * 64))
    assert rec["matches_config"] is False
    for bad in ("just text", "a: 1", "knn_snapshot: 1\nstate: 3", "knn_snapshot: 9\nstate: {}",
                "[unclosed"):
        with pytest.raises(ValueError):
            explorer.load_snapshot(bad)


# real data ---------------------------------------------------------------------

REAL = REPO / "config" / "mkw4.yaml"


@pytest.mark.skipif(not (REPO / "data" / "MKW4+v7.parquet").exists()
                    or not (REPO / "data" / "derived" / "MKW4_enriched.parquet").exists(),
                    reason="real data not available")
def test_real_data_default_environment_equals_enriched_columns():
    cfg = load_config(REAL)
    ex = Explorer.from_config(cfg)
    assert len(ex.df) == len(pd.read_parquet(resolve_path(cfg, cfg["output"])))
    d = ex.default_environment()
    assert (d["name"], d["k"]) == ("sigma2d", 10)
    for weight, suffix in (("count", ""), ("mass", "_mass")):
        env = ex.environment("sigma2d", 10, weight)
        for col in env.columns:
            # rtol: a parquet written before the sigma2d candidate search differs at 1e-10
            np.testing.assert_allclose(env[col].to_numpy(), ex.df[col].to_numpy(), rtol=1e-8,
                                       atol=0, equal_nan=True, err_msg=col)
    knn = ex.environment("knn3d", 10)
    assert knn.loc[ex.is_phot].isna().all().all()
    np.testing.assert_allclose(knn["log1p_delta_knn3d_k10"], ex.df["log1p_delta_knn3d_k10"],
                               rtol=1e-8, equal_nan=True)
    assert "quenched" in ex.boolean_columns()
