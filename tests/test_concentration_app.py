import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from knn.app import callbacks as cb
from knn.app import create_app, figures
from knn.app.layout import (PICK_HINT, STATE_CONTROLS, STATE_KEYS, STRUCTURE_HEADER,
                            conc_weight_options, default_state, default_struct, mass_table_rows)
from knn.build import tracers_path
from knn.concentration import (COLUMNS, ConcentrationError, default_spec, peak_table,
                               ridge_table)
from knn.config import resolve_path
from knn.explorer import MASS_TABLE_COLUMNS, mass_stats_table, wilson_interval
from dash import no_update
from conftest import CONC_SECTION

# the fixtures conc_built and conc_explorer are in conftest.py
SKY_TRACES = ["concentration", "survey area", "contours (2σ, 3σ, ...)", "1σ contour",
              "ridges halo", "ridges to the border halo", "ridges", "ridges to the border",
              "ridges off", "drawn paths", "drawn paths off", "peaks", "spec-z", "photo-z"]
MODEL_TRACES = ["model: 1σ band", "model: peak circles", "model: fitted paths"]
PEAKS, GALAXIES = SKY_TRACES.index("peaks"), SKY_TRACES.index("spec-z")


def output_path(cfg):
    return resolve_path(cfg, cfg["output"])


@pytest.fixture(scope="module")
def conc_stores(conc_explorer):
    state = default_state(conc_explorer)
    return state, cb.sample_info(conc_explorer, state), cb.env_info(conc_explorer, state)


def test_build_writes_columns_and_summary_but_no_sidecar(conc_built):
    cfg, df = conc_built
    assert set(COLUMNS) <= set(df.columns) and df["conc_inside"].dtype == bool
    prov = json.loads(Path(str(output_path(cfg)) + ".provenance.json").read_text())
    info = prov["build_info"]["concentration"]
    assert prov["config"]["concentration"] == CONC_SECTION
    assert info["n_peaks"] >= 2 and info["n_rows"] == len(df)
    assert {"mean", "sigma", "level_1sigma", "contour_levels", "n_ridges", "pixel_mpc",
            "counts"} <= set(info) and info["knn_scale_mpc"] > 0 and info["k"] == 20
    assert "lines" not in json.dumps(info) and "points" not in json.dumps(info)
    assert [k for k in prov["build_info"] if k.endswith("_file")] == ["tracers_file"]
    out = output_path(cfg)                                # no sidecar next to the catalogue
    assert {f.name for f in out.parent.iterdir()} == {
        out.name, tracers_path(out).name, out.name + ".provenance.json"}
    sample = df["z_best"].between(0.014, 0.028)
    assert info["n_sample"] == sample.sum()
    counts = info["counts"]
    assert counts["inside_spec"] + counts["inside_phot"] == (df["conc_inside"] & sample).sum()
    assert 0.02 < df.loc[sample, "conc_inside"].mean() < 0.6
    np.testing.assert_array_equal(df["conc_inside"], df["conc_density"] >= info["level_1sigma"])
    assert (df["conc_region"] >= 0).sum() == df["conc_inside"].sum()
    assert np.isfinite(df["conc_density"]).all() and np.isfinite(df["conc_ridge_dist_mpc"]).all()


def test_explorer_keeps_the_build_columns_and_reproduces_them_live(conc_explorer):
    ex = conc_explorer
    assert ex.conc_settings is not None and "conc_inside" in ex.boolean_columns()
    assert "conc_ridge_dist_mpc" in ex.numeric_columns()
    columns, result = ex.concentration()                   # the config sample, the config settings
    np.testing.assert_array_equal(columns["conc_inside"], ex.df["conc_inside"].to_numpy())
    np.testing.assert_allclose(columns["conc_density"], ex.df["conc_density"].to_numpy(),
                               equal_nan=True)
    again = ex.concentration()
    assert again[0] is columns and again[1] is result      # cached
    other = ex.concentration(weight="count")
    assert other[1]["info"]["settings"]["weight"] == "count" and other is not again
    assert ex.default_concentration() == {"weight": "mass", "k": 20, "pixel_mpc": None,
                                          "contour_step_sigma": 1.0,
                                          "min_prominence_sigma": 6.0, "ridge_floor_sigma": 1.0}
    with pytest.raises(ConcentrationError, match="only 3 galaxies"):
        ex.concentration(np.arange(len(ex.df)) < 3)


def test_inside_outside_sample_filter(conc_explorer):
    ex = conc_explorer
    base = ex.select({"z_range": [0.014, 0.028]})
    columns, _ = ex.concentration(base)
    inside = columns["conc_inside"]
    got = ex.select({"z_range": [0.014, 0.028], "conc_region": "inside"})
    np.testing.assert_array_equal(got, base & inside)
    out = ex.select({"z_range": [0.014, 0.028], "conc_region": "outside", "conc_params": {}})
    np.testing.assert_array_equal(out, base & ~inside)
    assert ex.select({"conc_region": "any"}).all()
    # the map is made of the galaxies passing the other filters: another slice, another map
    spec = ex.select({"z_range": [0.014, 0.028], "z_types": ["spec"], "conc_region": "inside"})
    assert spec.sum() > 0 and not (spec & ex.is_phot).any()
    with pytest.raises(ValueError, match="conc_region"):
        ex.select({"conc_region": "near"})
    state = cb.make_state(ex, concregion="inside", concw="count", conck=12, concstep=2.0,
                          concprom=0.5, concfloor=3.0)
    f = cb.filters_from_state(ex, state)
    assert f["conc_region"] == "inside"
    assert f["conc_params"] == {"weight": "count", "k": 12, "pixel_mpc": None,
                                "contour_step_sigma": 2.0, "min_prominence_sigma": 0.5,
                                "ridge_floor_sigma": 3.0}
    # a whole float from the number input is an int; an empty input falls back to the config
    assert cb.conc_params(cb.make_state(ex, conck=12.0))["k"] == 12
    assert isinstance(cb.conc_params(cb.make_state(ex, conck=12.0))["k"], int)
    assert cb.conc_params(cb.make_state(ex, conck=None, concfloor=None))["k"] is None
    assert cb.conc_params(cb.make_state(ex, conck=None, concfloor=None))["ridge_floor_sigma"] is None
    assert "conc_region" not in cb.filters_from_state(ex, default_state(ex))


def test_the_map_pixel_control_and_state_key(conc_explorer):
    ex = conc_explorer
    st = default_state(ex)
    assert st["concpix"] is None and "concpix" in cb.SAMPLE_KEYS
    assert ("concpix", "conc-pix", "value") in STATE_CONTROLS
    app = create_app(ex)
    box = next(c for c in app.layout._traverse() if getattr(c, "id", None) == "conc-pix")
    assert (box.type, box.min, box.step, box.placeholder, box.debounce) == (
        "number", 0.05, 0.05, "h/2", True) and box.value is None and not box.disabled
    # the control is the pixel_mpc setting; empty = h / 2
    assert cb.conc_params(cb.make_state(ex, concpix=0.6))["pixel_mpc"] == 0.6
    assert cb.conc_params(cb.make_state(ex, concpix=None))["pixel_mpc"] is None
    default = cb.sample_info(ex, cb.make_state(ex))
    coarse = cb.sample_info(ex, cb.make_state(ex, concpix=0.6))
    assert coarse["conc"]["pixel_mpc"] == 0.6 and coarse["map_key"] != default["map_key"]
    res0 = cb.concentration_result(ex, default)[1]["info"]
    res1 = cb.concentration_result(ex, coarse)[1]["info"]
    assert res0["pixel_mpc"] == pytest.approx(res0["knn_scale_mpc"] / 2.0)
    assert res1["pixel_mpc"] == 0.6 and res1["grid_nx"] < res0["grid_nx"] * res0["pixel_mpc"] / 0.5
    # the summary shows the pixel used and the grid
    lines = cb.concentration_callback(ex, cb.env_info(ex, cb.make_state(ex)), coarse)[0]
    assert f"cell 0.60 Mpc ({res1['grid_nx']} x {res1['grid_ny']} cells, mean of 3^2 sub-points)" in lines[0]
    # in snapshots, and a snapshot without it still loads
    text = ex.snapshot(cb.make_state(ex, concpix=0.6))
    values, _ = cb.restore_state(ex, ex.load_snapshot(text))
    assert dict(zip(STATE_KEYS, values))["concpix"] == 0.6
    old = {k: v for k, v in st.items() if k != "concpix"}
    assert dict(zip(STATE_KEYS, cb.restore_state(ex, {"state": old})[0]))["concpix"] is None


def test_default_state_follows_the_config_slice(conc_explorer):
    st = default_state(conc_explorer)
    assert st["z_range"] == [0.014, 0.028] and st["z_types"] == ["spec", "phot"]
    assert (st["concw"], st["conck"], st["concstep"], st["concprom"], st["concfloor"],
            st["concregion"]) == ("mass", 20, 1.0, 6.0, 1.0, "any")
    assert "coords" not in st and "conch" not in st
    options = conc_weight_options(conc_explorer)
    assert options[:2] == ["count", "mass"] and "logmstar" in options and "env" not in options


def test_sample_info_carries_the_controls_and_survives_a_tiny_sample(conc_explorer):
    ex = conc_explorer
    info = cb.sample_info(ex, default_state(ex))
    assert info["conc"] == {"weight": "mass", "k": 20, "pixel_mpc": None, "contour_step_sigma": 1.0,
                            "min_prominence_sigma": 6.0, "ridge_floor_sigma": 1.0}
    tiny = cb.sample_info(ex, cb.make_state(ex, concregion="inside", z_range=[0.0140, 0.0141]))
    assert "1 sigma region filter off" in tiny["text"] and "conc_region" not in tiny["filters"]
    narrow = cb.sample_info(ex, cb.make_state(ex, concregion="inside"))
    assert "conc_region" in narrow["filters"] and narrow["n_spec"] + narrow["n_phot"] > 0


def test_concentration_callback_returns_summary_and_table_that_follow_sample_and_selection(
        conc_explorer, conc_stores):
    ex = conc_explorer
    state, sample, env = conc_stores
    lines, table, caption = cb.concentration_callback(ex, env, sample)
    assert lines[0].startswith("Map of ") and "Peaks: " in lines[2]
    assert table.className == "conc-table"
    head, body = table.children
    assert [c.children for c in head.children.children] == [
        "statistic", "inside spec-z", "inside photo-z", "inside all", "outside spec-z",
        "outside photo-z", "outside all"]
    rows = [[c.children for c in tr.children] for tr in body.children]
    assert rows[0][0] == "N galaxies" and len(rows) >= 8
    columns, result = cb.concentration_result(ex, sample)
    mask = ex.select(sample["filters"])
    n_in = int((columns["conc_inside"] & mask).sum())
    assert int(rows[0][3]) == n_in and int(rows[0][6]) == int(mask.sum()) - n_in
    assert int(rows[0][1]) + int(rows[0][2]) == n_in
    assert f"Selected: {n_in} inside" in caption and "1 sigma contour" in caption
    stats = [r[0] for r in rows]
    assert any(s.startswith("distance to the nearest ridge") for s in stats)
    assert any(s.startswith("environment") for s in stats)
    # a sky selection restricts the table, not the map; z types restrict both
    inside_rows = list(np.flatnonzero(columns["conc_inside"] & mask)[:6])
    _, table2, caption2 = cb.concentration_callback(ex, env, sample, inside_rows)
    rows2 = [[c.children for c in tr.children] for tr in table2.children[1].children]
    assert rows2[0][3] == "6" and rows2[0][6] == "0" and "6 inside, 0 outside" in caption2
    spec_sample = cb.sample_info(ex, cb.make_state(ex, z_types=["spec"]))
    rows3 = [[c.children for c in tr.children]
             for tr in cb.concentration_callback(ex, env, spec_sample)[1].children[1].children]
    assert rows3[0][2] == "0" and rows3[0][5] == "0"
    # no map for a sample that is too small
    empty = cb.sample_info(ex, cb.make_state(ex, z_range=[0.0140, 0.0141]))
    assert cb.concentration_callback(ex, env, empty)[0][0].startswith("No map")


def test_sky_figure_has_map_contours_ridges_and_peaks(conc_explorer, conc_stores):
    ex = conc_explorer
    state, sample, env = conc_stores
    sky = cb.sky_callback(ex, state, env, sample)
    assert [t.name for t in sky.data] == SKY_TRACES
    heat, hull, others, first = sky.data[0], sky.data[1], sky.data[2], sky.data[3]
    z = np.array(heat.z, dtype=float)
    assert z.shape == (len(heat.y), len(heat.x)) and np.isnan(z).any() and np.isfinite(z).any()
    assert heat.opacity == 1.0 and heat.zmin == figures.MAP_FLOOR == -1.0
    # log10(map / mean): floored at 0.1 x the mean, finite everywhere in the survey, the top is
    # the 99.5th percentile (at least 1), the hover shows the linear map / mean
    linear = np.array(cb.concentration_overlay(ex, sample)["map_z"], dtype=float)
    assert np.array_equal(np.isnan(z), np.isnan(linear)) and np.nanmin(z) >= -1.0
    np.testing.assert_allclose(z[np.isfinite(z)], np.log10(np.maximum(
        linear[np.isfinite(linear)], 0.1)))
    assert heat.zmax == pytest.approx(np.log10(max(np.percentile(linear[np.isfinite(linear)], 99.5),
                                                   1.0)))
    np.testing.assert_allclose(np.array(heat.customdata, dtype=float), linear, atol=5e-4,
                               equal_nan=True)
    assert "%{customdata:.2f}" in heat.hovertemplate and "%{z" not in heat.hovertemplate
    assert heat.colorbar.title.text == "map / mean"
    assert list(heat.colorbar.tickvals) == [np.log10(float(t)) for t in heat.colorbar.ticktext]
    assert all(-1.0 - 1e-9 <= v <= heat.zmax + 1e-9 for v in heat.colorbar.tickvals)
    assert "1" in heat.colorbar.ticktext
    assert heat.colorscale[0][1].lower().startswith("#0") and heat.colorscale[-1][1].lower().startswith("#f")
    # the border of the survey: a thin closed grey line
    assert hull.line.color == figures.HULL and hull.line.width <= 1.0
    assert hull.x[0] == hull.x[-1] and hull.y[0] == hull.y[-1] and None not in list(hull.x)
    # contours: the 1 sigma one heavier than the others, lines separated by None
    assert first.line.width > others.line.width and first.line.color == others.line.color
    assert None in list(first.x) and len(first.x) == len(first.y)
    assert len(others.x) > 0 and first.customdata is None
    # ridges: cyan over wider dark halos, a halo hides with its line, open ones dashed
    ridges, open_ridges = sky.data[6], sky.data[7]
    halo, halo_open = sky.data[4], sky.data[5]
    assert ridges.line.color == figures.RIDGE and open_ridges.line.dash == "dash"
    assert ridges.line.dash == "solid" and halo.line.width > ridges.line.width
    assert halo.legendgroup == ridges.legendgroup and halo.showlegend is False
    assert halo_open.legendgroup == open_ridges.legendgroup
    assert list(halo.x) == list(ridges.x)
    # the planted bridge between the two clumps makes at least one ridge
    assert len(ridges.x) > 5 or len(open_ridges.x) > 5
    # peaks: an SVG trace for legend and hover, at least the two clumps
    peaks = sky.data[PEAKS]
    assert type(peaks).__name__ == "Scatter" and peaks.marker.symbol == "star"
    assert len(peaks.x) >= 2 and len(peaks.text) == len(peaks.x)
    assert peaks.text[0].startswith("peak 1, ") and "σ, prominence " in peaks.text[0]
    # and one annotation per significant peak, on top of the WebGL canvas: ★ closed, ☆ open
    result = cb.concentration_result(ex, sample)[1]
    notes = sky.layout.annotations
    assert len(notes) == len(result["peaks"]) == len(peaks.x)
    for note, peak, text in zip(notes, result["peaks"], peaks.text):
        assert note.text == ("☆" if peak["open"] else "★") and note.showarrow is False
        assert note.font.size >= 20 and note.font.color == "#ffffff" and note.hovertext == text
        assert (note.xref, note.yref) == ("x", "y")
    assert [n.x for n in notes] == list(peaks.x) and [n.y for n in notes] == list(peaks.y)
    again = cb.sky_callback(ex, state, env, sample)             # rebuilt, never accumulated
    assert len(again.layout.annotations) == len(notes)
    # no zero lines (the Dec = 0 line looks like a contour); the colour bar is clear of the axis
    assert sky.layout.xaxis.zeroline is False and sky.layout.yaxis.zeroline is False
    assert heat.colorbar.yanchor == "top" and heat.colorbar.y <= -0.25
    assert sky.layout.margin.b >= 140
    # galaxies are on top (drawn last) and keep their customdata
    assert sky.data[GALAXIES].customdata is not None and sky.data[GALAXIES].name == "spec-z"
    # the map follows the controls: count weight, other k (a smaller kNN scale), density switch off
    count_state = cb.make_state(ex, concw="count")
    count = cb.sky_callback(ex, count_state, env, cb.sample_info(ex, count_state))
    zc = np.array(count.data[0].z, dtype=float)
    assert zc.shape == z.shape and not np.allclose(np.nan_to_num(zc), np.nan_to_num(z))
    narrow_state = cb.make_state(ex, conck=12)
    narrow = cb.sky_callback(ex, narrow_state, env, cb.sample_info(ex, narrow_state))
    assert np.array(narrow.data[0].z, dtype=float).shape != z.shape      # a finer cell
    off = cb.sky_callback(ex, cb.make_state(ex, density=[]), env, sample)
    names = [t.name for t in off.data]
    assert "concentration" not in names and "peaks" in names and "1σ contour" in names
    other_slice = cb.sample_info(ex, cb.make_state(ex, z_range=[0.012, 0.03]))
    assert cb.concentration_result(ex, other_slice)[1]["info"]["n_sample"] > \
        cb.concentration_result(ex, sample)[1]["info"]["n_sample"]
    # the footprint does not follow the slice
    assert cb.concentration_result(ex, other_slice)[1]["info"]["hull_area_mpc2"] == \
        cb.concentration_result(ex, sample)[1]["info"]["hull_area_mpc2"]


def test_app_layout_and_http_callback(conc_explorer, conc_stores):
    ex = conc_explorer
    app = create_app(ex)
    ids = {c.id for c in app.layout._traverse() if getattr(c, "id", None)}
    assert {"conc-summary", "conc-table", "conc-caption", "f-concregion", "conc-weight",
            "conc-k", "conc-step", "conc-prom", "conc-floor", "model-table", "sky-pick", "btn-pick-all",
            "btn-pick-none", "btn-clear-paths", "btn-run-model", "btn-clear-model", "pick-info",
            "model-status", "store-struct", "store-model"} <= ids
    assert "conc-table.children" in " ".join(app.callback_map)
    state, sample, env = conc_stores
    payload = {"output": "..conc-summary.children...conc-table.children...conc-caption.children"
                         "...model-table.children..",
               "outputs": [{"id": "conc-summary", "property": "children"},
                           {"id": "conc-table", "property": "children"},
                           {"id": "conc-caption", "property": "children"},
                           {"id": "model-table", "property": "children"}],
               "inputs": [{"id": "store-sample", "property": "data", "value": sample},
                          {"id": "store-env", "property": "data", "value": env},
                          {"id": "store-sel", "property": "data", "value": None}],
               "changedPropIds": ["store-env.data"]}
    resp = app.server.test_client().post("/_dash-update-component", json=payload)
    assert resp.status_code == 200, resp.data
    body = resp.get_json()["response"]
    assert body["conc-table"]["children"]["type"] == "Table"
    assert body["conc-summary"]["children"][0]["props"]["children"].startswith("Map of")
    assert "Selected:" in body["conc-caption"]["children"]
    assert body["model-table"]["children"]["type"] == "Div"        # no model: nothing to show
    # the sky callback over HTTP draws the map, the contours, the ridges and the peaks
    inputs = [("store-sample", "data", sample), ("store-env", "data", env),
              ("clear-selection", "n_clicks", 0), ("view-color", "value", "env"),
              ("view-clip", "value", [2, 98]), ("view-size", "value", None),
              ("sky-density", "value", ["on"]), ("store-struct", "data", None),
              ("sky-pick", "value", ["paths"])]
    sky_payload = {"output": "sky-fig.figure", "outputs": {"id": "sky-fig", "property": "figure"},
                   "inputs": [{"id": i, "property": p, "value": v} for i, p, v in inputs],
                   "changedPropIds": ["store-env.data"]}
    resp = app.server.test_client().post("/_dash-update-component", json=sky_payload)
    assert resp.status_code == 200, resp.data
    figure = resp.get_json()["response"]["sky-fig"]["figure"]
    assert [t["name"] for t in figure["data"]] == SKY_TRACES
    assert figure["layout"]["dragmode"] == "drawopenpath"
    # in the drawing mode the ridges that are on are editable shapes named 'ridge <id>'
    shapes = figure["layout"]["shapes"]
    assert [s["name"] for s in shapes] == ["ridge 1"] and shapes[0]["editable"] is True
    assert shapes[0]["type"] == "path" and shapes[0]["path"].startswith("M")


def test_a_sample_without_ridges_still_gives_a_panel_and_a_figure(conc_explorer, conc_stores):
    ex = conc_explorer
    state, _, env = conc_stores
    none = cb.sample_info(ex, cb.make_state(ex, concprom=1000.0))   # only the global maximum
    lines, table, caption = cb.concentration_callback(ex, env, none)
    assert any("Peaks: 1 significant" in line and "ridges (floor 1 sigma, 1 peaks above it)" in line
               for line in lines)
    rows = [[c.children for c in tr.children] for tr in table.children[1].children]
    assert int(rows[0][6]) + int(rows[0][3]) == int(ex.select(none["filters"]).sum())
    sky = cb.sky_callback(ex, state, env, none)
    assert [t.name for t in sky.data] == SKY_TRACES
    assert list(sky.data[6].x) == [] and list(sky.data[7].x) == []
    assert len(sky.data[PEAKS].x) == 1 and len(sky.layout.annotations) == 1


def test_live_recompute_is_fast_and_cached(conc_explorer, conc_stores):
    import time
    ex = conc_explorer
    state, sample, env = conc_stores
    sample = cb.sample_info(ex, cb.make_state(ex, conck=12))
    t0 = time.perf_counter()
    cb.concentration_result(ex, sample)
    first = time.perf_counter() - t0
    t0 = time.perf_counter()
    cb.concentration_result(ex, sample)
    assert first < 5.0 and time.perf_counter() - t0 < max(0.05, first)


def test_without_concentration_there_is_no_panel_and_the_filter_is_ignored(explorer):
    assert explorer.conc_settings is None and "conc_inside" not in explorer.df.columns
    app = create_app(explorer)
    ids = {c.id for c in app.layout._traverse() if getattr(c, "id", None)}
    assert "conc-table" not in ids and {"f-concregion", "conc-weight"} <= ids
    assert "conc-table.children" not in " ".join(app.callback_map)
    assert {"sky-pick", "store-struct", "store-model", "btn-run-model"} <= ids
    assert not any(k.startswith(("store-struct", "store-model")) for k in app.callback_map)
    hidden = next(c for c in app.layout._traverse() if getattr(c, "id", None) == "pick-controls")
    assert hidden.style == {"display": "none"}
    state = cb.make_state(explorer, concregion="inside")
    assert "conc_region" not in cb.filters_from_state(explorer, state)
    sample = cb.sample_info(explorer, state)
    assert sample["conc"] is None and cb.concentration_result(explorer, sample) is None
    assert cb.concentration_overlay(explorer, sample) is None
    with pytest.raises(ValueError, match="no 'concentration' section"):
        explorer.concentration()
    assert default_state(explorer)["z_range"] == list(explorer.window)


def test_table_statistics_on_a_toy_frame():
    # rows 0-5 inside (spec: 0-3, phot: 4-5), rows 6-11 outside (spec: 6-8, phot: 9-11)
    frame = pd.DataFrame({
        "logmstar": [9.0, 10.0, 11.0, 8.0, 7.0, np.nan, 9.0, 9.0, 9.0, 8.0, 8.0, 8.0],
        "p_in_window": [1, 1, 1, 1, 0.5, 0.5, 1, 1, 1, 0.8, 0.8, 0.8],
        "conc_log_density": np.arange(12, dtype=float),
        "conc_ridge_dist_mpc": np.arange(12, dtype=float) / 10.0,
        "quenched": pd.array([True, True, False, None, False, False, True, False, False, False,
                              False, True], dtype="boolean"),
        "morph_Ser_n": [1.0, 2.0, 3.0, np.nan] + [np.nan] * 8,
        "grp_photz_id_group_final": pd.array([-1, -1, 3, None, -1, -1, 2, 2, -1, 0, -1, -1],
                                             dtype="Int64"),
    })
    inside = np.arange(12) < 6
    z = np.array(["spec"] * 4 + ["phot"] * 2 + ["spec"] * 3 + ["phot"] * 3, dtype=object)
    env = np.arange(12, dtype=float) / 10.0
    t = mass_stats_table(frame, inside, z, env=env)
    assert list(t.columns) == list(MASS_TABLE_COLUMNS)

    def get(stat, region, ztype):
        return t[(t["stat"] == stat) & (t["region"] == region) & (t["z_type"] == ztype)].iloc[0]

    assert get("n", "inside", "spec")["value"] == 4 and get("n", "outside", "all")["n"] == 6
    m = get("logm", "inside", "spec")                                  # 9, 10, 11, 8
    assert (m["value"], m["n"]) == (9.5, 4)
    assert (m["lo"], m["hi"]) == tuple(np.percentile([9, 10, 11, 8], [16, 84]))
    assert get("logm", "inside", "phot")["n"] == 1                     # the NaN mass is skipped
    total = get("mtot", "inside", "all")                               # 1e9 + 1e10 + 1e11 + 1e8 + 0.5e7
    assert total["value"] == pytest.approx(np.log10(1e9 + 1e10 + 1e11 + 1e8 + 0.5e7))
    assert get("mtot", "outside", "phot")["value"] == pytest.approx(np.log10(3 * 0.8e8))
    q = get("quenched", "inside", "spec")                              # True, True, False, NA
    lo, hi = wilson_interval(2, 3, 0.68)
    assert (q["value"], q["n"]) == (pytest.approx(2 / 3), 3)
    assert (q["lo"], q["hi"]) == (pytest.approx(float(lo)), pytest.approx(float(hi)))
    assert get("quenched", "outside", "spec")["value"] == pytest.approx(1 / 3)
    g = get("in_group", "outside", "spec")                             # ids 2, 2, -1: 2 of 3
    assert (g["value"], g["n"]) == (pytest.approx(2 / 3), 3)
    assert get("in_group", "inside", "spec")["n"] == 3                 # one unmatched row
    assert get("in_group", "inside", "spec")["value"] == pytest.approx(1 / 3)
    assert get("conc", "outside", "all")["value"] == 8.5               # median of 6..11
    assert get("dist", "inside", "all")["value"] == pytest.approx(0.25)
    assert get("env", "inside", "all")["value"] == pytest.approx(0.25)
    s = get("ser_n", "inside", "spec")
    assert (s["value"], s["n"]) == (2.0, 3)
    empty = get("ser_n", "outside", "phot")
    assert empty["n"] == 0 and np.isnan(empty["value"]) and np.isnan(empty["lo"])
    stats = set(t["stat"])                                             # missing columns are left out
    assert "col_gr" not in stats and "morph_c" not in stats and "r_r200" not in stats
    assert "env" not in set(mass_stats_table(frame, inside, z)["stat"])
    only = mass_stats_table(frame, inside, z, mask=np.arange(12) < 3)  # a mask restricts every group
    sub = only[(only["stat"] == "n")]
    assert sub[(sub["region"] == "inside") & (sub["z_type"] == "all")].iloc[0]["value"] == 3
    assert sub[sub["region"] == "outside"]["value"].sum() == 0
    rows = mass_table_rows(t)
    assert rows[0][0] == "statistic" and rows[1][:2] == ["N galaxies", "4"]
    assert rows[2][1].startswith("9.50 [")


def test_tracers_file_is_still_written(conc_built):
    cfg, _ = conc_built
    assert tracers_path(output_path(cfg)).is_file()


# ---- the interactive sky map: picking, drawing, running the model ---------------------------


def click(kind, ident):
    """A Graph's clickData on a vertex of a structure (customdata [kind, id])."""
    return {"points": [{"x": 181.0, "y": 2.0, "customdata": [kind, ident]}]}


def drawn(path):
    """The shape plotly reports for a freehand path."""
    return {"type": "path", "path": path, "editable": True, "xref": "x", "yref": "y"}


def named(kind, ident, path="M181.0,2.0L181.5,2.5"):
    """The shape of a ridge or drawn path that the figure put in layout.shapes (plotly keeps its
    name); its geometry is not read back."""
    return {**drawn(path), "name": f"{kind} {ident}"}


@pytest.fixture(scope="module")
def picked(conc_explorer, conc_stores):
    """(explorer, sample, env, state, struct at start, map result) of the default sample."""
    state, sample, env = conc_stores
    struct = cb.struct_update(conc_explorer, None, sample, [], "sample")
    return conc_explorer, sample, env, state, struct, cb.concentration_result(conc_explorer, sample)[1]


def test_parse_svg_path_reads_plotly_shapes():
    assert cb.parse_svg_path("M181.5,2.25L181.4,2.5L181.3,2.75") == (
        [181.5, 181.4, 181.3], [2.25, 2.5, 2.75])
    assert cb.parse_svg_path("M1,2L3,4Z") == ([1.0, 3.0], [2.0, 4.0])      # Z is ignored
    assert cb.parse_svg_path("M 1 2 L 3 4 z") == ([1.0, 3.0], [2.0, 4.0])
    assert cb.parse_svg_path("M-1.5e1,.5 L 2 -3") == ([-15.0, 2.0], [0.5, -3.0])
    assert cb.parse_svg_path("M1,2 3,4") == ([1.0, 3.0], [2.0, 4.0])      # pairs after one command
    assert cb.parse_svg_path("") == ([], [])
    with pytest.raises(ValueError, match="unsupported"):
        cb.parse_svg_path("M1,2C3,4,5,6,7,8")
    with pytest.raises(ValueError, match="malformed"):
        cb.parse_svg_path("M1")
    with pytest.raises(ValueError, match="malformed"):
        cb.parse_svg_path("1,2")


def test_map_key_follows_the_map_and_not_the_region(conc_explorer, conc_stores):
    ex = conc_explorer
    _, sample, _ = conc_stores
    assert re.fullmatch(r"[0-9a-f]{12}", sample["map_key"]) and sample["map_key"] == cb.map_key(sample)
    inside = cb.sample_info(ex, cb.make_state(ex, concregion="inside"))
    assert inside["map_key"] == sample["map_key"]                       # the region is not the map
    assert cb.sample_info(ex, cb.make_state(ex, conck=12))["map_key"] != sample["map_key"]
    assert cb.sample_info(ex, cb.make_state(ex, concfloor=3.0))["map_key"] != sample["map_key"]
    assert cb.sample_info(ex, cb.make_state(ex, z_types=["spec"]))["map_key"] != sample["map_key"]


def test_struct_update_toggles_by_click_and_mode(picked):
    ex, sample, _, _, s0, result = picked
    assert s0 == {**default_struct(), "map_key": sample["map_key"]}
    assert [p["id"] for p in result["peaks"]] == [1, 2] and [r["id"] for r in result["ridges"]] == [1]
    assert cb.struct_update(ex, s0, sample, ["peaks"], "sample") == s0
    # no mode, no toggle
    assert cb.struct_update(ex, s0, sample, [], "click", click("peak", 1)) == s0
    assert cb.struct_update(ex, s0, sample, [], "click", click("ridge", 1)) == s0
    off = cb.struct_update(ex, s0, sample, ["peaks"], "click", click("peak", 2))
    assert off["off_peaks"] == [2] and s0["off_peaks"] == []                # the input is kept
    assert cb.struct_update(ex, off, sample, ["peaks"], "click", click("peak", 2))["off_peaks"] == []
    # a peak needs 'peaks', a ridge needs 'peaks' or 'paths'
    assert cb.struct_update(ex, s0, sample, ["paths"], "click", click("peak", 1)) == s0
    for pick in (["peaks"], ["paths"], ["peaks", "paths"]):
        assert cb.struct_update(ex, s0, sample, pick, "click", click("ridge", 1))["off_ridges"] == [1]
    # galaxies (customdata of numbers) and anything else are ignored
    galaxy = {"points": [{"customdata": [12.0, 0.02, 9.1, 0.3, 181.0, 2.0]}]}
    for other in (galaxy, {"points": [{}]}, {"points": []}, {}, None,
                  {"points": [{"customdata": ["star", 1]}]}):
        assert cb.struct_update(ex, s0, sample, ["peaks", "paths"], "click", other) == s0
    # the star annotation (clickAnnotationData) switches a peak, in 'peaks' mode only
    note = {"event": {}, "fullAnnotation": {"name": "peak 2", "text": "★"}, "timestamp": 1}
    assert cb.struct_update(ex, s0, sample, ["peaks"], "annotation", note)["off_peaks"] == [2]
    assert cb.struct_update(ex, s0, sample, ["paths"], "annotation", note) == s0
    other_note = {"fullAnnotation": {"name": "label"}}
    assert cb.struct_update(ex, s0, sample, ["peaks"], "annotation", other_note) == s0
    assert cb.struct_update(ex, s0, None, ["peaks"], "click", click("peak", 1)) == s0   # no sample


def test_struct_update_stores_a_drawn_path_once(picked):
    ex, sample, _, _, s0, _ = picked
    path = "M181.5,2.0L181.2,2.2L180.9,2.4"
    event = {"shapes": [drawn(path)]}
    s1 = cb.struct_update(ex, s0, sample, ["paths"], "relayout", relayout=event)
    assert s1["paths"] == [{"id": 1, "ra": [181.5, 181.2, 180.9], "dec": [2.0, 2.2, 2.4],
                            "on": True, "src": path}] and s1["next_id"] == 2
    again = cb.struct_update(ex, s1, sample, ["paths"], "relayout", relayout=event)   # same shapes
    assert again["paths"] == s1["paths"] and again["next_id"] == 2 and again["rev"] == s1["rev"] + 1
    two = {"shapes": [drawn(path), drawn("M181.0,1.0L180.5,1.5")]}
    s2 = cb.struct_update(ex, s1, sample, ["paths"], "relayout", relayout=two)
    assert [q["id"] for q in s2["paths"]] == [1, 2] and s2["next_id"] == 3
    assert s2["paths"][1]["ra"] == [181.0, 180.5]
    # only in 'paths' mode; an edited shape or an event without shapes changes nothing
    assert cb.struct_update(ex, s0, sample, ["peaks"], "relayout", relayout=event) == s0
    edit = cb.struct_update(ex, s1, sample, ["paths"], "relayout",
                            relayout={"shapes[0].path": "M1,1L2,2"})
    assert {**edit, "rev": 0} == {**s1, "rev": 0} and edit["rev"] == s1["rev"] + 1   # put back
    assert cb.struct_update(ex, s1, sample, ["paths"], "relayout",
                            relayout={"xaxis.range[0]": 1.0}) == s1
    # one point, a repeated point, a command that is not supported, text that is no path
    for bad in ("M181.5,2.0", "M181.5,2.0L181.5,2.0", "M1,2C3,4,5,6,7,8", "garbage"):
        got = cb.struct_update(ex, s0, sample, ["paths"], "relayout", relayout={"shapes": [
            drawn(bad), {"type": "rect"}]})
        assert got["paths"] == [] and got["next_id"] == 1
    # clicks: a path is switched with 'paths' only
    off = cb.struct_update(ex, s1, sample, ["paths"], "click", click("path", 1))
    assert off["paths"][0]["on"] is False and s1["paths"][0]["on"] is True
    assert cb.struct_update(ex, off, sample, ["paths"], "click", click("path", 1))["paths"][0]["on"]
    assert cb.struct_update(ex, s1, sample, [], "click", click("path", 1)) == s1
    assert cb.struct_update(ex, s1, sample, ["paths"], "click", click("path", 9)) == s1
    # 'peaks' is the click mode of every structure: it switches a drawn path too
    for pick in (["peaks"], ["peaks", "paths"]):
        flipped = cb.struct_update(ex, s1, sample, pick, "click", click("path", 1))
        assert flipped["paths"][0]["on"] is False


def test_struct_update_all_none_clear_and_a_new_map(picked):
    ex, sample, _, _, s0, result = picked
    s = cb.struct_update(ex, s0, sample, ["paths"], "relayout", relayout={"shapes": [
        named("ridge", 1), drawn("M181.5,2.0L181.2,2.2"), drawn("M181.0,1.0L180.5,1.5")]})
    none = cb.struct_update(ex, s, sample, [], "none")
    assert none["off_peaks"] == [1, 2] and none["off_ridges"] == [1]
    assert [q["on"] for q in none["paths"]] == [False, False]
    every = cb.struct_update(ex, none, sample, [], "all")
    assert every["off_peaks"] == [] and every["off_ridges"] == [] and all(q["on"] for q in every["paths"])
    cleared = cb.struct_update(ex, none, sample, [], "clear_paths")
    assert cleared["paths"] == [] and cleared["off_peaks"] == [1, 2] and cleared["next_id"] == 3
    # another map: the ids belong to the old map, the drawn paths stay
    other = cb.sample_info(ex, cb.make_state(ex, conck=12))
    moved = cb.struct_update(ex, none, other, [], "sample")
    assert moved["map_key"] == other["map_key"] != none["map_key"]
    assert moved["off_peaks"] == [] and moved["off_ridges"] == [] and moved["paths"] == none["paths"]
    # a sample too small for a map: nothing to switch off
    tiny = cb.sample_info(ex, cb.make_state(ex, z_range=[0.0140, 0.0141]))
    assert cb.struct_update(ex, s, tiny, [], "none")["off_peaks"] == []


def test_pick_counts_and_run_spec_take_the_structures_that_are_on(picked):
    ex, sample, _, _, s0, result = picked
    assert cb.pick_counts(ex, sample, s0) == "peaks on 2/2, ridges on 1/1, drawn 0, added peaks 0"
    spec = cb.run_spec(ex, sample, s0)                      # all on: the default spec of the map
    ref = default_spec(result)
    np.testing.assert_allclose(spec["peaks"], ref["peaks"])
    assert spec["paths"] == ref["paths"] and spec["paths"][0]["label"] == "ridge 1"
    assert spec["map_key"] == sample["map_key"] and re.fullmatch(r"\d\d:\d\d:\d\d", spec["time"])
    peak = peak_table(result)[0]
    assert spec["peaks"][0] == [peak["ra"], peak["dec"]]
    assert spec["paths"][0]["ra"] == ridge_table(result)[0]["ra"]
    # some off, one drawn path on and one off
    s = cb.struct_update(ex, s0, sample, ["paths"], "relayout", relayout={"shapes": [
        named("ridge", 1), drawn("M181.5,2.0L181.2,2.2"), drawn("M181.0,1.0L180.5,1.5")]})
    s = cb.struct_update(ex, s, sample, ["peaks", "paths"], "click", click("path", 2))
    s = cb.struct_update(ex, s, sample, ["peaks"], "click", click("peak", 1))
    s = cb.struct_update(ex, s, sample, ["peaks"], "click", click("ridge", 1))
    assert cb.pick_counts(ex, sample, s) == "peaks on 1/2, ridges on 0/1, drawn 1/2, added peaks 0"
    spec = cb.run_spec(ex, sample, s)
    assert spec["peaks"] == [[peak_table(result)[1]["ra"], peak_table(result)[1]["dec"]]]
    assert spec["paths"] == [{"label": "drawn 1", "ra": [181.5, 181.2], "dec": [2.0, 2.2]}]
    # nothing selected
    nothing = cb.struct_update(ex, s0, sample, [], "none")
    with pytest.raises(ConcentrationError, match="select at least one"):
        cb.run_spec(ex, sample, nothing)
    # the ids of another map are not applied to this one
    stale = {**s, "map_key": "0123456789ab"}
    assert len(cb.run_spec(ex, sample, stale)["peaks"]) == 2
    assert cb.pick_counts(ex, sample, stale).startswith("peaks on 2/2, ridges on 1/1")


def test_run_model_buttons_and_status(picked):
    ex, sample, _, _, s0, _ = picked
    spec, status = cb.run_model(ex, sample, s0)
    assert status == f"model: 2 peaks, 1 paths (run {spec['time']})"
    assert cb.model_status(None) == "model: not run"
    assert cb.run_model(ex, sample, s0, "clear") == (None, "model: not run")
    none = cb.struct_update(ex, s0, sample, [], "none")
    model, status = cb.run_model(ex, sample, none)
    assert model is no_update and status == "model: cannot run (select at least one peak or path)"
    assert cb.run_model(ex, None, s0)[0] is no_update
    # a path drawn outside the survey is skipped, and the status says so
    away = {**none, "paths": [{"id": 1, "ra": [10.0, 10.5], "dec": [-40.0, -40.5], "on": True,
                               "src": "x"}]}
    spec, status = cb.run_model(ex, sample, away)
    assert spec["paths"][0]["label"] == "drawn 1"
    assert "skipped: drawn 1 (fewer than 2 distinct points inside the survey area)" in status


def test_model_drives_the_sample_filter_the_table_and_the_overlay(picked):
    ex, sample, env, state, s0, _ = picked
    spec, _ = cb.run_model(ex, sample, s0)
    plain = cb.sample_info(ex, cb.make_state(ex, model=spec))
    inside = cb.sample_info(ex, cb.make_state(ex, model=spec, concregion="inside"))
    # no region: the model is carried but filters nothing; the map does not change
    assert "conc_model" not in plain["filters"] and plain["model"] == spec
    assert plain["map_key"] == inside["map_key"] == sample["map_key"]
    assert sample["model"] is None
    # inside: the 1 sigma band of the model (without time and map_key)
    assert inside["filters"]["conc_model"] == cb.model_spec(spec) == {
        "peaks": spec["peaks"], "paths": spec["paths"]} and inside["model"] == spec
    base = ex.select(plain["filters"])
    columns, model = cb.model_result(ex, plain)
    np.testing.assert_array_equal(ex.select(inside["filters"]), base & columns["model_inside"])
    outside = cb.sample_info(ex, cb.make_state(ex, model=spec, concregion="outside"))
    np.testing.assert_array_equal(ex.select(outside["filters"]), base & ~columns["model_inside"])
    assert inside["n_spec"] + inside["n_phot"] == int((base & columns["model_inside"]).sum())
    # the map and its figure do not move with the region
    np.testing.assert_array_equal(cb._map_mask(ex, inside), base)
    # the statistics table, the summary and the caption follow the model
    lines, table, caption = cb.concentration_callback(ex, env, plain)
    rows = [[c.children for c in tr.children] for tr in table.children[1].children]
    contour = cb.concentration_result(ex, plain)[0]["conc_inside"]
    assert int(rows[0][3]) == int((base & columns["model_inside"]).sum())
    assert int(rows[0][6]) == int((base & ~columns["model_inside"]).sum())
    assert not np.array_equal(base & columns["model_inside"], base & contour)
    assert "Inside = within the 1σ band of the model (2 peaks, 1 paths)" in caption
    assert lines[0].startswith("Map of ") and lines[4].startswith("Model: 2 peaks and 1 paths")
    assert any(line.startswith("Inside the 1 sigma band of the model: ") for line in lines)
    assert any(line.startswith("ridge 1: ") and "intrinsic" not in line for line in lines)
    assert not any("intrinsic" in line for line in lines)
    # the structures table: one row per peak and path, N inside the band of each
    box = cb.structures_table(ex, plain)
    assert box.children[1].className == "conc-table"
    srows = [[c.children for c in tr.children] for tr in box.children[1].children[1].children]
    assert [c.children for c in box.children[1].children[0].children.children] == list(STRUCTURE_HEADER)
    assert [r[:2] for r in srows] == [["peak 1", "peak"], ["peak 2", "peak"], ["ridge 1", "path"]]
    assert sum(int(r[6]) for r in srows) == int((base & columns["model_inside"]).sum())
    assert float(srows[0][2]) > 0 and " / " in srows[2][3] and srows[0][4] == "-"
    assert "intrinsic s [Mpc]" not in STRUCTURE_HEADER and len(STRUCTURE_HEADER) == 7
    # a sky selection restricts the counts of the structures, as it does the table
    some = list(np.flatnonzero(base & columns["model_inside"])[:5])
    few = cb.structures_table(ex, plain, some).children[1].children[1].children
    assert sum(int(tr.children[6].children) for tr in few) == 5
    assert cb.structures_table(ex, sample).children is None                  # no model: empty Div
    # the overlay: the fitted paths, the 1 sigma band and the circles of the peaks
    sky = cb.sky_callback(ex, state, env, plain, struct=s0, pick=[])
    assert [t.name for t in sky.data] == SKY_TRACES[:11] + MODEL_TRACES + SKY_TRACES[11:]
    band, circles, fitted = sky.data[11], sky.data[12], sky.data[13]
    assert fitted.line.color == band.line.color == circles.line.color == figures.MODEL
    assert (fitted.line.width, band.line.width, band.line.dash) == (3.0, 1.0, "dash")
    assert circles.line.width == 1.5 and circles.showlegend is False
    assert circles.legendgroup == band.legendgroup == "model: 1σ band"
    assert list(band.x).count(None) == 2 and list(circles.x).count(None) == 2
    assert list(fitted.x).count(None) == 1 and len(fitted.x) > 5
    assert not any(t.name in MODEL_TRACES for t in cb.sky_callback(ex, state, env, sample).data)


def test_a_model_that_fails_is_dropped_with_a_note(picked):
    ex, sample, env, state, s0, _ = picked
    bad = {"peaks": [[1.0]], "paths": [], "time": "00:00:00", "map_key": "x"}   # malformed
    info = cb.sample_info(ex, cb.make_state(ex, model=bad, concregion="inside"))
    assert "model region filter off" in info["text"] and "malformed model spec" in info["text"]
    assert "conc_model" not in info["filters"] and info["filters"]["conc_region"] == "inside"
    assert info["model"] is None
    np.testing.assert_array_equal(ex.select(info["filters"]),
                                  ex.select(cb.sample_info(ex, cb.make_state(
                                      ex, concregion="inside"))["filters"]))
    # without a region the failure shows where the model is used, never as an exception
    held = cb.sample_info(ex, cb.make_state(ex, model=bad))
    assert held["model"] == bad
    lines, table, caption = cb.concentration_callback(ex, env, held)
    assert any(line.startswith("Model failed: malformed model spec") for line in lines)
    assert "1 sigma contour" in caption and table.className == "conc-table"
    assert cb.structures_table(ex, held).children is None
    assert [t.name for t in cb.sky_callback(ex, state, env, held).data] == SKY_TRACES
    empty = cb.sample_info(ex, cb.make_state(ex, model={"peaks": [], "paths": []},
                                             concregion="inside"))
    assert "conc_model" not in empty["filters"] and empty["model"] == {"peaks": [], "paths": []}
    assert any(line.startswith("Model failed: select at least one")
               for line in cb.concentration_callback(ex, env, empty)[0])


def test_sky_figure_pick_modes_and_the_state_of_the_structures(picked):
    ex, sample, env, state, s0, _ = picked
    base = cb.sky_callback(ex, state, env, sample)
    assert base.layout.dragmode == "select" and base.layout.shapes == ()
    assert base.layout.newshape.line.color == figures.DRAWN and base.layout.newshape.line.width == 2
    assert base.layout.editrevision == "sky0-0-view" and base.layout.uirevision == "sky0"
    galaxies = base.data[GALAXIES]
    assert galaxies.hoverinfo is None and "ridge" not in galaxies.hovertemplate
    assert base.data[0].hovertemplate.startswith("RA ") and base.data[0].hoverinfo is None
    for pick, mode in ((["peaks"], "select"), (["paths"], "drawopenpath"),
                       (["peaks", "paths"], "drawopenpath")):
        fig = cb.sky_callback(ex, state, env, sample, struct=s0, pick=pick)
        assert fig.layout.dragmode == mode
        assert fig.data[GALAXIES].hoverinfo == "skip" and fig.data[GALAXIES].hovertemplate is None
        assert fig.data[0].hoverongaps is False                  # no click outside the survey
        if "peaks" in pick:                                      # a click on the map adds a peak
            assert fig.data[0].hovertemplate == figures.CLICK_HOVER and fig.data[0].hoverinfo is None
        else:
            assert fig.data[0].hoverinfo == "skip" and fig.data[0].hovertemplate is None
        assert fig.data[PEAKS].hovertemplate == "%{text}<extra></extra>"
        if "paths" in pick:      # ridges that are on are editable shapes, their traces are empty
            assert list(fig.data[6].x) == [] and list(fig.data[7].x) == []
            assert [s.name for s in fig.layout.shapes] == ["ridge 1"]
        else:
            assert fig.data[6].hoverinfo is None and len(fig.data[6].x) + len(fig.data[7].x) > 5
            assert fig.layout.shapes == ()
    # everything on: peaks white, the ridge cyan with customdata [kind, id] on every vertex
    notes = base.layout.annotations
    assert [n.name for n in notes] == ["peak 1", "peak 2"] and all(n.captureevents for n in notes)
    assert all(n.font.color == figures.PEAK and n.font.size == 22 for n in notes)
    ridges = next(t for t in base.data if t.name in ("ridges", "ridges to the border") and len(t.x))
    cd = [list(c) for c in ridges.customdata]
    assert cd[:-1] == [["ridge", 1]] * (len(cd) - 1) and cd[-1] == [None, None]
    assert ridges.mode == "lines+markers" and ridges.hovertemplate == "%{text}<extra></extra>"
    assert ridges.text[0].startswith("ridge 1, saddle ") and ridges.text[0].endswith("(on)")
    assert [list(c) for c in base.data[PEAKS].customdata] == [["peak", 1], ["peak", 2]]
    assert list(base.data[PEAKS].marker.color) == [figures.PEAK, figures.PEAK]
    # switched off: grey and smaller stars, the ridge moves to 'ridges off', drawn paths show
    s = cb.struct_update(ex, s0, sample, ["paths"], "relayout", relayout={"shapes": [
        named("ridge", 1), drawn("M181.5,2.0L181.2,2.2"), drawn("M181.0,1.0L180.5,1.5")]})
    s = cb.struct_update(ex, s, sample, ["peaks"], "click", click("peak", 1))
    s = cb.struct_update(ex, s, sample, ["peaks"], "click", click("ridge", 1))
    s = cb.struct_update(ex, s, sample, ["paths"], "click", click("path", 2))
    fig = cb.sky_callback(ex, state, env, sample, clear_clicks=2, struct=s, pick=["peaks"])
    assert fig.layout.editrevision == f"sky2-{s['rev']}-view" and fig.layout.uirevision == "sky2"
    off_note, on_note = fig.layout.annotations
    assert (off_note.font.color, off_note.font.size, off_note.name) == (figures.OFF, 16, "peak 1")
    assert (on_note.font.color, on_note.font.size, on_note.name) == (figures.PEAK, 22, "peak 2")
    assert off_note.hovertext.endswith("(off)") and on_note.hovertext.endswith("(on)")
    assert list(fig.data[PEAKS].marker.color) == [figures.OFF, figures.PEAK]
    names = {t.name: t for t in fig.data}
    assert len(names["ridges"].x) == 0 and names["ridges"].showlegend is False
    off_ridge = names["ridges off"]
    assert off_ridge.line.color == figures.OFF and off_ridge.line.dash == "dot"
    assert off_ridge.opacity == 0.6 and off_ridge.showlegend is True
    assert [list(c) for c in off_ridge.customdata][0] == ["ridge", 1]
    assert off_ridge.text[0].endswith("(off)")
    path_on, path_off = names["drawn paths"], names["drawn paths off"]
    assert path_on.line.color == figures.DRAWN and path_on.line.dash == "dash"
    assert [list(c) for c in path_on.customdata] == [["path", 1], ["path", 1], [None, None]]
    assert list(path_on.x)[:2] == [181.5, 181.2] and path_on.text[0] == "drawn path 1 (on)"
    assert path_off.line.color == figures.OFF and path_off.line.dash == "dot"
    assert [list(c) for c in path_off.customdata][0] == ["path", 2]
    # in the drawing mode only what is on is an editable shape: the drawn path 1, not the ridge
    # that is off nor the drawn path 2; the traces of what is on are empty, the others stay
    draw = cb.sky_callback(ex, state, env, sample, clear_clicks=2, struct=s, pick=["paths"])
    assert draw.layout.editrevision == f"sky2-{s['rev']}-draw"
    assert [sh.name for sh in draw.layout.shapes] == ["path 1"]
    shape = draw.layout.shapes[0]
    assert shape.type == "path" and shape.editable is True and shape.line.color == figures.DRAWN
    assert cb.parse_svg_path(shape.path) == ([181.5, 181.2], [2.0, 2.2])
    drawn_names = {t.name: t for t in draw.data}
    assert len(drawn_names["drawn paths"].x) == 0 and len(drawn_names["drawn paths off"].x) > 0
    assert len(drawn_names["ridges off"].x) > 0
    # the ridge shapes are the ridges that are on: cyan, dashed when they run to the border
    on = cb.sky_callback(ex, state, env, sample, struct=s0, pick=["paths"]).layout.shapes
    assert [sh.name for sh in on] == ["ridge 1"] and on[0].line.color == figures.RIDGE
    ridge = ridge_table(cb.concentration_result(ex, sample)[1])[0]
    assert cb.parse_svg_path(on[0].path)[0] == pytest.approx(ridge["ra"], abs=1e-4)
    assert cb.parse_svg_path(on[0].path)[1] == pytest.approx(ridge["dec"], abs=1e-4)
    # the overlay groups: no low-crest group, the shapes of what is on
    overlay = cb.concentration_overlay(ex, sample, s0)
    assert set(overlay["ridge_groups"]) == {"strong", "open", "off"}
    assert not any(t.name == "low-crest ridges" for t in base.data)
    assert "low-crest ridges" not in [n for n, *_ in figures.RIDGE_STYLES]
    assert [sh["name"] for sh in overlay["shapes"]] == ["ridge 1"]
    assert len(overlay["ridge_groups"]["strong"]["x"]) + len(overlay["ridge_groups"]["open"]["x"]) > 0
    assert overlay["peak_id"] == [1, 2] and overlay["peak_on"] == [True, True] and overlay["model"] is None
    assert overlay["user_peak_id"] == []


def heat_click(ra=181.2, dec=2.1, z=1.5):
    """A Graph's clickData on a cell of the map heatmap: no customdata, x y z of the cell."""
    return {"points": [{"curveNumber": 0, "pointNumber": [3, 4], "pointIndex": [3, 4],
                        "x": ra, "y": dec, "z": z}]}


def test_a_click_on_the_map_adds_a_peak_and_a_click_on_its_star_removes_it(picked):
    ex, sample, _, _, s0, result = picked
    # only in the 'peaks' mode: a click on the map does nothing while drawing or with no mode
    for pick in ([], ["paths"]):
        assert cb.struct_update(ex, s0, sample, pick, "click", heat_click()) == s0
    s1 = cb.struct_update(ex, s0, sample, ["peaks"], "click", heat_click(181.2, 2.1))
    assert s1["user_peaks"] == [{"id": 1, "ra": 181.2, "dec": 2.1}] and s1["next_id"] == 2
    assert s0["user_peaks"] == [] and s0["next_id"] == 1                  # the input is kept
    assert s1["off_peaks"] == [] and s1["off_ridges"] == [] and s1["paths"] == []
    s2 = cb.struct_update(ex, s1, sample, ["peaks", "paths"], "click", heat_click(180.9, 1.8))
    assert [(q["id"], q["ra"], q["dec"]) for q in s2["user_peaks"]] == [
        (1, 181.2, 2.1), (2, 180.9, 1.8)] and s2["next_id"] == 3
    # the ids come from next_id, shared with the drawn paths
    s3 = cb.struct_update(ex, s2, sample, ["paths"], "relayout", relayout={"shapes": [
        named("ridge", 1), drawn("M181.5,2.0L181.2,2.2")]})
    assert s3["paths"][0]["id"] == 3 and s3["next_id"] == 4
    # a cell with no value (outside the survey), no position, or a galaxy is not a map click
    for bad in ({"points": [{"x": 181.2, "y": 2.1, "z": None}]},
                {"points": [{"x": 181.2, "y": 2.1, "z": float("nan")}]},
                {"points": [{"x": 181.2, "z": 1.0}]}, {"points": [{"x": "a", "y": 2.1, "z": 1.0}]},
                {"points": [{"customdata": [12.0, 0.02, 9.1, 0.3, 181.0, 2.0], "x": 181.2,
                             "y": 2.1}]}, {"points": []}, None):
        assert cb.struct_update(ex, s0, sample, ["peaks"], "click", bad) == s0
    # a ridge, path or peak vertex under the cursor wins over the map cell of the same click
    both = {"points": [{**heat_click()["points"][0], "customdata": ["ridge", 1]}]}
    on_ridge = cb.struct_update(ex, s0, sample, ["peaks"], "click", both)
    assert on_ridge["off_ridges"] == [1] and on_ridge["user_peaks"] == []
    # the star of a peak added by hand (annotation 'user peak <id>') removes it, in 'peaks' mode
    note = {"event": {}, "fullAnnotation": {"name": "user peak 1", "text": "★"}, "timestamp": 1}
    s4 = cb.struct_update(ex, s2, sample, ["peaks"], "annotation", note)
    assert [q["id"] for q in s4["user_peaks"]] == [2] and s4["off_peaks"] == []
    assert cb.struct_update(ex, s2, sample, ["paths"], "annotation", note) == s2
    assert cb.struct_update(ex, s2, sample, ["peaks"], "annotation",
                            {"fullAnnotation": {"name": "user peak 9"}}) == s2
    # the star of an automatic peak 1 is another thing, though the number is the same
    auto = cb.struct_update(ex, s2, sample, ["peaks"], "annotation",
                            {"fullAnnotation": {"name": "peak 1"}})
    assert auto["off_peaks"] == [1] and auto["user_peaks"] == s2["user_peaks"]
    # a click event on a user peak vertex (customdata [kind, id]) removes it as well
    gone = cb.struct_update(ex, s2, sample, ["peaks"], "click", click("user_peak", 2))
    assert [q["id"] for q in gone["user_peaks"]] == [1]
    assert cb.struct_update(ex, s2, sample, ["paths"], "click", click("user_peak", 2)) == s2
    # All on / All off leave them, Clear drawn removes them with the drawn paths
    assert cb.struct_update(ex, s3, sample, [], "none")["user_peaks"] == s3["user_peaks"]
    assert cb.struct_update(ex, s3, sample, [], "all")["user_peaks"] == s3["user_peaks"]
    cleared = cb.struct_update(ex, s3, sample, [], "clear_paths")
    assert cleared["paths"] == [] and cleared["user_peaks"] == [] and cleared["next_id"] == 4
    # another map: the peaks added by hand go with the old one
    other = cb.sample_info(ex, cb.make_state(ex, conck=12))
    moved = cb.struct_update(ex, s3, other, [], "sample")
    assert moved["user_peaks"] == [] and moved["paths"] == s3["paths"]
    # a sample too small for a map keeps the struct and adds nothing
    assert cb.struct_update(ex, s0, None, ["peaks"], "click", heat_click()) == s0
    # an old store-struct without the key is read as empty
    legacy = {k: v for k, v in s0.items() if k != "user_peaks"}
    assert cb.struct_update(ex, legacy, sample, ["peaks"], "click", heat_click())["user_peaks"] == [
        {"id": 1, "ra": 181.2, "dec": 2.1}]


def test_the_peaks_added_by_hand_are_drawn_counted_and_run_with_the_model(picked):
    ex, sample, env, state, s0, result = picked
    peak = peak_table(result)[1]
    ra, dec = peak["ra"] + 0.05, peak["dec"] - 0.04                    # a little off peak 2
    s = cb.struct_update(ex, s0, sample, ["peaks"], "click", heat_click(ra, dec))
    s = cb.struct_update(ex, s, sample, ["peaks"], "click", heat_click(181.9, 2.9))
    assert cb.pick_counts(ex, sample, s) == "peaks on 2/2, ridges on 1/1, drawn 0, added peaks 2"
    # the figure: magenta stars as annotations (clicks reach Dash) and a legend entry
    fig = cb.sky_callback(ex, state, env, sample, struct=s, pick=["peaks"])
    notes = fig.layout.annotations
    assert [n.name for n in notes] == ["peak 1", "peak 2", "user peak 1", "user peak 2"]
    mine = notes[2]
    assert (mine.x, mine.y, mine.text) == (ra, dec, "★") and mine.captureevents is True
    assert mine.font.color == figures.DRAWN != notes[0].font.color and mine.font.size >= 20
    trace = next(t for t in fig.data if t.name == "peaks added by hand")
    assert list(trace.x) == [ra, 181.9] and [list(c) for c in trace.customdata] == [
        ["user_peak", 1], ["user_peak", 2]]
    assert not any(t.name == "peaks added by hand" for t in cb.sky_callback(ex, state, env, sample,
                                                                            struct=s0).data)
    assert [t.name for t in fig.data] == SKY_TRACES[:12] + ["peaks added by hand"] + SKY_TRACES[12:]
    # the model takes them like the automatic peaks that are on, after them
    spec = cb.run_spec(ex, sample, s)
    assert len(spec["peaks"]) == 4 and spec["peaks"][2:] == [[ra, dec], [181.9, 2.9]]
    off = cb.struct_update(ex, s, sample, ["peaks"], "click", click("peak", 1))
    assert len(cb.run_spec(ex, sample, off)["peaks"]) == 3
    stale = {**s, "map_key": "0123456789ab"}                     # the peaks belong to their map
    assert len(cb.run_spec(ex, sample, stale)["peaks"]) == 2
    spec, status = cb.run_model(ex, sample, s)
    assert status == f"model: 4 peaks, 1 paths (run {spec['time']})"
    columns, model = cb.model_result(ex, {**sample, "model": spec})
    h = cb.concentration_result(ex, sample)[1]["info"]["knn_scale_mpc"]
    assert [p["label"] for p in model["peaks"]] == ["peak 1", "peak 2", "peak 3", "peak 4"]
    # refined to the highest cell within PEAK_SEARCH h of the click, which is near peak 2's top
    assert model["peaks"][2]["shift_mpc"] <= 1.0 * h + result["grid"]["pixel"]
    assert np.hypot(model["peaks"][2]["x"] - model["peaks"][1]["x"],
                    model["peaks"][2]["y"] - model["peaks"][1]["y"]) < 2.0 * h
    assert model["info"]["n_peaks"] == 4 and model["skipped"] == []


def test_erasing_a_named_shape_switches_the_structure_off_and_a_new_shape_adds_a_path(picked):
    ex, sample, env, state, s0, _ = picked
    kept = cb.struct_update(ex, s0, sample, ["paths"], "relayout",
                            relayout={"shapes": [named("ridge", 1)]})
    assert {**kept, "rev": 0} == {**s0, "rev": 0} and kept["off_ridges"] == []
    # the modebar eraser: the remaining shapes, the ridge is not among them
    erased = cb.struct_update(ex, s0, sample, ["paths"], "relayout",
                              relayout={"shapes": [], "_deactivateShape": True})
    assert erased["off_ridges"] == [1] and erased["paths"] == [] and erased["rev"] == 1
    assert s0["off_ridges"] == []
    assert cb.struct_update(ex, erased, sample, ["paths"], "relayout",
                            relayout={"shapes": []})["off_ridges"] == [1]            # once
    # a click in the 'peaks' mode switches it on again; peaks and drawn paths are untouched
    back = cb.struct_update(ex, erased, sample, ["peaks"], "click", click("ridge", 1))
    assert back["off_ridges"] == [] and erased["off_peaks"] == [] == back["off_peaks"]
    # only in the drawing mode (that is where the shapes are)
    assert cb.struct_update(ex, s0, sample, ["peaks"], "relayout", relayout={"shapes": []}) == s0
    assert cb.struct_update(ex, s0, sample, [], "relayout", relayout={"shapes": []}) == s0
    # a drawn path, then the ridge and the path both shown as named shapes
    path = "M181.5,2.0L181.2,2.2L180.9,2.4"
    one = cb.struct_update(ex, s0, sample, ["paths"], "relayout",
                           relayout={"shapes": [named("ridge", 1), drawn(path)]})
    assert [q["id"] for q in one["paths"]] == [1] and one["off_ridges"] == []
    both = {"shapes": [named("ridge", 1), named("path", 1)]}
    same = cb.struct_update(ex, one, sample, ["paths"], "relayout", relayout=both)
    assert same["paths"] == one["paths"] and same["off_ridges"] == [] and same["next_id"] == 2
    # a new unnamed shape among the named ones is a new drawn path
    new = cb.struct_update(ex, one, sample, ["paths"], "relayout", relayout={"shapes": [
        named("ridge", 1), named("path", 1), drawn("M181.0,1.0L180.5,1.5")]})
    assert [q["id"] for q in new["paths"]] == [1, 2] and new["paths"][1]["ra"] == [181.0, 180.5]
    assert new["paths"][0]["on"] and new["off_ridges"] == [] and new["next_id"] == 3
    # a named path shape that is gone: the path is switched off, not deleted; the ridge stays on
    gone = cb.struct_update(ex, new, sample, ["paths"], "relayout",
                            relayout={"shapes": [named("ridge", 1), named("path", 2)]})
    assert [(q["id"], q["on"]) for q in gone["paths"]] == [(1, False), (2, True)]
    assert gone["off_ridges"] == []
    # ... and a click on it in the 'peaks' mode switches it on again
    assert cb.struct_update(ex, gone, sample, ["peaks"], "click", click("path", 1))["paths"][0]["on"]
    # a path that is already off has no shape and is not erased again
    again = cb.struct_update(ex, gone, sample, ["paths"], "relayout",
                             relayout={"shapes": [named("ridge", 1), named("path", 2)]})
    assert [(q["id"], q["on"]) for q in again["paths"]] == [(1, False), (2, True)]
    # a browser that still holds the unnamed shape of a stored path keeps that path on
    stale = cb.struct_update(ex, one, sample, ["paths"], "relayout",
                             relayout={"shapes": [named("ridge", 1), drawn(path)]})
    assert stale["paths"][0]["on"] is True and len(stale["paths"]) == 1
    # shapes of other ids and names that are not ours are ignored
    odd = cb.struct_update(ex, s0, sample, ["paths"], "relayout", relayout={"shapes": [
        named("ridge", 1), named("ridge", 99), {**drawn("M1,1L2,2"), "name": "label"},
        {"type": "rect", "name": "path x"}, "text", None]})
    assert odd["off_ridges"] == [] and odd["paths"] == [] and odd["next_id"] == 1
    # edits of a named shape (plotly: 'shapes[i].path', 'shapes[i].x0', ...) never touch the
    # structures; only 'rev' moves so that the figure puts the shape back
    for key in ("shapes[0].path", "shapes[1].x0", "shapes[12].y1"):
        edit = cb.struct_update(ex, one, sample, ["paths"], "relayout", relayout={key: "M9,9L8,8"})
        assert {**edit, "rev": 0} == {**one, "rev": 0} and edit["rev"] == one["rev"] + 1
    for harmless in ({"xaxis.range[0]": 1.0}, {"shapesx": 1}, {}, None, "text", {"shapes": "no"}):
        assert cb.struct_update(ex, one, sample, ["paths"], "relayout", relayout=harmless) == one
    # the figure then shows what is left: the erased ridge is a trace 'ridges off', not a shape
    fig = cb.sky_callback(ex, state, env, sample, struct=erased, pick=["paths"])
    assert fig.layout.shapes == () and fig.layout.editrevision == "sky0-1-draw"
    assert len(next(t for t in fig.data if t.name == "ridges off").x) > 0


def test_the_colour_scale_is_log_with_a_floor_and_clicks_do_not_depend_on_z(picked):
    ex, sample, env, state, s0, _ = picked
    z = np.array([[np.nan, 0.0, 0.05, 0.1], [0.5, 1.0, 12.0, 400.0]])
    logz, zmax, ticks, labels = figures.map_scale(z)
    assert np.isnan(logz[0, 0]) and logz[0, 1] == logz[0, 2] == logz[0, 3] == -1.0
    assert np.isfinite(logz[~np.isnan(z)]).all()                       # empty cells stay finite
    assert logz[1, 1] == 0.0 and logz[1, 2] == pytest.approx(np.log10(12.0))
    assert zmax == pytest.approx(np.log10(np.percentile([0.0, 0.05, 0.1, 0.5, 1.0, 12.0, 400.0],
                                                        99.5)))
    assert labels == ["0.1", "0.3", "1", "3", "10", "30", "100"] and ticks[2] == 0.0
    # a map that never reaches the mean: the top is the mean, the ticks stop there
    low = figures.map_scale(np.array([[0.2, 0.3], [np.nan, 0.5]]))
    assert low[1] == 0.0 and low[3] == ["0.1", "0.3", "1"]
    assert figures.map_scale(np.full((2, 2), np.nan))[1] == 0.0
    # a click on a cell of the floor (z = -1, an empty cell inside the survey) adds a peak
    for value in (-1.0, 0.0, 0.7, 1.9):
        got = cb.struct_update(ex, s0, sample, ["peaks"], "click", heat_click(181.2, 2.1, value))
        assert got["user_peaks"] == [{"id": 1, "ra": 181.2, "dec": 2.1}]
    # the figure in the peaks mode: the same map, no linear customdata (the hover is the click text)
    fig = cb.sky_callback(ex, state, env, sample, struct=s0, pick=["peaks"])
    assert fig.data[0].customdata is None and fig.data[0].zmin == -1.0
    off = cb.sky_callback(ex, cb.make_state(ex, density=[]), env, sample, struct=s0, pick=["peaks"])
    np.testing.assert_array_equal(np.array(off.data[0].z, dtype=float),
                                  np.array(fig.data[0].z, dtype=float))


def test_the_map_answers_clicks_in_the_peaks_mode_with_or_without_the_density(picked):
    ex, sample, env, state, s0, _ = picked
    on = cb.sky_callback(ex, state, env, sample, struct=s0, pick=["peaks"])
    heat = on.data[0]
    assert heat.name == "concentration" and heat.hovertemplate == figures.CLICK_HOVER
    assert heat.hoverongaps is False and heat.hoverinfo is None
    assert "click" in heat.hovertemplate and "add peak" in heat.hovertemplate
    assert [t.name for t in on.data] == SKY_TRACES
    # the density background off: a fully transparent heatmap on the same grid takes its place
    off_state = cb.make_state(ex, density=[])
    off = cb.sky_callback(ex, off_state, env, sample, struct=s0, pick=["peaks"])
    catcher = off.data[0]
    assert type(catcher).__name__ == "Heatmap" and catcher.name == "click map"
    assert catcher.showlegend is False and catcher.showscale is False and catcher.hoverongaps is False
    assert catcher.hovertemplate == figures.CLICK_HOVER
    assert all(c[1] == "rgba(0,0,0,0)" for c in catcher.colorscale)
    np.testing.assert_array_equal(np.array(catcher.z, dtype=float), np.array(heat.z, dtype=float))
    assert list(catcher.x) == list(heat.x) and list(catcher.y) == list(heat.y)
    assert [t.name for t in off.data] == ["click map"] + SKY_TRACES[1:]
    # no catcher without the mode, nor in the drawing mode alone; no heatmap then
    for pick in ([], ["paths"]):
        fig = cb.sky_callback(ex, off_state, env, sample, struct=s0, pick=pick)
        assert [t.name for t in fig.data] == SKY_TRACES[1:]
    # the structures stay above the map: stars are annotations and vertices have customdata
    assert PICK_HINT.count("empty place") == 1 and "add a peak" in PICK_HINT
    assert "eraser" in PICK_HINT and "Erase active shape" in PICK_HINT
    assert "remove it" in PICK_HINT and "uncheck it to click" in PICK_HINT


def test_user_peak_and_shape_events_over_http(conc_explorer, conc_stores):
    ex = conc_explorer
    app = create_app(ex)
    state, sample, env = conc_stores
    s0 = cb.struct_update(ex, None, sample, [], "sample")

    def post(inputs, states, changed):
        outputs = ["store-struct.data", "pick-info.children"]
        out = [{"id": o.split(".")[0], "property": o.split(".", 1)[1]} for o in outputs]
        payload = {"output": ".." + "...".join(outputs) + "..", "outputs": out,
                   "inputs": [{"id": i, "property": p, "value": v} for i, p, v in inputs],
                   "state": [{"id": i, "property": p, "value": v} for i, p, v in states],
                   "changedPropIds": [changed]}
        resp = app.server.test_client().post("/_dash-update-component", json=payload)
        assert resp.status_code == 200, resp.data
        return resp.get_json()["response"]

    def inputs(click_data=None, annotation=None, relayout=None):
        return [("sky-fig", "clickData", click_data), ("sky-fig", "clickAnnotationData", annotation),
                ("sky-fig", "relayoutData", relayout), ("btn-pick-all", "n_clicks", 0),
                ("btn-pick-none", "n_clicks", 0), ("btn-clear-paths", "n_clicks", 0),
                ("store-sample", "data", sample)]

    peaks_mode = [("sky-pick", "value", ["peaks"]), ("store-struct", "data", s0)]
    body = post(inputs(click_data=heat_click(181.2, 2.1)), peaks_mode, "sky-fig.clickData")
    added = body["store-struct"]["data"]
    assert added["user_peaks"] == [{"id": 1, "ra": 181.2, "dec": 2.1}]
    assert body["pick-info"]["children"].endswith("added peaks 1")
    body = post(inputs(annotation={"fullAnnotation": {"name": "user peak 1"}}),
                [("sky-pick", "value", ["peaks"]), ("store-struct", "data", added)],
                "sky-fig.clickAnnotationData")
    assert body["store-struct"]["data"]["user_peaks"] == []
    # the eraser: the ridge shape is not in the remaining shapes
    body = post(inputs(relayout={"shapes": [], "_deactivateShape": True}),
                [("sky-pick", "value", ["paths"]), ("store-struct", "data", s0)],
                "sky-fig.relayoutData")
    assert body["store-struct"]["data"]["off_ridges"] == [1]
    assert body["pick-info"]["children"].startswith("peaks on 2/2, ridges on 0/1")
    # Clear drawn removes the added peaks too
    body = post([*inputs()[:5], ("btn-clear-paths", "n_clicks", 1), ("store-sample", "data", sample)],
                [("sky-pick", "value", ["peaks"]), ("store-struct", "data", added)],
                "btn-clear-paths.n_clicks")
    assert body["store-struct"]["data"]["user_peaks"] == []


def test_struct_and_run_callbacks_over_http(conc_explorer, conc_stores):
    ex = conc_explorer
    app = create_app(ex)
    state, sample, env = conc_stores
    s0 = cb.struct_update(ex, None, sample, [], "sample")

    def post(outputs, inputs, states, changed):
        out = [{"id": o.split(".")[0], "property": o.split(".", 1)[1]} for o in outputs]
        payload = {"output": ".." + "...".join(outputs) + "..", "outputs": out,
                   "inputs": [{"id": i, "property": p, "value": v} for i, p, v in inputs],
                   "state": [{"id": i, "property": p, "value": v} for i, p, v in states],
                   "changedPropIds": [changed]}
        resp = app.server.test_client().post("/_dash-update-component", json=payload)
        assert resp.status_code == 200, resp.data
        return resp.get_json()["response"]

    inputs = [("sky-fig", "clickData", click("peak", 1)), ("sky-fig", "clickAnnotationData", None),
              ("sky-fig", "relayoutData", None), ("btn-pick-all", "n_clicks", 0),
              ("btn-pick-none", "n_clicks", 0), ("btn-clear-paths", "n_clicks", 0),
              ("store-sample", "data", sample)]
    states = [("sky-pick", "value", ["peaks"]), ("store-struct", "data", s0)]
    outputs = ["store-struct.data", "pick-info.children"]
    body = post(outputs, inputs, states, "sky-fig.clickData")
    assert body["store-struct"]["data"]["off_peaks"] == [1]
    assert body["pick-info"]["children"] == "peaks on 1/2, ridges on 1/1, drawn 0, added peaks 0"
    # a drawing becomes a stored path
    inputs[0] = ("sky-fig", "clickData", None)
    inputs[2] = ("sky-fig", "relayoutData", {"shapes": [drawn("M181.5,2.0L181.2,2.2")]})
    body = post(outputs, inputs, [("sky-pick", "value", ["paths"]), states[1]], "sky-fig.relayoutData")
    assert body["store-struct"]["data"]["paths"][0]["ra"] == [181.5, 181.2]
    assert body["pick-info"]["children"].endswith("drawn 1, added peaks 0")
    # an event that changes nothing leaves the outputs alone
    inputs[2] = ("sky-fig", "relayoutData", {"xaxis.range[0]": 1.0})
    assert post(outputs, inputs, [("sky-pick", "value", ["paths"]), states[1]],
                "sky-fig.relayoutData") == {}
    # Run model, Clear model
    run_outputs = ["store-model.data", "model-status.children"]
    run_inputs = [("btn-run-model", "n_clicks", 1), ("btn-clear-model", "n_clicks", 0)]
    run_states = [("store-sample", "data", sample), ("store-struct", "data", s0)]
    body = post(run_outputs, run_inputs, run_states, "btn-run-model.n_clicks")
    model = body["store-model"]["data"]
    assert len(model["peaks"]) == 2 and model["paths"][0]["label"] == "ridge 1"
    assert model["map_key"] == sample["map_key"]
    assert body["model-status"]["children"] == f"model: 2 peaks, 1 paths (run {model['time']})"
    body = post(run_outputs, [("btn-run-model", "n_clicks", 1), ("btn-clear-model", "n_clicks", 1)],
                run_states, "btn-clear-model.n_clicks")
    assert body["store-model"]["data"] is None and body["model-status"]["children"] == "model: not run"
    nothing = cb.struct_update(ex, s0, sample, [], "none")
    body = post(run_outputs, run_inputs, [run_states[0], ("store-struct", "data", nothing)],
                "btn-run-model.n_clicks")
    assert "store-model" not in body and "cannot run" in body["model-status"]["children"]
    # the sample callback carries the model, the panel callback shows its table
    spec = cb.run_model(ex, sample, s0)[0]
    keys = list(cb.SAMPLE_KEYS)
    full = cb.make_state(ex, model=spec, concregion="inside")
    body = post(["store-sample.data", "sample-counts.children"],
                [(cid, prop, full[key]) for key, cid, prop in STATE_CONTROLS if key in keys],
                [], "store-model.data")
    assert body["store-sample"]["data"]["filters"]["conc_model"]["peaks"] == spec["peaks"]
    assert body["store-sample"]["data"]["model"]["time"] == spec["time"]
    plain = cb.sample_info(ex, cb.make_state(ex, model=spec))
    table = post(["conc-summary.children", "conc-table.children", "conc-caption.children",
                  "model-table.children"],
                 [("store-sample", "data", plain), ("store-env", "data", env), ("store-sel", "data", None)],
                 [], "store-sample.data")
    assert table["model-table"]["children"]["type"] == "Div"
    assert "Inside = within the 1σ band" in table["conc-caption"]["children"]


def test_snapshot_keeps_the_model_and_the_layout_has_the_controls(picked):
    ex, sample, _, _, s0, _ = picked
    assert ("model", "store-model", "data") in STATE_CONTROLS and "model" in cb.SAMPLE_KEYS
    assert default_state(ex)["model"] is None and set(default_state(ex)) == set(STATE_KEYS)
    spec = cb.run_model(ex, sample, s0)[0]
    text = ex.snapshot(cb.make_state(ex, model=spec, concregion="inside"))
    values, _ = cb.restore_state(ex, ex.load_snapshot(text))
    restored = dict(zip(STATE_KEYS, values))
    assert restored["model"] == json.loads(json.dumps(spec)) and restored["concregion"] == "inside"
    assert cb.model_status(restored["model"]) == f"model: 2 peaks, 1 paths (run {spec['time']})"
    old = {k: v for k, v in default_state(ex).items() if k != "model"}     # a snapshot without a model
    assert dict(zip(STATE_KEYS, cb.restore_state(ex, {"state": old})[0]))["model"] is None
    # the controls of the sky map
    app = create_app(ex)
    by_id = {c.id: c for c in app.layout._traverse() if getattr(c, "id", None)}
    pick = by_id["sky-pick"]
    assert [(o["label"], o["value"]) for o in pick.options] == [(" Select peaks", "peaks"),
                                                                (" Select maximums", "paths")]
    assert pick.value == [] and pick.inline is True
    assert "click a star, a ridge or a drawn path" in PICK_HINT and "uncheck it to click" in PICK_HINT
    assert [by_id[i].children for i in ("btn-pick-all", "btn-pick-none", "btn-clear-paths",
                                        "btn-run-model", "btn-clear-model")] == [
        "All on", "All off", "Clear drawn", "Run model", "Clear model"]
    assert by_id["model-status"].children == "model: not run"
    assert by_id["store-struct"].data == default_struct() and by_id["store-model"].data is None
    assert by_id["pick-controls"].style == {} and PICK_HINT in [
        getattr(c, "children", None) for c in by_id["pick-controls"]._traverse()]
    assert not by_id["btn-run-model"].disabled


def test_csv_export_has_the_region_and_the_model(picked):
    ex, sample, env, state, s0, _ = picked
    frame = cb.export_csv_frame(ex, state, env, sample)
    idx = frame["row"].to_numpy()
    conc_inside = cb.concentration_result(ex, sample)[0]["conc_inside"]
    np.testing.assert_array_equal(frame["conc_inside"].to_numpy(), conc_inside[idx])
    assert not {"model_inside", "model_dist_sigma", "model_structure"} & set(frame.columns)
    spec = cb.run_model(ex, sample, s0)[0]
    held = cb.sample_info(ex, cb.make_state(ex, model=spec))
    with_model = cb.export_csv_frame(ex, state, env, held)
    columns = cb.model_result(ex, held)[0]
    for name in ("model_inside", "model_dist_sigma", "model_structure"):
        np.testing.assert_array_equal(with_model[name].to_numpy(),
                                      np.asarray(columns[name])[with_model["row"].to_numpy()])
    assert set(with_model["model_structure"]) <= {"", "peak 1", "peak 2", "ridge 1"}
    assert "conc_inside" in with_model.columns


def test_line_trace_cuts_polylines_at_missing_points():
    nan = float("nan")
    xs, ys = cb._line_trace([([1.0, 2.0, nan, 4.0, 5.0], [1.0, 2.0, 3.0, 4.0, 5.0]),
                             ([nan, nan], [1.0, 2.0]), ([7.0, 8.0], [None, None]),
                             ([np.float64(9.0), 10.0], [9.0, 10.0])])
    assert xs == [1.0, 2.0, None, 4.0, 5.0, None, 9.0, 10.0, None]
    assert ys == [1.0, 2.0, None, 4.0, 5.0, None, 9.0, 10.0, None]
    assert all(type(v) in (float, type(None)) for v in xs + ys)
    assert cb._line_trace([]) == ([], [])


def test_model_overlay_never_sends_nan(picked, monkeypatch):
    import copy
    ex, sample, env, state, s0, _ = picked
    spec = cb.run_model(ex, sample, s0)[0]
    held = cb.sample_info(ex, cb.make_state(ex, model=spec))
    real = cb.model_result
    nan = float("nan")

    def outside(explorer, sample_):
        columns, model = real(explorer, sample_)
        model = copy.deepcopy(model)                       # the cached model is not modified
        path = model["paths"][0]
        for key in ("left_ra", "left_dec", "right_ra", "right_dec"):
            path[key][2] = nan                             # one point of each edge is outside
        path["left_ra"][-1] = nan
        for key in ("circle_ra", "circle_dec"):
            model["peaks"][0][key][5] = nan
            model["peaks"][1][key][0] = nan
        return columns, model

    monkeypatch.setattr(cb, "model_result", outside)
    sky = cb.sky_callback(ex, state, env, held, struct=s0, pick=[])
    band = next(t for t in sky.data if t.name == "model: 1σ band")
    circles = next(t for t in sky.data if t.name == "model: peak circles")
    fitted = next(t for t in sky.data if t.name == "model: fitted paths")
    for trace in (band, circles, fitted):
        for values in (trace.x, trace.y):
            assert all(v is None or np.isfinite(v) for v in values)
        assert list(trace.x).count(None) == list(trace.y).count(None)
        assert [v is None for v in trace.x] == [v is None for v in trace.y]
    assert list(band.x).count(None) > 2 and list(circles.x).count(None) > 2   # the lines were cut
    assert "NaN" not in sky.to_json()
    assert cb.concentration_overlay(ex, held, s0)["model"] is not None
