import base64
import json

import numpy as np
import pytest

from knn.app import callbacks as cb
from knn.app import create_app
from knn.app.layout import STATE_CONTROLS, STATE_KEYS, default_state, logm_bounds

EXPECTED_IDS = {
    "f-z", "f-ztype", "f-pwin", "f-odds", "f-logm", "f-mr", "f-member", "f-subs", "f-group",
    "f-quenched", "f-morph", "sample-counts",
    "env-estimator", "env-k", "env-weight", "env-dv", "env-nsigma", "env-pzmode", "env-note",
    "env-status", "v3d-coords", "view-color", "view-size", "view-clip", "sky-density",
    "v3d-fig", "sky-fig", "clear-selection", "sel-info", "an-y", "an-x", "an-massbins",
    "an-nbins", "an-fig", "an-hist", "an-counts", "btn-csv", "btn-snapshot", "btn-pdf",
    "up-snapshot", "snapshot-status", "export-imgfmt", "dl-csv", "dl-snapshot", "dl-pdf",
    "store-sample", "store-env", "store-sel",
}


@pytest.fixture(scope="module")
def app(explorer):
    return create_app(explorer)


def _ids(component):
    return {c.id for c in component._traverse() if getattr(c, "id", None)}


def test_create_app_and_layout_ids(app, explorer):
    assert EXPECTED_IDS <= _ids(app.layout)
    assert {cid for _, cid, _ in STATE_CONTROLS} <= _ids(app.layout)
    assert len(set(STATE_KEYS)) == len(STATE_KEYS)


def test_http_endpoints(app):
    client = app.server.test_client()
    page = client.get("/")
    assert page.status_code == 200 and b"_dash-config" in page.data
    layout = client.get("/_dash-layout")
    assert layout.status_code == 200 and b"sky-fig" in layout.data
    css = client.get("/assets/style.css")
    assert css.status_code == 200 and b".sidebar" in css.data
    assert client.get("/_dash-dependencies").status_code == 200


def test_callbacks_are_registered(app):
    outputs = " ".join(app.callback_map)
    for target in ("sky-fig.figure", "v3d-fig.figure", "an-fig.figure", "an-hist.figure",
                   "store-env.data", "store-sample.data", "store-sel.data", "dl-csv.data",
                   "dl-pdf.data", "dl-snapshot.data", "env-note.children", "snapshot-status"):
        assert target in outputs, target


def _call(app, outputs, inputs, changed=None, states=()):
    """POST to /_dash-update-component the way the browser does."""
    out = [{"id": o.split(".")[0], "property": o.split(".", 1)[1]} for o in outputs]
    payload = {"output": ".." + "...".join(outputs) + ".." if len(outputs) > 1 else outputs[0],
               "outputs": out if len(out) > 1 else out[0],
               "inputs": [{"id": i, "property": p, "value": v} for i, p, v in inputs],
               "state": [{"id": i, "property": p, "value": v} for i, p, v in states],
               "changedPropIds": changed or [f"{inputs[0][0]}.{inputs[0][1]}"]}
    return app.server.test_client().post("/_dash-update-component", json=payload)


def test_update_component_endpoint_changes_k(app, explorer):
    inputs = [("env-estimator", "value", "sigma2d"), ("env-k", "value", 7),
              ("env-weight", "value", "count"), ("env-dv", "value", 1500.0),
              ("env-nsigma", "value", 1.5), ("env-pzmode", "value", "gauss")]
    resp = _call(app, ["store-env.data", "env-status.children"], inputs, ["env-k.value"])
    assert resp.status_code == 200, resp.data
    body = resp.get_json()["response"]
    assert body["store-env"]["data"]["k"] == 7 and body["store-env"]["data"]["n_valid"] > 0
    assert "spec-z" in body["env-status"]["children"]
    # the figure callback then draws from the stored parameters
    sample = cb.sample_info(explorer, default_state(explorer))
    inputs = [("store-sample", "data", sample), ("store-env", "data", body["store-env"]["data"]),
              ("clear-selection", "n_clicks", 0), ("view-color", "value", "env"),
              ("view-clip", "value", [2, 98]), ("view-size", "value", None),
              ("sky-density", "value", ["on"])]
    resp = _call(app, ["sky-fig.figure"], inputs, ["store-env.data"])
    assert resp.status_code == 200, resp.data
    figure = resp.get_json()["response"]["sky-fig"]["figure"]
    assert any(t["type"] == "scattergl" for t in figure["data"])


@pytest.fixture(scope="module")
def stores(explorer):
    state = default_state(explorer)
    return state, cb.sample_info(explorer, state), cb.env_info(explorer, state)


def test_default_state_and_filters(explorer):
    state = default_state(explorer)
    assert set(state) == set(STATE_KEYS)
    assert (state["estimator"], state["k"], state["weight"]) == ("sigma2d", 5, "count")
    assert state["y"] == "quenched" and state["x"] == "env" and state["mass_bins"] == "8, 9, 10, 11"
    # a default state restricts nothing but the redshift window and the z types
    f = cb.filters_from_state(explorer, state)
    assert set(f) == {"z_range", "z_types"}
    assert explorer.select(f).all()
    lo, hi = logm_bounds(explorer)
    assert lo < hi
    f = cb.filters_from_state(explorer, cb.make_state(
        explorer, logm_range=[lo + 1, hi], mr_max=-15.0, member=[1.0], subs=["S1", "(none)"],
        groups=["2", "(none)"], quenched="yes", morph=["yes"], p_min=0.5, odds_min=0.4,
        z_types=[]))
    assert f["group"] == [2, "(none)"] and f["quenched"] == "yes" and f["morphology"] is True
    assert f["z_types"] == ["__none__"] and f["logm_range"] == [lo + 1, hi]
    assert not explorer.select(f).any()
    with pytest.raises(KeyError):
        cb.make_state(explorer, nonsense=1)


def test_sample_and_env_callbacks(explorer, stores):
    state, sample, env = stores
    assert sample["n_spec"] + sample["n_phot"] == len(explorer.df)
    assert "Selected:" in sample["text"]
    assert env["name"] == "sigma2d" and env["k"] == 5 and env["n_valid"] > 0
    assert env["seconds"] >= 0 and "photo-z" in env["text"]
    # a second call is a cache hit
    assert cb.env_info(explorer, state)["seconds"] < max(0.05, env["seconds"])
    narrow = cb.sample_info(explorer, cb.make_state(explorer, z_types=["spec"], quenched="yes"))
    assert narrow["n_phot"] == 0 and 0 < narrow["n_spec"] < sample["n_spec"]
    knn = cb.env_info(explorer, cb.make_state(explorer, estimator="knn3d"))
    assert knn["n_valid"] == int(explorer.is_spec.sum())
    assert "REPORT_photoz_validation.md" in cb.env_note("knn3d") and cb.env_note("sigma2d") == ""


def test_figure_callbacks_with_default_inputs(explorer, stores):
    state, sample, env = stores
    sky = cb.sky_callback(explorer, state, env, sample)
    assert [t.name for t in sky.data] == ["density", "spec-z", "photo-z"]
    f3 = cb.scatter3d_callback(explorer, state, env, sample)
    assert [t.name for t in f3.data] == ["spec-z", "photo-z"]
    trend, hist, text = cb.analysis_callback(explorer, state, env, sample)
    assert len(trend.data) > 0 and [t.name for t in hist.data] == ["full sample", "selection"]
    assert "N/bin" in text and "log M*" in text
    n_points = sum(len(t.x) for t in sky.data[1:])
    assert n_points == len(explorer.df)


def test_figure_callbacks_with_other_options(explorer, stores):
    _, sample, env = stores
    state = cb.make_state(explorer, coords="xyz", color="logmstar", size="M_r", clip=[10, 90],
                          density=[], y="logmstar", x="logmstar", mass_bins="", nbins=4)
    f3 = cb.scatter3d_callback(explorer, state, env, sample, sel_rows=[0, 1, 2])
    assert f3.data[-1].name == "selection" and len(f3.data[-1].x) == 3
    assert f3.layout.scene.xaxis.title.text == "x [Mpc]"
    sky = cb.sky_callback(explorer, state, env, sample, clear_clicks=2)
    assert sky.data[0].name != "density" and sky.layout.uirevision == "sky2"
    trend, hist, text = cb.analysis_callback(explorer, state, env, sample)
    assert (cb.analysis_trend(explorer, state, env, sample)[0]["mass_bin"] == "all").all()
    boolean_colour = cb.sky_callback(explorer, cb.make_state(explorer, color="quenched"), env,
                                     sample)
    assert boolean_colour.layout.coloraxis.cmin <= 0.5 <= boolean_colour.layout.coloraxis.cmax
    knn_env = cb.env_info(explorer, cb.make_state(explorer, estimator="knn3d"))
    f = cb.sky_callback(explorer, cb.make_state(explorer, estimator="knn3d"), knn_env, sample)
    assert any(t.name == "no value" for t in f.data)          # photo-z rows have no knn3d value


def test_selection_restricts_the_analysis(explorer, stores):
    state, sample, env = stores
    points = {"points": [{"customdata": [float(r), 0.0]} for r in (3, 5, 5, 8)] + [{}]}
    rows = cb.selection_rows(points)
    assert rows == [3, 5, 8]
    assert cb.selection_rows(None) is None and cb.selection_rows({"points": []}) is None
    mask = cb.selection_mask(explorer, sample, rows)
    assert np.flatnonzero(mask).tolist() == rows
    assert cb.selection_mask(explorer, sample, None).all()
    narrow = cb.make_state(explorer, mass_bins="", nbins=3, y="logmstar")
    trend, _, _ = cb.analysis_trend(explorer, narrow, env, sample, list(range(60)))
    assert trend["n"].sum() <= 60
    narrowed = cb.sample_info(explorer, cb.make_state(explorer, z_types=["spec"]))
    assert cb.selection_mask(explorer, narrowed, list(range(len(explorer.df)))).sum() == narrowed["n_spec"]


def test_bad_mass_bins_give_a_message_not_an_exception(explorer, stores):
    _, sample, env = stores
    for text in ("8, abc", "9"):
        state = cb.make_state(explorer, mass_bins=text)
        trend, hist, msg = cb.analysis_callback(explorer, state, env, sample)
        assert msg.startswith("Invalid input") and len(trend.data) == 0
    assert cb.parse_mass_bins("") is None and cb.parse_mass_bins(None) is None
    assert cb.parse_mass_bins("11 9,10") == [9.0, 10.0, 11.0]


def test_export_csv_pdf_and_snapshot(explorer, stores):
    state, sample, env = stores
    frame = cb.export_csv_frame(explorer, cb.make_state(explorer, color="M_r", size="rs_offset"),
                                env, sample, [0, 1, 2, 3])
    assert len(frame) == 4
    assert list(frame.columns)[:10] == list(cb.CSV_BASE_COLUMNS)
    assert {"M_r", "rs_offset", "quenched"} <= set(frame.columns)
    assert frame["row"].tolist() == [0, 1, 2, 3]
    np.testing.assert_array_equal(frame["env"], explorer.env("sigma2d", 5)["env"].to_numpy()[:4])
    full = cb.export_csv_frame(explorer, state, env, sample)
    assert len(full) == len(explorer.df)
    pdf = cb.publication_pdf(explorer, state, env, sample)
    assert pdf.startswith(b"%PDF")

    modified = cb.make_state(explorer, k=9, quenched="no", z_range=[0.012, 0.03], color="M_r",
                             mass_bins="9, 10", imgfmt="svg")
    text = explorer.snapshot(modified)
    record = explorer.load_snapshot(text)
    values, message = cb.restore_state(explorer, record)
    assert dict(zip(STATE_KEYS, values)) == json.loads(json.dumps(modified))
    assert message == "Snapshot loaded."
    values, message = cb.restore_state(explorer, {"state": {"k": 4, "bogus": 1},
                                                  "matches_config": False})
    restored = dict(zip(STATE_KEYS, values))
    assert restored["k"] == 4 and restored["estimator"] == "sigma2d" and "different config" in message


def test_download_callbacks_over_http(app, explorer, stores):
    state, sample, env = stores
    shared = [("store-sample", "data", sample), ("store-env", "data", env),
              ("store-sel", "data", [0, 1, 2, 3, 4]), ("view-color", "value", "env"),
              ("view-size", "value", None), ("an-y", "value", "quenched"),
              ("an-x", "value", "env"), ("an-massbins", "value", "8, 9, 10, 11"),
              ("an-nbins", "value", 8)]
    resp = _call(app, ["dl-csv.data"], [("btn-csv", "n_clicks", 1)], states=shared)
    assert resp.status_code == 200, resp.data
    data = resp.get_json()["response"]["dl-csv"]["data"]
    assert data["filename"] == "knn_selection.csv" and len(data["content"].splitlines()) == 6
    resp = _call(app, ["dl-pdf.data"], [("btn-pdf", "n_clicks", 1)], states=shared)
    assert resp.status_code == 200, resp.data
    data = resp.get_json()["response"]["dl-pdf"]["data"]
    assert data["filename"] == "knn_trend.pdf" and data["base64"]
    assert base64.b64decode(data["content"]).startswith(b"%PDF")
    snap = [(cid, prop, state[key]) for key, cid, prop in STATE_CONTROLS]
    resp = _call(app, ["dl-snapshot.data"], [("btn-snapshot", "n_clicks", 1)] + snap)
    assert resp.status_code == 200, resp.data
    data = resp.get_json()["response"]["dl-snapshot"]["data"]
    assert data["filename"] == "knn_snapshot.yaml"
    assert explorer.load_snapshot(data["content"])["state"]["k"] == state["k"]


def test_graph_config_callback_over_http(app):
    key = next(k for k in app.callback_map if "v3d-fig.config" in k)
    resp = _call(app, key[2:-2].split("..."), [("export-imgfmt", "value", "svg")])
    assert resp.status_code == 200
    configs = resp.get_json()["response"]
    assert configs["sky-fig"]["config"]["toImageButtonOptions"]["format"] == "svg"
    assert configs["an-hist"]["config"]["toImageButtonOptions"]["filename"] == "knn_hist"


def test_restore_callback_over_http(app, explorer):
    text = explorer.snapshot(cb.make_state(explorer, k=11))
    contents = "data:text/yaml;base64," + base64.b64encode(text.encode()).decode()
    key = next(k for k in app.callback_map if "snapshot-status" in k)
    outputs = key[2:-2].split("...")
    resp = _call(app, outputs, [("up-snapshot", "contents", contents)], ["up-snapshot.contents"])
    assert resp.status_code == 200, resp.data
    body = resp.get_json()["response"]
    assert body["env-k"]["value"] == 11 and body["snapshot-status"]["children"] == "Snapshot loaded."
    bad = _call(app, outputs, [("up-snapshot", "contents", "data:text/yaml;base64,Zm9v")],
                ["up-snapshot.contents"])
    assert bad.status_code == 200
    assert "Cannot load the snapshot" in bad.get_json()["response"]["snapshot-status"]["children"]


def test_serve_command_line(explorer, monkeypatch):
    from knn import cli
    seen = {}
    monkeypatch.setattr(cli, "serve", lambda cfg, **kw: seen.update(cfg=cfg, **kw))
    path = explorer.cfg["_meta"]["path"]
    assert cli.main(["serve", path, "--port", "8765", "--debug"]) == 0
    assert seen["port"] == 8765 and seen["debug"] is True and seen["host"] == "127.0.0.1"
    assert seen["cfg"]["_meta"]["path"] == path
    assert cli.main(["serve", path, "--host", "0.0.0.0"]) == 0 and seen["port"] == 8050
    assert cli.main(["serve", "/nonexistent.yaml"]) == 2
