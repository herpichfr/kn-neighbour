"""Component tree of the explorer and its default state.

STATE_CONTROLS lists every control as (state key, component id, property);
default_state(explorer) gives the value of each state key at start, and the
snapshot of the app is a dict with these keys. The selection in progress on the sky map
(store-struct) is not part of the snapshot; the model that was run (store-model) is.
"""

import math

import numpy as np
from dash import dcc, html

from ..explorer import MASS_TABLE_REGIONS, MASS_TABLE_STATS, MASS_TABLE_ZTYPES, NONE_LABEL
from .figures import empty_figure, graph_config

STATE_CONTROLS = (
    ("z_range", "f-z", "value"),
    ("z_types", "f-ztype", "value"),
    ("p_min", "f-pwin", "value"),
    ("odds_min", "f-odds", "value"),
    ("logm_range", "f-logm", "value"),
    ("mr_max", "f-mr", "value"),
    ("member", "f-member", "value"),
    ("subs", "f-subs", "value"),
    ("groups", "f-group", "value"),
    ("quenched", "f-quenched", "value"),
    ("morph", "f-morph", "value"),
    ("concregion", "f-concregion", "value"),
    ("concw", "conc-weight", "value"),
    ("conch", "conc-bw", "value"),
    ("concstep", "conc-step", "value"),
    ("concprom", "conc-prom", "value"),
    ("estimator", "env-estimator", "value"),
    ("k", "env-k", "value"),
    ("weight", "env-weight", "value"),
    ("dv", "env-dv", "value"),
    ("n_sigma", "env-nsigma", "value"),
    ("pzmode", "env-pzmode", "value"),
    ("coords", "v3d-coords", "value"),
    ("color", "view-color", "value"),
    ("size", "view-size", "value"),
    ("clip", "view-clip", "value"),
    ("density", "sky-density", "value"),
    ("y", "an-y", "value"),
    ("x", "an-x", "value"),
    ("mass_bins", "an-massbins", "value"),
    ("nbins", "an-nbins", "value"),
    ("imgfmt", "export-imgfmt", "value"),
    ("model", "store-model", "data"),
)
STATE_KEYS = tuple(key for key, _, _ in STATE_CONTROLS)
SAMPLE_KEYS = ("z_range", "z_types", "p_min", "odds_min", "logm_range", "mr_max", "member",
               "subs", "groups", "quenched", "morph", "concregion", "concw", "conch", "concstep",
               "concprom", "model")
ENV_KEYS = ("estimator", "k", "weight", "dv", "n_sigma", "pzmode")
KNN3D_NOTE = "not defined for photo-z galaxies (see REPORT_photoz_validation.md)"
DEFAULT_MASS_BINS = "8, 9, 10, 11"
PICK_HINT = ("Select peaks: click a star, a ridge or a drawn path to switch it on / off. Select "
             "maximums: drag along a ridge to draw a path (clicks do not register while drawing: "
             "uncheck it to click). Run model fits the structures that are on to the crest of the "
             "map and gives the 1σ band around them.")
STRUCTURE_HEADER = ("structure", "kind", "s [Mpc]", "intrinsic s [Mpc]", "crest [σ] (mean / min)",
                    "length [Mpc]", "shift [Mpc]", "N in band")


def default_struct():
    """Value of store-struct at start: the structures of the sky map that are switched off, the
    drawn paths ({id, ra, dec, on, src}) and the map (map_key) the off ids belong to."""
    return {"map_key": None, "off_peaks": [], "off_ridges": [], "paths": [], "next_id": 1}


def logm_bounds(explorer):
    """Integer (floor, ceil) of the log mass range of the catalogue; (6, 12) without masses."""
    if not explorer.has_column("logmstar"):
        return 6, 12
    m = explorer.df["logmstar"].to_numpy(float)
    m = m[np.isfinite(m)]
    if m.size == 0:
        return 6, 12
    return int(math.floor(m.min())), int(math.ceil(m.max()))


def default_state(explorer):
    """Value of every control at start (keys STATE_KEYS)."""
    env = explorer.default_environment()
    lo, hi = logm_bounds(explorer)
    boolean = explorer.boolean_columns()
    numeric = explorer.numeric_columns()
    y = "quenched" if "quenched" in boolean else (boolean[0] if boolean else numeric[0])
    conc = explorer.conc_settings
    z_range = list(conc["z_range"]) if conc and conc["z_range"] else list(explorer.window)
    z_types = ["spec"] if conc and conc["sample"] == "spec" else ["spec", "phot"]
    conc_ctl = explorer.default_concentration() if conc else {
        "weight": "count", "bandwidth_mpc": None, "contour_step_sigma": 1.0,
        "min_prominence_sigma": 1.0}
    return {
        "z_range": z_range, "z_types": z_types, "p_min": 0.0,
        "odds_min": 0.0, "logm_range": [lo, hi], "mr_max": None, "member": [], "subs": [],
        "groups": [], "quenched": "any", "morph": [], "concregion": "any",
        "concw": conc_ctl["weight"], "conch": conc_ctl["bandwidth_mpc"],
        "concstep": conc_ctl["contour_step_sigma"], "concprom": conc_ctl["min_prominence_sigma"],
        "estimator": env["name"], "k": env["k"], "weight": env["weight"],
        "dv": env["dv"] if env["dv"] is not None else 1500.0,
        "n_sigma": env["n_sigma"] if env["n_sigma"] is not None else 1.5,
        "pzmode": env["photoz_mode"] if env["photoz_mode"] != "none" else explorer.settings["mode"],
        "coords": "radecz", "color": "env", "size": None, "clip": [2, 98], "density": ["on"],
        "y": y, "x": "env", "mass_bins": DEFAULT_MASS_BINS if explorer.has_column("logmstar") else "",
        "nbins": 8, "imgfmt": "png", "model": None,
    }


def _options(values):
    return [{"label": f"{v:g}" if isinstance(v, float) else str(v), "value": v} for v in values]


def conc_weight_options(explorer):
    """Weights of the concentration map: count, mass (if the masses exist), numeric columns."""
    names = ["count"] + (["mass"] if explorer.has_column("logmstar") else [])
    return names + [c for c in explorer.numeric_columns(include_env=False) if c not in names]


def _field(label, control, hint=None):
    children = [html.Label(label, className="field-label"), control]
    if hint:
        children.append(html.Div(hint, className="hint"))
    return html.Div(children, className="field")


def _section(title, children, open_=True):
    return html.Details([html.Summary(title)] + children, open=open_, className="section")


def _sample_section(explorer, st):
    lo, hi = logm_bounds(explorer)
    zlo, zhi = explorer.window
    choices = explorer.choices()
    marks = {int(v): str(int(v)) for v in range(lo, hi + 1)}
    zmarks = {float(v): f"{v:g}" for v in (zlo, (zlo + zhi) / 2, zhi)}
    has_conc = explorer.conc_settings is not None
    return _section("Sample", [
        _field("Redshift", dcc.RangeSlider(id="f-z", min=zlo, max=zhi, step=0.001,
                                           value=st["z_range"], marks=zmarks)),
        _field("Redshift type", dcc.Checklist(
            id="f-ztype", options=[{"label": " spec-z", "value": "spec"},
                                   {"label": " photo-z", "value": "phot"}],
            value=st["z_types"], inline=True, className="check")),
        _field("Min p(in window), photo-z", dcc.Input(
            id="f-pwin", type="number", min=0, max=1, step=0.05, value=st["p_min"],
            debounce=True, className="num")),
        _field("Min photo-z odds", dcc.Input(
            id="f-odds", type="number", min=0, max=1, step=0.05, value=st["odds_min"],
            debounce=True, className="num")),
        _field("log M* range", dcc.RangeSlider(id="f-logm", min=lo, max=hi, step=0.1,
                                               value=st["logm_range"], marks=marks)),
        _field("M_r max", dcc.Input(id="f-mr", type="number", step=0.1, value=st["mr_max"],
                                    placeholder="no cut", debounce=True, className="num")),
        _field("Membership (flag_member)", dcc.Dropdown(
            id="f-member", options=_options(choices.get("flag_member", [])),
            value=st["member"], multi=True, placeholder="all")),
        _field("Substructure", dcc.Dropdown(
            id="f-subs", options=_options(choices.get("substructure", []) + [NONE_LABEL]),
            value=st["subs"], multi=True, placeholder="all")),
        _field("Group (photo-z group id)", dcc.Dropdown(
            id="f-group", options=_options([str(g) for g in choices.get("group", [])]
                                           + [NONE_LABEL]),
            value=st["groups"], multi=True, placeholder="all")),
        _field("Quenched", dcc.RadioItems(
            id="f-quenched", options=[{"label": " any", "value": "any"},
                                      {"label": " yes", "value": "yes"},
                                      {"label": " no", "value": "no"}],
            value=st["quenched"], inline=True, className="check")),
        dcc.Checklist(id="f-morph", options=[{"label": " morphology available",
                                              "value": "yes"}],
                      value=st["morph"], className="check"),
        _field("1σ region", dcc.RadioItems(
            id="f-concregion",
            options=[{"label": " all", "value": "any", "disabled": not has_conc},
                     {"label": " inside", "value": "inside", "disabled": not has_conc},
                     {"label": " outside", "value": "outside", "disabled": not has_conc}],
            value=st["concregion"], inline=True, className="check"),
            None if has_conc else "needs a concentration section in the config"),
        html.Div(id="sample-counts", className="counts"),
    ])


def _concentration_section(explorer, st):
    off = explorer.conc_settings is None
    return _section("Concentration", [
        _field("Weight of the map", dcc.Dropdown(
            id="conc-weight", options=_options(conc_weight_options(explorer)), value=st["concw"],
            clearable=False, disabled=off),
               "count, stellar mass (capped), or a catalogue column"),
        _field("Bandwidth [Mpc]", dcc.Input(
            id="conc-bw", type="number", min=0.1, step=0.1, value=st["conch"], debounce=True,
            placeholder="Scott", className="num", disabled=off),
               "empty = Scott's rule on the sample positions"),
        _field("Contour step [σ]", dcc.Input(
            id="conc-step", type="number", min=0.1, step=0.1, value=st["concstep"],
            debounce=True, className="num", disabled=off),
               "levels at mean + 1σ, then every step"),
        _field("Min prominence [σ]", dcc.Input(
            id="conc-prom", type="number", min=0, step=0.1, value=st["concprom"],
            debounce=True, className="num", disabled=off),
               "peaks lower than this above their saddle are not marked"),
    ], open_=not off)


def _environment_section(explorer, st):
    weights = [{"label": " count", "value": "count"}]
    if explorer.has_column("logmstar"):
        weights.append({"label": " mass", "value": "mass"})
    return _section("Environment", [
        _field("Estimator", dcc.Dropdown(id="env-estimator", options=_options(explorer.estimators()),
                                         value=st["estimator"], clearable=False)),
        _field("k (neighbour)", dcc.Input(id="env-k", type="number", min=3, max=50, step=1,
                                          value=st["k"], debounce=True, className="num")),
        _field("Weight", dcc.RadioItems(id="env-weight", options=weights, value=st["weight"],
                                        inline=True, className="check")),
        _field("dv [km/s] (sigma2d, spec-z slice)", dcc.Input(
            id="env-dv", type="number", min=100, step=100, value=st["dv"], debounce=True,
            className="num")),
        _field("n_sigma (sigma2d, photo-z slice)", dcc.Input(
            id="env-nsigma", type="number", min=0.1, step=0.1, value=st["n_sigma"],
            debounce=True, className="num")),
        _field("Photo-z mode", dcc.Dropdown(
            id="env-pzmode", options=_options(["gauss", "point", "pdf"]), value=st["pzmode"],
            clearable=False)),
        html.Div(id="env-note", className="note"),
        html.Div(id="env-status", className="counts"),
    ])


def _view_section(explorer, st):
    cols = explorer.numeric_columns() + explorer.boolean_columns()
    return _section("View", [
        _field("3D coordinates", dcc.RadioItems(
            id="v3d-coords", options=[{"label": " RA, Dec, z", "value": "radecz"},
                                      {"label": " x, y, z [Mpc]", "value": "xyz"}],
            value=st["coords"], inline=True, className="check")),
        _field("Colour by", dcc.Dropdown(id="view-color", options=_options(cols),
                                         value=st["color"], clearable=False)),
        _field("Size by", dcc.Dropdown(id="view-size", options=_options(cols),
                                       value=st["size"], placeholder="constant")),
        _field("Colour range percentiles", dcc.RangeSlider(
            id="view-clip", min=0, max=100, step=1, value=st["clip"],
            marks={0: "0", 25: "25", 50: "50", 75: "75", 100: "100"})),
        dcc.Checklist(id="sky-density", options=[{"label": " density background (sky)",
                                                  "value": "on"}],
                      value=st["density"], className="check"),
    ])


def _export_section(explorer, st):
    return _section("Export", [
        html.Div([
            html.Button("Selection (CSV)", id="btn-csv", n_clicks=0),
            html.Button("Snapshot (YAML)", id="btn-snapshot", n_clicks=0),
            html.Button("Publication figure (PDF)", id="btn-pdf", n_clicks=0),
        ], className="buttons"),
        dcc.Upload(id="up-snapshot", children=html.Div("Drop or click to load a snapshot"),
                   className="upload", multiple=False),
        html.Div(id="snapshot-status", className="note"),
        _field("Modebar image format", dcc.RadioItems(
            id="export-imgfmt", options=[{"label": " PNG", "value": "png"},
                                         {"label": " SVG", "value": "svg"}],
            value=st["imgfmt"], inline=True, className="check"),
               "SVG of WebGL traces (3D, sky) is rasterised by plotly."),
        dcc.Download(id="dl-csv"), dcc.Download(id="dl-snapshot"), dcc.Download(id="dl-pdf"),
    ])


def _graph(graph_id, height, name):
    return dcc.Loading(dcc.Graph(id=graph_id, figure=empty_figure("Loading..."),
                                 config=graph_config("png", name),
                                 style={"height": height}), type="dot")


def _analysis_section(explorer, st):
    y_cols = explorer.boolean_columns() + explorer.numeric_columns()
    return html.Div([
        html.H3("Indicator vs environment"),
        html.Div([
            _field("y (indicator)", dcc.Dropdown(id="an-y", options=_options(y_cols),
                                                 value=st["y"], clearable=False)),
            _field("x", dcc.Dropdown(id="an-x", options=_options(explorer.numeric_columns()),
                                     value=st["x"], clearable=False)),
            _field("log M* bin edges", dcc.Input(
                id="an-massbins", type="text", value=st["mass_bins"], debounce=True,
                placeholder="empty = no mass bins", className="text"),
                   "comma separated"),
            _field("Bins in x", dcc.Input(id="an-nbins", type="number", min=3, max=30, step=1,
                                          value=st["nbins"], debounce=True, className="num")),
        ], className="row"),
        html.Div([
            html.Div(_graph("an-fig", "420px", "knn_trend"), className="col"),
            html.Div(_graph("an-hist", "420px", "knn_hist"), className="col"),
        ], className="row"),
        html.Pre(id="an-counts", className="counts-box"),
    ], className="analysis")


def _format_cell(row):
    """Text of one cell of the statistics table from a row of Explorer.mass_table."""
    if row["stat"] == "n":
        return f"{int(row['n'])}"
    if row["n"] == 0 or not np.isfinite(row["value"]):
        return "-"
    if row["stat"] == "mtot":
        return f"{row['value']:.2f} (N={int(row['n'])})"
    return f"{row['value']:.2f} [{row['lo']:.2f}, {row['hi']:.2f}] N={int(row['n'])}"


def mass_table_rows(table):
    """Rows of strings of the statistics table: a header and one row per statistic.

    table is Explorer.mass_table. Columns: the statistic, then inside
    (spec-z, photo-z, all) and outside (spec-z, photo-z, all).
    """
    names = {"spec": "spec-z", "phot": "photo-z", "all": "all"}
    header = ["statistic"] + [f"{region} {names[z]}" for region in MASS_TABLE_REGIONS
                              for z in MASS_TABLE_ZTYPES]
    rows = [header]
    present = set(table["stat"])
    for stat, label, _, _ in MASS_TABLE_STATS:
        if stat not in present:
            continue
        sub = table[table["stat"] == stat]
        cells = [label]
        for region in MASS_TABLE_REGIONS:
            for ztype in MASS_TABLE_ZTYPES:
                hit = sub[(sub["region"] == region) & (sub["z_type"] == ztype)]
                cells.append(_format_cell(hit.iloc[0]) if len(hit) else "-")
        rows.append(cells)
    return rows


def mass_table_component(table):
    """html.Table of mass_table_rows."""
    rows = mass_table_rows(table)
    head = html.Thead(html.Tr([html.Th(c) for c in rows[0]]))
    body = html.Tbody([html.Tr([html.Td(c) for c in r]) for r in rows[1:]])
    return html.Table([head, body], className="conc-table")


def structure_table_rows(model, counts):
    """Rows of strings of the table of the structures of a model: a header and one row per
    peak and path. model is the second element of Explorer.model; counts maps a structure
    label to the number of galaxies of the selection inside its band."""
    rows = [list(STRUCTURE_HEADER)]
    for p in model["peaks"]:
        rows.append([p["label"], "peak", f"{p['width_mpc']:.2f}", f"{p['width_intrinsic_mpc']:.2f}",
                     f"{p['n_sigma']:.1f}", "-", f"{p['shift_mpc']:.2f}",
                     str(counts.get(p["label"], 0))])
    for p in model["paths"]:
        rows.append([p["label"], "path", f"{p['width_median_mpc']:.2f}",
                     f"{p['width_intrinsic_median_mpc']:.2f}",
                     f"{p['crest_n_sigma_mean']:.1f} / {p['crest_n_sigma_min']:.1f}",
                     f"{p['length_mpc']:.1f}", f"{p['shift_max_mpc']:.2f}",
                     str(counts.get(p["label"], 0))])
    return rows


def structure_table_component(model, counts):
    """html.Table of structure_table_rows."""
    rows = structure_table_rows(model, counts)
    head = html.Thead(html.Tr([html.Th(c) for c in rows[0]]))
    body = html.Tbody([html.Tr([html.Td(c) for c in r]) for r in rows[1:]])
    return html.Table([head, body], className="conc-table")


def _mass_section(explorer):
    """Panel with the summary of the map, the statistics table and the table of the structures
    of the model (filled by a callback)."""
    return html.Div([
        html.H3("Mass concentration"),
        html.Div(id="conc-summary", className="conc-summary"),
        html.Div(id="conc-caption", className="counts"),
        html.Div(html.Div(id="conc-table"), className="conc-table-wrap"),
        html.Div(html.Div(id="model-table"), className="conc-table-wrap"),
    ], className="analysis")


def _pick_controls(explorer):
    """Controls of the interactive sky map: pick modes, switch all, draw, run the model.

    Always in the tree (the callbacks read them); hidden and disabled without a
    'concentration' section.
    """
    off = explorer.conc_settings is None
    buttons = [html.Button(label, id=cid, n_clicks=0, disabled=off) for cid, label in (
        ("btn-pick-all", "All on"), ("btn-pick-none", "All off"),
        ("btn-clear-paths", "Clear drawn paths"))]
    run = [html.Button(label, id=cid, n_clicks=0, disabled=off) for cid, label in (
        ("btn-run-model", "Run model"), ("btn-clear-model", "Clear model"))]
    return html.Div([
        html.Div([dcc.Checklist(
            id="sky-pick", options=[{"label": " Select peaks", "value": "peaks"},
                                    {"label": " Select maximums", "value": "paths"}],
            value=[], inline=True, className="check", inputStyle={"marginRight": "3px"}),
            *buttons, html.Span(id="pick-info", className="counts")], className="row tight"),
        html.Div([*run, html.Span("model: not run", id="model-status", className="counts")],
                 className="row tight"),
        html.Div(PICK_HINT, className="hint"),
    ], id="pick-controls", className="pick-controls", style={"display": "none"} if off else {})


def build_layout(explorer):
    """Full page: header, sidebar (sample, environment, view, export), views, analysis."""
    st = default_state(explorer)
    n_spec, n_phot = explorer.counts(np.ones(len(explorer.df), dtype=bool))
    header = html.Div([
        html.H1("kn-neighbour environment explorer"),
        html.Div(f"{n_spec} spec-z + {n_phot} photo-z galaxies in "
                 f"{explorer.window[0]} < z < {explorer.window[1]} "
                 f"(table from {explorer.source}, config {str(explorer.config_sha256)[:8]})",
                 className="subtitle"),
    ], className="header")
    sidebar = html.Div([
        _sample_section(explorer, st), _concentration_section(explorer, st),
        _environment_section(explorer, st), _view_section(explorer, st),
        _export_section(explorer, st),
    ], className="sidebar")
    main = html.Div([
        html.Div([
            html.Div([html.H3("3D map"), _graph("v3d-fig", "520px", "knn_3d")], className="col"),
            html.Div([
                html.H3("Sky"),
                html.Div([html.Button("Clear selection", id="clear-selection", n_clicks=0),
                          html.Span(id="sel-info", className="counts")], className="row tight"),
                _pick_controls(explorer),
                _graph("sky-fig", "560px", "knn_sky"),
            ], className="col"),
        ], className="row"),
        _analysis_section(explorer, st),
    ] + ([_mass_section(explorer)] if explorer.conc_settings is not None else []),
        className="main")
    stores = [dcc.Store(id="store-sample"), dcc.Store(id="store-env"), dcc.Store(id="store-sel"),
              dcc.Store(id="store-struct", data=default_struct()),
              dcc.Store(id="store-model", data=st["model"])]
    return html.Div([header, html.Div([sidebar, main], className="body")] + stores,
                    className="page")
