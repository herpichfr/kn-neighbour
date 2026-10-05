"""Callbacks of the explorer.

The functions of this module hold the logic and take the Explorer and a state
dict (keys layout.STATE_KEYS), so they can be called without a browser;
register_callbacks wires them to the components of layout.build_layout.

Data flow: the sample controls make store-sample (filters and counts; the
mask is recomputed server side from the filters, memoised in the Explorer);
the environment controls make store-env (estimator parameters; the columns
come from the Explorer's cache); a box / lasso selection on the sky makes
store-sel (catalogue rows). The figure callbacks read these stores and send
only the plotted columns to the browser.
"""

import time

import numpy as np
import pandas as pd
from dash import Input, Output, State, ctx, dcc, no_update

from ..explorer import NONE_LABEL, _as_float
from .figures import (counts_text, empty_figure, graph_config, hist_figure, scatter3d_figure, sky_figure,
                      trend_figure, trend_matplotlib)
from .layout import (ENV_KEYS, KNN3D_NOTE, SAMPLE_KEYS, STATE_CONTROLS, STATE_KEYS, default_state,
                     logm_bounds)

CSV_BASE_COLUMNS = ("id", "row", "ra", "dec", "z_best", "z_type", "logmstar", "env", "env_lo",
                    "env_hi")


def make_state(explorer, **overrides):
    """default_state with some keys replaced (unknown keys raise KeyError)."""
    state = default_state(explorer)
    for key, value in overrides.items():
        if key not in state:
            raise KeyError(f"unknown state key {key!r}")
        state[key] = value
    return state


def filters_from_state(explorer, state):
    """The explorer.select filters of the sample controls.

    A control left at its neutral value adds no filter: the full log M*
    range, no M_r cut, empty multi-selects. No redshift type checked selects
    nothing.
    """
    f = {"z_range": list(state["z_range"]),
         "z_types": list(state["z_types"]) or ["__none__"]}
    if state.get("p_min"):
        f["p_in_window_min"] = float(state["p_min"])
    if state.get("odds_min") and explorer.has_column("z_phot_odds"):
        f["odds_min"] = float(state["odds_min"])
    lo, hi = logm_bounds(explorer)
    if explorer.has_column("logmstar") and list(state["logm_range"]) != [lo, hi]:
        f["logm_range"] = list(state["logm_range"])
    if state.get("mr_max") is not None and explorer.has_column("M_r"):
        f["mr_max"] = float(state["mr_max"])
    if state.get("member") and explorer.has_column("flag_member"):
        f["flag_member"] = [float(v) for v in state["member"]]
    if state.get("subs") and explorer.has_column("substructure"):
        f["substructure"] = list(state["subs"])
    if state.get("groups") and explorer.has_column("grp_photz_id_group_final"):
        f["group"] = [g if g == NONE_LABEL else int(g) for g in state["groups"]]
    if state.get("quenched", "any") != "any" and explorer.has_column("quenched"):
        f["quenched"] = state["quenched"]
    if state.get("morph"):
        f["morphology"] = True
    return f


def env_params(state):
    """Arguments of Explorer.env / environment from the environment controls."""
    return {"name": state["estimator"], "k": int(state["k"] or 10), "weight": state["weight"],
            "photoz_mode": state["pzmode"], "dv": state["dv"], "n_sigma": state["n_sigma"]}


def env_note(estimator):
    """Warning shown under the estimator: non-empty only for knn3d (no photo-z values)."""
    return KNN3D_NOTE if estimator == "knn3d" else ""


def sample_info(explorer, state):
    """Content of store-sample: the filters and the counts of the selected sample."""
    filters = filters_from_state(explorer, state)
    mask = explorer.select(filters)
    n_spec, n_phot = explorer.counts(mask)
    return {"filters": filters, "n_spec": n_spec, "n_phot": n_phot,
            "text": f"Selected: {n_spec} spec-z, {n_phot} photo-z ({n_spec + n_phot} of "
                    f"{len(mask)})"}


def env_info(explorer, state):
    """Content of store-env: the parameters; computes (and caches) the environment."""
    params = env_params(state)
    t0 = time.perf_counter()
    frame = explorer.env(**params)
    seconds = time.perf_counter() - t0
    valid = np.isfinite(frame["env"].to_numpy())
    n_spec, n_phot = explorer.counts(valid)
    params.update(n_valid=int(valid.sum()), seconds=round(seconds, 3),
                  text=f"log(1+delta) defined for {n_spec} spec-z and {n_phot} photo-z "
                       f"galaxies ({seconds:.2f} s)")
    return params


def _env_args(info):
    return {k: info[k] for k in ("name", "k", "weight", "photoz_mode", "dv", "n_sigma")}


def sample_mask(explorer, sample):
    """Boolean row mask of the sample described by store-sample (all rows if empty)."""
    return explorer.select(sample["filters"] if sample else None)


def selection_rows(selected_data):
    """Catalogue rows (customdata[0]) of the points of a Graph's selectedData."""
    if not selected_data or not selected_data.get("points"):
        return None
    rows = []
    for p in selected_data["points"]:
        cd = p.get("customdata")
        if cd is not None and len(cd):
            rows.append(int(round(cd[0])))
    return sorted(set(rows)) if rows else None


def selection_mask(explorer, sample, sel_rows):
    """Sample mask restricted to the sky selection (if any)."""
    mask = sample_mask(explorer, sample)
    if sel_rows:
        keep = np.zeros(len(mask), dtype=bool)
        keep[np.asarray(sel_rows, dtype=int)] = True
        mask &= keep
    return mask


def column_values(explorer, name, env_frame):
    """float array (one entry per row) of a catalogue column or the live 'env'."""
    if name == "env":
        return env_frame["env"].to_numpy(float)
    return _as_float(explorer.df[name])


def _label(explorer, name, info):
    if name != "env":
        return name
    suffix = " mass-weighted" if info["weight"] == "mass" else ""
    return f"log(1+δ) {info['name']} k={info['k']}{suffix}"


def build_view(explorer, state, env, sample, sel_rows=None):
    """View frame (figures module docstring) of the sample rows."""
    info = env
    frame = explorer.env(**_env_args(info))
    mask = sample_mask(explorer, sample)
    idx = np.flatnonzero(mask)
    df = explorer.df
    nan = np.full(len(df), np.nan)
    logm = df["logmstar"].to_numpy(float) if explorer.has_column("logmstar") else nan
    d = pd.DataFrame({
        "row": idx, "id": explorer.ids[idx], "ra": df["ra"].to_numpy(float)[idx],
        "dec": df["dec"].to_numpy(float)[idx], "z": df["z_best"].to_numpy(float)[idx],
        "z_type": explorer.z_type[idx], "logm": logm[idx], "env": frame["env"].to_numpy()[idx],
        "cx": df["cx"].to_numpy(float)[idx], "cy": df["cy"].to_numpy(float)[idx],
        "cz": df["cz"].to_numpy(float)[idx],
        "color": column_values(explorer, state["color"], frame)[idx]})
    if state.get("size"):
        d["size"] = column_values(explorer, state["size"], frame)[idx]
    d["selected"] = np.isin(idx, np.asarray(sel_rows, dtype=int)) if sel_rows else False
    return d


def clip_of(state):
    """Colour-range percentile clip (lo, hi) from the view state."""
    lo, hi = state["clip"]
    return (float(lo), float(hi))


def sky_callback(explorer, state, env, sample, clear_clicks=0):
    """Sky (RA, Dec) figure for the current sample, environment and colour settings."""
    d = build_view(explorer, state, env, sample)
    return sky_figure(d, _label(explorer, state["color"], env), clip_of(state),
                      density=bool(state["density"]), uirevision=f"sky{clear_clicks}")


def scatter3d_callback(explorer, state, env, sample, sel_rows=None):
    """3D figure for the current sample, highlighting the sky-view selection rows."""
    d = build_view(explorer, state, env, sample, sel_rows)
    return scatter3d_figure(d, state["coords"], _label(explorer, state["color"], env),
                            clip_of(state))


def parse_mass_bins(text):
    """Edges from '8, 9, 10, 11' (comma or space separated); None for empty text.

    Raises ValueError for a value that is not a number or fewer than two edges.
    """
    if text is None or not str(text).strip():
        return None
    edges = [float(t) for t in str(text).replace(",", " ").split()]
    if len(edges) < 2:
        raise ValueError("give at least two bin edges, or leave the field empty")
    return sorted(edges)


def analysis_trend(explorer, state, env, sample, sel_rows=None):
    """(trend table, x label, y label) of the analysis panel."""
    frame = explorer.env(**_env_args(env))
    mask = selection_mask(explorer, sample, sel_rows)
    bins = parse_mass_bins(state["mass_bins"]) if explorer.has_column("logmstar") else None
    x = column_values(explorer, state["x"], frame)
    y_name = state["y"]
    y = frame["env"].to_numpy(float) if y_name == "env" else explorer.df[y_name]
    trend = explorer.trend(x, y, mask, mass_bins=bins, nbins=int(state["nbins"] or 8))
    return trend, _label(explorer, state["x"], env), _label(explorer, y_name, env)


def analysis_callback(explorer, state, env, sample, sel_rows=None):
    """(trend figure, histogram figure, N-per-bin text)."""
    try:
        trend, xl, yl = analysis_trend(explorer, state, env, sample, sel_rows)
    except ValueError as err:
        msg = f"Invalid input: {err}"
        return empty_figure(msg), empty_figure(msg), msg
    frame = explorer.env(**_env_args(env))
    mask = selection_mask(explorer, sample, sel_rows)
    values = column_values(explorer, state["x"], frame)
    hist = hist_figure(values[mask], values, xl)
    return trend_figure(trend, xl, yl), hist, counts_text(trend)


def export_csv_frame(explorer, state, env, sample, sel_rows=None):
    """The selection (sample and sky selection) with the visible columns."""
    frame = explorer.env(**_env_args(env))
    mask = selection_mask(explorer, sample, sel_rows)
    idx = np.flatnonzero(mask)
    df = explorer.df
    out = pd.DataFrame({"id": explorer.ids[idx], "row": idx})
    for col in ("ra", "dec", "z_best", "z_type", "logmstar"):
        out[col] = df[col].to_numpy()[idx] if col in df.columns else np.nan
    out["env"] = frame["env"].to_numpy()[idx]
    out["env_lo"] = frame["env_lo"].to_numpy()[idx]
    out["env_hi"] = frame["env_hi"].to_numpy()[idx]
    for name in (state.get("color"), state.get("size"), state.get("x"), state.get("y")):
        if name and name not in out.columns:
            out[name] = (frame["env"].to_numpy() if name == "env" else df[name].to_numpy())[idx]
    return out


def publication_pdf(explorer, state, env, sample, sel_rows=None):
    """PDF bytes of the analysis plot, rendered with matplotlib."""
    trend, xl, yl = analysis_trend(explorer, state, env, sample, sel_rows)
    return trend_matplotlib(trend, xl, yl, title=f"{env['name']} k={env['k']}")


def state_from_values(values):
    """State dict from the values of the STATE_CONTROLS components, in order."""
    return dict(zip(STATE_KEYS, values))


def restore_state(explorer, record):
    """Values for every STATE_CONTROLS component from a loaded snapshot.

    Keys missing from the snapshot keep their default. Returns (values, message).
    """
    state = default_state(explorer)
    state.update({k: v for k, v in record["state"].items() if k in state})
    msg = "Snapshot loaded."
    if not record.get("matches_config"):
        msg += " Warning: it was made with a different config (sha256 differs)."
    return [state[k] for k in STATE_KEYS], msg


_STATE_INPUTS = [State(cid, prop) for _, cid, prop in STATE_CONTROLS]


def _pick(keys):
    """Inputs for a subset of state keys: (list of Input, list of keys)."""
    comps = {k: (cid, prop) for k, cid, prop in STATE_CONTROLS}
    return [Input(*comps[k]) for k in keys]


def register_callbacks(app, explorer):
    """Attach every callback of the explorer to a Dash app."""
    defaults = default_state(explorer)

    def partial_state(keys, values):
        state = dict(defaults)
        state.update(dict(zip(keys, values)))
        return state

    @app.callback(Output("env-dv", "disabled"), Output("env-nsigma", "disabled"),
                  Output("env-pzmode", "disabled"), Output("env-note", "children"),
                  Input("env-estimator", "value"))
    def _env_controls(estimator):
        return estimator != "sigma2d", estimator != "sigma2d", estimator == "knn3d", \
            env_note(estimator)

    sample_keys = list(SAMPLE_KEYS)

    @app.callback(Output("store-sample", "data"), Output("sample-counts", "children"),
                  *_pick(sample_keys))
    def _sample(*values):
        info = sample_info(explorer, partial_state(sample_keys, values))
        return info, info["text"]

    env_keys = list(ENV_KEYS)

    @app.callback(Output("store-env", "data"), Output("env-status", "children"),
                  *_pick(env_keys))
    def _env(*values):
        try:
            info = env_info(explorer, partial_state(env_keys, values))
        except (ValueError, KeyError, TypeError) as err:
            return no_update, f"Cannot compute the environment: {err}"
        return info, info["text"]

    @app.callback(Output("store-sel", "data"), Output("sel-info", "children"),
                  Input("sky-fig", "selectedData"), Input("clear-selection", "n_clicks"),
                  prevent_initial_call=True)
    def _selection(selected, _clicks):
        if ctx.triggered_id == "clear-selection":
            return None, ""
        rows = selection_rows(selected)
        return rows, (f"{len(rows)} galaxies selected on the sky" if rows else "")

    sky_keys = ["color", "clip", "size", "density"]

    @app.callback(Output("sky-fig", "figure"), Input("store-sample", "data"),
                  Input("store-env", "data"), Input("clear-selection", "n_clicks"),
                  *_pick(sky_keys))
    def _sky(sample, env, clicks, *values):
        if not sample or not env:
            return no_update
        return sky_callback(explorer, partial_state(sky_keys, values), env, sample, clicks)

    v3d_keys = ["coords", "color", "size", "clip"]

    @app.callback(Output("v3d-fig", "figure"), Input("store-sample", "data"),
                  Input("store-env", "data"), Input("store-sel", "data"), *_pick(v3d_keys))
    def _v3d(sample, env, sel, *values):
        if not sample or not env:
            return no_update
        return scatter3d_callback(explorer, partial_state(v3d_keys, values), env, sample, sel)

    an_keys = ["y", "x", "mass_bins", "nbins"]

    @app.callback(Output("an-fig", "figure"), Output("an-hist", "figure"),
                  Output("an-counts", "children"), Input("store-sample", "data"),
                  Input("store-env", "data"), Input("store-sel", "data"), *_pick(an_keys))
    def _analysis(sample, env, sel, *values):
        if not sample or not env:
            return no_update, no_update, no_update
        return analysis_callback(explorer, partial_state(an_keys, values), env, sample, sel)

    graph_ids = ("v3d-fig", "sky-fig", "an-fig", "an-hist")
    graph_names = {"v3d-fig": "knn_3d", "sky-fig": "knn_sky", "an-fig": "knn_trend",
                   "an-hist": "knn_hist"}

    @app.callback(*[Output(g, "config") for g in graph_ids], Input("export-imgfmt", "value"))
    def _graph_config(fmt):
        return tuple(graph_config(fmt, graph_names[g]) for g in graph_ids)

    export_keys = ["color", "size", "x", "y", "mass_bins", "nbins"]

    @app.callback(Output("dl-csv", "data"), Input("btn-csv", "n_clicks"),
                  State("store-sample", "data"), State("store-env", "data"),
                  State("store-sel", "data"), *[State(*c[1:]) for c in STATE_CONTROLS
                                                if c[0] in export_keys],
                  prevent_initial_call=True)
    def _csv(_clicks, sample, env, sel, *values):
        if not sample or not env:
            return no_update
        frame = export_csv_frame(explorer, partial_state(
            [c[0] for c in STATE_CONTROLS if c[0] in export_keys], values), env, sample, sel)
        return dcc.send_data_frame(frame.to_csv, "knn_selection.csv", index=False)

    @app.callback(Output("dl-pdf", "data"), Input("btn-pdf", "n_clicks"),
                  State("store-sample", "data"), State("store-env", "data"),
                  State("store-sel", "data"), *[State(*c[1:]) for c in STATE_CONTROLS
                                                if c[0] in export_keys],
                  prevent_initial_call=True)
    def _pdf(_clicks, sample, env, sel, *values):
        if not sample or not env:
            return no_update
        state = partial_state([c[0] for c in STATE_CONTROLS if c[0] in export_keys], values)
        return dcc.send_bytes(publication_pdf(explorer, state, env, sample, sel),
                              "knn_trend.pdf")

    @app.callback(Output("dl-snapshot", "data"), Input("btn-snapshot", "n_clicks"),
                  *_STATE_INPUTS, prevent_initial_call=True)
    def _snapshot(_clicks, *values):
        return dcc.send_string(explorer.snapshot(state_from_values(values)),
                               "knn_snapshot.yaml")

    @app.callback(*[Output(cid, prop, allow_duplicate=True) for _, cid, prop in STATE_CONTROLS],
                  Output("snapshot-status", "children"), Input("up-snapshot", "contents"),
                  prevent_initial_call=True)
    def _restore(contents):
        import base64
        n = len(STATE_CONTROLS)
        if not contents:
            return (*([no_update] * n), no_update)
        try:
            text = base64.b64decode(contents.split(",", 1)[1]).decode("utf-8")
            values, msg = restore_state(explorer, explorer.load_snapshot(text))
        except (ValueError, IndexError, UnicodeDecodeError) as err:
            return (*([no_update] * n), f"Cannot load the snapshot: {err}")
        return (*values, msg)
