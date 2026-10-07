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

The sky map is interactive (with a 'concentration' section): clicks and drawn paths
update store-struct (struct_update: which peaks and ridge segments are off, the paths
drawn by the user); "Run model" turns the structures that are on into a model spec
(run_spec), kept in store-model, which store-sample carries (sample['model']) so that
every view follows it (the inside / outside filter, the statistics table, the overlay).
"""

import copy
import hashlib
import json
import re
import time

import numpy as np
import pandas as pd
from dash import Input, Output, State, ctx, dcc, html, no_update

from ..explorer import NONE_LABEL, _as_float
from ..concentration import (ConcentrationError, contour_table, describe, describe_model, hull_sky,
                             peak_table, ridge_table, sky_grid)
from .figures import (counts_text, empty_figure, graph_config, hist_figure, scatter3d_figure, sky_figure,
                      trend_figure, trend_matplotlib)
from .layout import (ENV_KEYS, KNN3D_NOTE, SAMPLE_KEYS, STATE_CONTROLS, STATE_KEYS, default_state,
                     default_struct, logm_bounds, mass_table_component, structure_table_component)

CSV_BASE_COLUMNS = ("id", "row", "ra", "dec", "z_best", "z_type", "logmstar", "env", "env_lo",
                    "env_hi")
# filters that do not change the map (the galaxies it is made of): the region of the 1 sigma
# contour or band and what that region needs
REGION_FILTERS = ("conc_region", "conc_params", "conc_model")


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
    if state.get("concregion", "any") != "any" and explorer.conc_settings is not None:
        f["conc_region"] = state["concregion"]
        f["conc_params"] = conc_params(state)
    return f


def conc_params(state):
    """Arguments of Explorer.concentration from the concentration controls (None = config / Scott)."""
    return {"weight": state.get("concw") or None,
            "bandwidth_mpc": state.get("conch") or None,
            "contour_step_sigma": state.get("concstep") or None,
            "min_prominence_sigma": state.get("concprom")}


def env_params(state):
    """Arguments of Explorer.env / environment from the environment controls."""
    return {"name": state["estimator"], "k": int(state["k"] or 10), "weight": state["weight"],
            "photoz_mode": state["pzmode"], "dv": state["dv"], "n_sigma": state["n_sigma"]}


def env_note(estimator):
    """Warning shown under the estimator: non-empty only for knn3d (no photo-z values)."""
    return KNN3D_NOTE if estimator == "knn3d" else ""


def map_key(sample):
    """12 hex digits that identify the map of a store-sample sample (its filters without the
    region ones, and the concentration controls): the ids of peaks and ridges belong to a map."""
    filters = {k: v for k, v in sample["filters"].items() if k not in REGION_FILTERS}
    text = json.dumps({"filters": filters, "conc": sample.get("conc")}, sort_keys=True,
                      default=str)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def model_spec(model):
    """The spec ({'peaks', 'paths'}) of a stored model (the run spec has also time and map_key)."""
    model = model if isinstance(model, dict) else {}
    return {"peaks": list(model.get("peaks") or []), "paths": list(model.get("paths") or [])}


def _select(explorer, filters):
    """(filters, mask, note) of explorer.select(filters); the region filters that fail are dropped.

    A model that fails (a malformed spec) is dropped alone, with a note; a map that fails (a
    sample that is too small) drops the whole 1 sigma region filter.
    """
    try:
        return filters, explorer.select(filters), ""
    except ConcentrationError as err:
        if "conc_model" in filters:
            plain = {k: v for k, v in filters.items() if k != "conc_model"}
            try:
                return plain, explorer.select(plain), f" (model region filter off: {err})"
            except ConcentrationError:
                pass
        plain = {k: v for k, v in filters.items() if k not in REGION_FILTERS}
        return plain, explorer.select(plain), f" (1 sigma region filter off: {err})"


def sample_info(explorer, state):
    """Content of store-sample: the filters and the counts of the selected sample.

    With a model in the state (state['model'], the run spec) and an inside / outside region,
    the region is that of the model (filter conc_model). 'model' is the run spec that applies,
    None if there is none or it failed; 'map_key' identifies the map (map_key).
    """
    filters = filters_from_state(explorer, state)
    model = state.get("model") if isinstance(state.get("model"), dict) else None
    spec = model_spec(model)
    asked = model is not None and "conc_region" in filters and bool(spec["peaks"] or spec["paths"])
    if asked:
        filters["conc_model"] = spec
    filters, mask, note = _select(explorer, filters)
    if asked and "conc_model" not in filters:
        model = None
    n_spec, n_phot = explorer.counts(mask)
    info = {"filters": filters, "n_spec": n_spec, "n_phot": n_phot,
            "conc": conc_params(state) if explorer.conc_settings is not None else None,
            "model": model if explorer.conc_settings is not None else None,
            "text": f"Selected: {n_spec} spec-z, {n_phot} photo-z ({n_spec + n_phot} of "
                    f"{len(mask)}){note}"}
    info["map_key"] = map_key(info)
    return info


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


def concentration_result(explorer, sample):
    """(columns, result) of the map of a store-sample sample, or None without 'concentration'.

    The map is made of the galaxies passing every sample filter except the
    inside / outside one (and its model); its settings are the concentration controls
    (sample['conc']).
    Raises ConcentrationError for a sample that is too small.
    """
    if explorer.conc_settings is None or not sample:
        return None
    return explorer.concentration(_map_mask(explorer, sample), **(sample.get("conc") or {}))


def _map_mask(explorer, sample):
    """Mask of the galaxies the map of a sample is made of (every filter but the region ones)."""
    filters = {k: v for k, v in sample["filters"].items() if k not in REGION_FILTERS}
    return explorer.select(filters)


def model_result(explorer, sample):
    """(columns, model) of the model of a store-sample sample (sample['model']), or None.

    The model is fitted on the map of the sample (Explorer.model, cached); the columns cover
    every catalogue row. Raises ConcentrationError (a ValueError) for a map that is too small or
    a spec that is malformed.
    """
    if explorer.conc_settings is None or not sample or not sample.get("model"):
        return None
    return explorer.model(_map_mask(explorer, sample), model_spec(sample["model"]),
                          **(sample.get("conc") or {}))


def _line_trace(polylines):
    """(x, y) lists of several polylines joined with None separators.

    A polyline is cut at every point that is None or not finite (the model puts NaN where a
    band edge or a circle leaves the survey area): the lists hold floats and None only, never
    NaN, which the browser could not read.
    """
    xs, ys = [], []
    for x, y in polylines:
        run = False
        for a, b in zip(x, y):
            if a is None or b is None or not (np.isfinite(a) and np.isfinite(b)):
                if run:
                    xs.append(None)
                    ys.append(None)
                run = False
                continue
            xs.append(float(a))
            ys.append(float(b))
            run = True
        if run:
            xs.append(None)
            ys.append(None)
    return xs, ys


def _polylines(items):
    """Lists x, y, customdata, text of polylines joined with None separators.

    items: [(ra, dec, [kind, id], text)]; every vertex carries the customdata and the text of
    its polyline, the separators a [None, None] row and no text.
    """
    out = {"x": [], "y": [], "customdata": [], "text": []}
    for ra, dec, cd, text in items:
        n = len(ra)
        out["x"] += list(ra) + [None]
        out["y"] += list(dec) + [None]
        out["customdata"] += [list(cd)] * n + [[None, None]]
        out["text"] += [text] * n + [""]
    return out


def _off_ids(struct, sample, key):
    """Ids of the structures switched off ('off_peaks' / 'off_ridges') on the map of sample."""
    if not struct or struct.get("map_key") != (sample or {}).get("map_key"):
        return set()
    return {int(i) for i in struct.get(key) or []}


def _strong(ridge):
    """True for a ridge segment whose saddle is at least 1 sigma (or has none)."""
    s = ridge["saddle_n_sigma"]
    return s is None or not np.isfinite(s) or s >= 1.0


def _model_overlay(model):
    """Polylines (lists of degrees, None separated) of the fitted paths, band edges and peak circles."""
    path_x, path_y = _line_trace([(q["ra"], q["dec"]) for q in model["paths"]])
    band_x, band_y = _line_trace(
        [(q[f"{side}_ra"], q[f"{side}_dec"]) for q in model["paths"] for side in ("left", "right")])
    circle_x, circle_y = _line_trace([(q["circle_ra"], q["circle_dec"]) for q in model["peaks"]])
    return {"path_x": path_x, "path_y": path_y, "band_x": band_x, "band_y": band_y,
            "circle_x": circle_x, "circle_y": circle_y}


def concentration_overlay(explorer, sample, struct=None):
    """Map, contours, survey border, ridges, drawn paths, model and peaks for the sky figure, or None.

    struct is store-struct: ridges and peaks it switches off (on the map it belongs to) and
    the drawn paths; None = everything on, nothing drawn. The model overlay is that of
    sample['model'] (model_result); a model that fails is left out.
    """
    try:
        found = concentration_result(explorer, sample)
    except ConcentrationError:
        return None
    if found is None:
        return None
    result = found[1]
    ra, dec, z = sky_grid(result)
    contours = contour_table(result)
    contour1_x, contour1_y = _line_trace([(ln["ra"], ln["dec"]) for c in contours[:1]
                                          for ln in c["lines"]])
    contour_x, contour_y = _line_trace([(ln["ra"], ln["dec"]) for c in contours[1:]
                                        for ln in c["lines"]])
    off_ridges, off_peaks = _off_ids(struct, sample, "off_ridges"), _off_ids(struct, sample,
                                                                             "off_peaks")
    styled = {"strong": [], "open": [], "low": [], "off": []}
    for r in ridge_table(result):
        on = r["id"] not in off_ridges
        s = r["saddle_n_sigma"]
        text = (f"ridge {r['id']}, saddle {s:.1f}σ, {r['length_mpc']:.1f} Mpc "
                f"({'on' if on else 'off'})" if s is not None and np.isfinite(s) else
                f"ridge {r['id']}, {r['length_mpc']:.1f} Mpc ({'on' if on else 'off'})")
        key = "off" if not on else ("low" if not _strong(r) else ("open" if r["open"] else "strong"))
        styled[key].append((r["ra"], r["dec"], ["ridge", r["id"]], text))
    drawn = {"on": [], "off": []}
    for q in (struct or {}).get("paths") or []:
        drawn["on" if q["on"] else "off"].append(
            (q["ra"], q["dec"], ["path", q["id"]],
             f"drawn path {q['id']} ({'on' if q['on'] else 'off'})"))
    try:
        fitted = model_result(explorer, sample)
    except ValueError:
        fitted = None
    peaks = peak_table(result)
    hull_ra, hull_dec = hull_sky(result)
    return {"map_ra": ra, "map_dec": dec, "map_z": z, "hull_x": hull_ra.tolist(),
            "hull_y": hull_dec.tolist(), "contour_x": contour_x, "contour_y": contour_y,
            "contour1_x": contour1_x, "contour1_y": contour1_y,
            "ridge_groups": {k: _polylines(v) for k, v in styled.items()},
            "drawn": {k: _polylines(v) for k, v in drawn.items()},
            "model": None if fitted is None else _model_overlay(fitted[1]),
            "peak_id": [p["id"] for p in peaks], "peak_ra": [p["ra"] for p in peaks],
            "peak_dec": [p["dec"] for p in peaks], "peak_open": [p["open"] for p in peaks],
            "peak_on": [p["id"] not in off_peaks for p in peaks],
            "peak_text": [f"peak {p['id']}, {p['n_sigma']:.1f}σ, prominence "
                          f"{p['prominence_sigma']:.1f}σ ({'off' if p['id'] in off_peaks else 'on'})"
                          for p in peaks]}


_SVG_TOKEN = re.compile(r"[A-Za-z]|[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def parse_svg_path(text):
    """(ra list, dec list) of the vertices of a plotly shape path 'M x,y L x,y ...'.

    Absolute M and L commands (also written 'Mx,yLx,y' or with several pairs after one
    command) and Z (ignored: the path stays open). Raises ValueError for any other command
    or a malformed path.
    """
    tokens = _SVG_TOKEN.findall(text or "")
    ra, dec, command, i = [], [], None, 0
    while i < len(tokens):
        token = tokens[i]
        if token.isalpha():
            if token not in ("M", "L", "Z", "z"):
                raise ValueError(f"unsupported path command {token!r}")
            command = token.upper()
            i += 1
            continue
        if command not in ("M", "L") or i + 1 >= len(tokens) or tokens[i + 1].isalpha():
            raise ValueError(f"malformed path {text!r}")
        ra.append(float(token))
        dec.append(float(tokens[i + 1]))
        i += 2
    return ra, dec


STRUCT_TRIGGERS = {"sky-fig.clickData": "click", "sky-fig.clickAnnotationData": "annotation",
                   "sky-fig.relayoutData": "relayout", "btn-pick-all.n_clicks": "all",
                   "btn-pick-none.n_clicks": "none", "btn-clear-paths.n_clicks": "clear_paths",
                   "store-sample.data": "sample"}


def _click_target(click):
    """(kind, id) of the structure vertex of a Graph's clickData, or None (galaxies carry numbers)."""
    points = (click or {}).get("points") or []
    cd = points[0].get("customdata") if points and isinstance(points[0], dict) else None
    if isinstance(cd, (list, tuple)) and len(cd) == 2 and cd[0] in ("peak", "ridge", "path"):
        try:
            return cd[0], int(round(float(cd[1])))
        except (TypeError, ValueError):
            return None
    return None


def _annotation_target(data):
    """('peak', id) of a Graph's clickAnnotationData whose annotation is named 'peak <id>'."""
    data = data if isinstance(data, dict) else {}
    note = data.get("fullAnnotation") or data.get("annotation") or data
    found = re.fullmatch(r"peak (\d+)", str(note.get("name") or "")) if isinstance(note, dict) else None
    return ("peak", int(found.group(1))) if found else None


def _toggled(ids, ident):
    return sorted(set(ids) ^ {ident})


def _map_ids(explorer, sample):
    """(peak ids, ridge ids) of the map of a sample; empty lists without a map."""
    try:
        found = concentration_result(explorer, sample)
    except ConcentrationError:
        return [], []
    if found is None:
        return [], []
    return [p["id"] for p in found[1]["peaks"]], [r["id"] for r in found[1]["ridges"]]


def struct_update(explorer, struct, sample, pick, trigger, click=None, relayout=None):
    """New store-struct after an event of the sky map (the input struct is not modified).

    trigger: 'click' (click = the Graph's clickData), 'annotation' (click = clickAnnotationData
    of a peak star), 'relayout' (relayout = relayoutData: new drawn shapes), 'all', 'none',
    'clear_paths' or 'sample' (nothing but the map check). pick: the pick modes that are on
    ('peaks', 'paths'). 'peaks' is the click mode: a click toggles a peak, a ridge or a drawn
    path ('paths' also toggles ridges and drawn paths, but plotly sends no click while it is
    drawing); a shape drawn while 'paths' is on becomes a path
    (once: shapes already stored, by their SVG path string, are ignored). A change of map
    (sample['map_key']) switches every peak and ridge on again (the ids belong to the old map);
    the drawn paths stay. 'rev' counts the drawings read, so that the figure drops the shapes.
    """
    struct = copy.deepcopy(struct) if struct else default_struct()
    if not sample:
        return struct
    pick = set(pick or ())
    key = sample.get("map_key")
    if struct.get("map_key") != key:
        struct.update(map_key=key, off_peaks=[], off_ridges=[])
    if trigger in ("click", "annotation"):
        target = _click_target(click) if trigger == "click" else _annotation_target(click)
        if target is None:
            return struct
        kind, ident = target
        if kind == "peak" and "peaks" in pick:
            struct["off_peaks"] = _toggled(struct["off_peaks"], ident)
        elif kind == "ridge" and pick & {"peaks", "paths"}:
            struct["off_ridges"] = _toggled(struct["off_ridges"], ident)
        elif kind == "path" and pick & {"peaks", "paths"}:
            for path in struct["paths"]:
                if path["id"] == ident:
                    path["on"] = not path["on"]
    elif trigger == "relayout" and "paths" in pick:
        shapes = (relayout or {}).get("shapes")
        if isinstance(shapes, list):
            known = {q["src"] for q in struct["paths"]}
            for shape in shapes:
                src = shape.get("path") if isinstance(shape, dict) else None
                if not isinstance(src, str) or src in known:
                    continue
                known.add(src)
                try:
                    ra, dec = parse_svg_path(src)
                except ValueError:
                    continue
                if len(set(zip(ra, dec))) < 2:
                    continue
                struct["paths"].append({"id": struct["next_id"], "ra": ra, "dec": dec, "on": True,
                                        "src": src})
                struct["next_id"] += 1
            struct["rev"] = struct.get("rev", 0) + 1
    elif trigger == "all":
        struct.update(off_peaks=[], off_ridges=[])
        for path in struct["paths"]:
            path["on"] = True
    elif trigger == "none":
        peak_ids, ridge_ids = _map_ids(explorer, sample)
        struct.update(off_peaks=sorted(peak_ids), off_ridges=sorted(ridge_ids))
        for path in struct["paths"]:
            path["on"] = False
    elif trigger == "clear_paths":
        struct["paths"] = []
    return struct


def run_spec(explorer, sample, struct):
    """Model spec (Explorer.model) of the structures that are on, with 'time' and 'map_key'.

    Peaks (ra, dec) and ridge segments (label 'ridge <id>') of the map of the sample, minus
    the ones struct switches off, and the drawn paths that are on (label 'drawn <id>').
    Raises ConcentrationError when there is no map or nothing is selected.
    """
    found = concentration_result(explorer, sample)
    if found is None:
        raise ConcentrationError("the config has no 'concentration' section")
    off_peaks, off_ridges = _off_ids(struct, sample, "off_peaks"), _off_ids(struct, sample,
                                                                            "off_ridges")
    result = found[1]
    paths = [{"label": f"ridge {r['id']}", "ra": r["ra"], "dec": r["dec"]}
             for r in ridge_table(result) if r["id"] not in off_ridges]
    paths += [{"label": f"drawn {q['id']}", "ra": list(q["ra"]), "dec": list(q["dec"])}
              for q in (struct or {}).get("paths") or [] if q["on"]]
    spec = {"peaks": [[p["ra"], p["dec"]] for p in peak_table(result) if p["id"] not in off_peaks],
            "paths": paths, "time": time.strftime("%H:%M:%S"), "map_key": sample.get("map_key")}
    if not spec["peaks"] and not spec["paths"]:
        raise ConcentrationError("select at least one peak or path")
    return spec


def pick_counts(explorer, sample, struct):
    """Text of pick-info: 'peaks on 3/4, ridges on 5/6, drawn 1'."""
    paths = (struct or {}).get("paths") or []
    n_on = sum(1 for q in paths if q["on"])
    drawn = f"drawn {n_on}" + (f"/{len(paths)}" if n_on != len(paths) else "")
    peak_ids, ridge_ids = _map_ids(explorer, sample)
    off_peaks = _off_ids(struct, sample, "off_peaks") & set(peak_ids)
    off_ridges = _off_ids(struct, sample, "off_ridges") & set(ridge_ids)
    return (f"peaks on {len(peak_ids) - len(off_peaks)}/{len(peak_ids)}, ridges on "
            f"{len(ridge_ids) - len(off_ridges)}/{len(ridge_ids)}, {drawn}")


def model_status(model, skipped=()):
    """Text of model-status for a stored run spec (None = not run)."""
    if not isinstance(model, dict):
        return "model: not run"
    text = (f"model: {len(model.get('peaks') or [])} peaks, {len(model.get('paths') or [])} paths "
            f"(run {model.get('time', '?')})")
    if skipped:
        text += "; skipped: " + ", ".join(f"{s['label']} ({s['reason']})" for s in skipped)
    return text


def run_model(explorer, sample, struct, action="run"):
    """(model, status text) of the Run model / Clear model buttons.

    'run': the run spec of run_spec, after fitting it once (Explorer.model, cached) so that a
    failure shows here; on failure the model is no_update and the text says why. 'clear': None.
    """
    if action == "clear":
        return None, model_status(None)
    if not sample:
        return no_update, "model: no sample yet"
    try:
        spec = run_spec(explorer, sample, struct or default_struct())
        _, model = model_result(explorer, {**sample, "model": spec})
    except ValueError as err:
        return no_update, f"model: cannot run ({err})"
    return spec, model_status(spec, model["skipped"])


def sky_callback(explorer, state, env, sample, clear_clicks=0, struct=None, pick=()):
    """Sky (RA, Dec) figure for the current sample, environment and colour settings.

    struct (store-struct) and pick (the pick modes) make the sky map interactive; the model
    comes with the sample.
    """
    d = build_view(explorer, state, env, sample)
    return sky_figure(d, _label(explorer, state["color"], env), clip_of(state),
                      density=bool(state["density"]), uirevision=f"sky{clear_clicks}",
                      overlay=concentration_overlay(explorer, sample, struct), pick=pick,
                      editrevision=f"sky{clear_clicks}-{(struct or {}).get('rev', 0)}")


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


def concentration_callback(explorer, env, sample, sel_rows=None):
    """(summary lines, table component, caption) of the concentration panel.

    The map is that of the sample (concentration_result); the table (Explorer.mass_table)
    covers the sample galaxies, restricted to the sky selection, with the
    environment of the current estimator. With a model (sample['model']) inside / outside is
    the 1 sigma band of the model instead of the 1 sigma contour of the map, and the summary
    adds the lines of describe_model.
    """
    try:
        columns, result = concentration_result(explorer, sample)
    except ConcentrationError as err:
        return [f"No map: {err}"], html.Div(), ""
    frame = explorer.env(**_env_args(env))
    mask = selection_mask(explorer, sample, sel_rows)
    lines = describe(result["info"])
    try:
        fitted = model_result(explorer, sample)
    except ValueError as err:
        fitted = None
        lines.append(f"Model failed: {err}")
    env_text = f"environment {_label(explorer, 'env', env)}. "
    cells = "Each cell: value [16-84% or Wilson 68%] and the number of galaxies with a value."
    if fitted is None:
        inside = columns["conc_inside"]
        table = explorer.mass_table(columns, mask, frame["env"].to_numpy(float))
        caption = (f"Selected: {int((mask & inside).sum())} inside, {int((mask & ~inside).sum())} "
                   f"outside the 1 sigma contour; {env_text}{cells}")
        return lines, mass_table_component(table), caption
    model_columns, model = fitted
    inside = model_columns["model_inside"]
    table = explorer.mass_table(columns, mask, frame["env"].to_numpy(float), inside=inside)
    info = model["info"]
    lines += describe_model(model, mask, explorer.df)
    caption = (f"Selected: {int((mask & inside).sum())} inside, {int((mask & ~inside).sum())} "
               f"outside. Inside = within the 1σ band of the model ({info['n_peaks']} peaks, "
               f"{info['n_paths']} paths); {env_text}{cells}")
    return lines, mass_table_component(table), caption


def structures_table(explorer, sample, sel_rows=None):
    """Component of the model-table output: one row per peak and path of the model.

    Width s and intrinsic width, crest (n sigma), length, shift from the input and the
    number of galaxies of the selection (sample and sky selection) inside the band of the
    structure. An empty Div without a model, or when it fails.
    """
    try:
        fitted = model_result(explorer, sample)
    except ValueError:
        return html.Div()
    if fitted is None:
        return html.Div()
    columns, model = fitted
    mask = selection_mask(explorer, sample, sel_rows) & columns["model_inside"]
    who = np.asarray(columns["model_structure"], dtype=object)
    labels = [q["label"] for q in model["peaks"] + model["paths"]]
    counts = {label: int((mask & (who == label)).sum()) for label in labels}
    return html.Div([html.Div("Structures of the model (width s: 1σ distance from the structure)",
                              className="counts"),
                     structure_table_component(model, counts)])


def export_csv_frame(explorer, state, env, sample, sel_rows=None):
    """The selection (sample and sky selection) with the visible columns.

    With a concentration section also conc_inside (the 1 sigma contour of the map), and with
    a model model_inside, model_dist_sigma and model_structure.
    """
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
    if explorer.conc_settings is not None and sample:
        try:
            out["conc_inside"] = concentration_result(explorer, sample)[0]["conc_inside"][idx]
            fitted = model_result(explorer, sample)
        except ValueError:
            fitted = None
        if fitted is not None:
            for name in ("model_inside", "model_dist_sigma", "model_structure"):
                out[name] = np.asarray(fitted[0][name])[idx]
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
                  *_pick(sky_keys), Input("store-struct", "data"), Input("sky-pick", "value"))
    def _sky(sample, env, clicks, *values):
        if not sample or not env:
            return no_update
        n = len(sky_keys)
        struct, pick = values[n:]
        return sky_callback(explorer, partial_state(sky_keys, values[:n]), env, sample, clicks,
                            struct, pick or ())

    if explorer.conc_settings is not None:
        @app.callback(Output("store-struct", "data"), Output("pick-info", "children"),
                      Input("sky-fig", "clickData"), Input("sky-fig", "clickAnnotationData"),
                      Input("sky-fig", "relayoutData"), Input("btn-pick-all", "n_clicks"),
                      Input("btn-pick-none", "n_clicks"), Input("btn-clear-paths", "n_clicks"),
                      Input("store-sample", "data"), State("sky-pick", "value"),
                      State("store-struct", "data"))
        def _struct(click, annotation, relayout, _all, _none, _clear, sample, pick, struct):
            if not sample:
                return no_update, no_update
            fired = ctx.triggered[0]["prop_id"] if ctx.triggered else ""
            trigger = STRUCT_TRIGGERS.get(fired, "sample")
            new = struct_update(explorer, struct, sample, pick, trigger,
                                click=annotation if trigger == "annotation" else click,
                                relayout=relayout)
            if new == struct:
                return no_update, no_update
            return new, pick_counts(explorer, sample, new)

        @app.callback(Output("store-model", "data"), Output("model-status", "children"),
                      Input("btn-run-model", "n_clicks"), Input("btn-clear-model", "n_clicks"),
                      State("store-sample", "data"), State("store-struct", "data"),
                      prevent_initial_call=True)
        def _run(_run_clicks, _clear_clicks, sample, struct):
            action = "clear" if ctx.triggered_id == "btn-clear-model" else "run"
            return run_model(explorer, sample, struct, action)

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

    if explorer.conc_settings is not None:
        @app.callback(Output("conc-summary", "children"), Output("conc-table", "children"),
                      Output("conc-caption", "children"), Output("model-table", "children"),
                      Input("store-sample", "data"), Input("store-env", "data"),
                      Input("store-sel", "data"))
        def _concentration(sample, env, sel):
            if not sample or not env:
                return no_update, no_update, no_update, no_update
            lines, table, caption = concentration_callback(explorer, env, sample, sel)
            return ([html.Div(line) for line in lines], table, caption,
                    structures_table(explorer, sample, sel))

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
                  Output("snapshot-status", "children"),
                  Output("model-status", "children", allow_duplicate=True),
                  Input("up-snapshot", "contents"), prevent_initial_call=True)
    def _restore(contents):
        import base64
        n = len(STATE_CONTROLS)
        if not contents:
            return (*([no_update] * n), no_update, no_update)
        try:
            text = base64.b64decode(contents.split(",", 1)[1]).decode("utf-8")
            values, msg = restore_state(explorer, explorer.load_snapshot(text))
        except (ValueError, IndexError, UnicodeDecodeError) as err:
            return (*([no_update] * n), f"Cannot load the snapshot: {err}", no_update)
        return (*values, msg, model_status(values[STATE_KEYS.index("model")]))
