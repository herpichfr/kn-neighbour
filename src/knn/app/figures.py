"""Plotly and matplotlib figures of the explorer, as pure functions (data in, figure out).

The view frame 'd' taken by sky_figure is a DataFrame with one row per plotted galaxy and
the columns

    row, id, ra, dec, z, z_type ('spec' | 'phot'), logm, env,
    color (float, NaN = no value), size (float, optional, NaN = smallest)

Spec-z and photo-z galaxies are separate traces (circles / diamonds); rows
whose colour is NaN go into a grey 'no value' trace; empty traces are omitted.
"""

import io

import numpy as np
import plotly.graph_objects as go

# categorical palette, light surface (dataviz reference palette, fixed order)
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7",
           "#e34948"]
GREY = "#8a8f98"
RIDGE = "#00e5ff"
CONTOUR = "#f4f4f4"
HULL = "#8a8f98"
PEAK = "#ffffff"
OFF = "#9aa0a6"        # structures that are switched off
DRAWN = "#ff4dff"      # paths drawn by the user
MODEL = "#7fff00"      # fitted paths, 1σ band and peak circles of the model
HALO = "rgba(8,8,24,0.8)"
FONT = dict(family="system-ui, -apple-system, Segoe UI, Roboto, sans-serif", size=12,
            color="#222")
CONFIG_FORMATS = ("png", "svg")
# ridge styles: (group key, trace name, colour, width, dash, opacity); the first two are the
# switched-on ones (a halo is drawn under them), the last one is for the switched-off ridges
RIDGE_STYLES = (("strong", "ridges", RIDGE, 3.5, "solid", 1.0),
                ("open", "ridges to the border", RIDGE, 3.5, "dash", 1.0),
                ("off", "ridges off", OFF, 1.5, "dot", 0.6))
NO_LINES = {"x": [], "y": [], "customdata": [], "text": []}
CLICK_HOVER = "click: add peak<extra></extra>"
MAP_FLOOR = -1.0                      # lower bound of the colour scale: log10 of 0.1 x the mean
MAP_TICKS = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
CLEAR = [[0.0, "rgba(0,0,0,0)"], [1.0, "rgba(0,0,0,0)"]]


def graph_config(fmt="png", name="knn", erase=False):
    """dcc.Graph config: modebar image button saves PNG or SVG client side.

    erase: add the modebar button 'eraseshape' (Erase active shape), which removes the ridge or
    drawn path that is active in the Select maximums mode of the sky map.
    """
    if fmt not in CONFIG_FORMATS:
        raise ValueError(f"image format must be one of {CONFIG_FORMATS}")
    config = {"displaylogo": False, "responsive": True,
              "toImageButtonOptions": {"format": fmt, "filename": name, "scale": 2},
              "modeBarButtonsToRemove": ["sendDataToCloud"]}
    if erase:
        config["modeBarButtonsToAdd"] = ["eraseshape"]
    return config


def clip_range(values, clip=(2.0, 98.0)):
    """(lo, hi) percentiles of the finite values; widened if degenerate; (0, 1) if none."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(v, [clip[0], clip[1]])
    if not hi > lo:
        lo, hi = float(lo) - 0.5, float(hi) + 0.5
    return float(lo), float(hi)


def marker_sizes(values, out_range=(3.0, 10.0), clip=(2.0, 98.0)):
    """Marker sizes from a column: linear in the clipped range, smallest for NaN."""
    v = np.asarray(values, dtype=float)
    lo, hi = clip_range(v, clip)
    t = np.clip((v - lo) / (hi - lo), 0.0, 1.0)
    t = np.where(np.isfinite(t), t, 0.0)
    return out_range[0] + t * (out_range[1] - out_range[0])


def empty_figure(message="Nothing to show"):
    """Blank figure with a centred message, used when there is nothing to plot."""
    fig = go.Figure()
    fig.update_layout(template="plotly_white", font=FONT, xaxis=dict(visible=False),
                      yaxis=dict(visible=False), margin=dict(l=10, r=10, t=10, b=10),
                      annotations=[dict(text=message, showarrow=False, xref="paper",
                                        yref="paper", x=0.5, y=0.5,
                                        font=dict(size=14, color=GREY))])
    return fig


def _hover(kind):
    return ("id %{text}<br>RA %{customdata[4]:.4f}  Dec %{customdata[5]:.4f}<br>"
            f"z %{{customdata[1]:.4f}} ({kind})<br>log M* %{{customdata[2]:.2f}}<br>"
            "env %{customdata[3]:.2f}<extra></extra>")


def _customdata(d):
    return np.column_stack([d["row"], d["z"], d["logm"], d["env"], d["ra"],
                            d["dec"]]).astype(np.float32)


def _groups(d):
    """[(name, kind label, symbol, opacity, base size, sub-frame, show colour)] of non-empty groups."""
    color = d["color"].to_numpy(float)
    has = np.isfinite(color)
    spec = (d["z_type"] == "spec").to_numpy()
    out = []
    for name, sel, symbol, opacity, size in (
            ("spec-z", spec & has, "circle", 0.9, 1.0),
            ("photo-z", ~spec & has, "diamond", 0.5, 1.15)):
        if sel.any():
            out.append((name, name, symbol, opacity, size, d[sel], True))
    if (~has).any():
        out.append(("no value", "no colour value", "circle", 0.35, 0.8, d[~has], False))
    return out


def _sizes(sub, base):
    if "size" in sub.columns and np.isfinite(sub["size"].to_numpy(float)).any():
        return sub["size"].to_numpy(float)
    return base


def _coloraxis(d, label, clip, colorscale):
    lo, hi = clip_range(d["color"], clip)
    return dict(colorscale=colorscale, cmin=lo, cmax=hi,
                colorbar=dict(title=dict(text=label, side="right"), thickness=14, len=0.8))


def _scaled_sizes(d, size_range, clip=(2.0, 98.0)):
    """d with 'size' replaced by marker sizes (computed over the whole frame)."""
    if "size" not in d.columns or not np.isfinite(d["size"].to_numpy(float)).any():
        return d.drop(columns=["size"], errors="ignore")
    d = d.copy()
    d["size"] = marker_sizes(d["size"].to_numpy(float), size_range, clip)
    return d


def _structure_trace(group, name, colour, width, dash, opacity=1.0, legendgroup=None):
    """Scattergl of a structure style: a line whose vertices answer hover and click.

    group: dict x, y, customdata ([kind, id] per vertex, [None, None] at the None separators)
    and text (hover). The vertices are small markers of the line colour: a Scattergl line
    without markers is not hoverable, and invisible markers (opacity 0) are not either.
    """
    return go.Scattergl(
        x=group["x"], y=group["y"], mode="lines+markers", name=name, opacity=opacity,
        line=dict(color=colour, width=width, dash=dash),
        marker=dict(size=4 if width >= 3 else 3, color=colour),
        customdata=group["customdata"], text=group["text"], hovertemplate="%{text}<extra></extra>",
        legendgroup=legendgroup or name, showlegend=len(group["x"]) > 0)


def map_scale(z):
    """(log z, zmax, tickvals, ticktext) of the map over its mean for the colour scale.

    z: the map over its mean (NaN outside the survey, 0 or small for an empty cell). The colours
    are log10(z) between MAP_FLOOR (z <= 0.1; the empty cells are set to it, so every cell inside
    the survey keeps a finite value) and zmax = log10 of the 99.5th percentile of the finite cells
    (at least z = 1). The ticks are the MAP_TICKS inside that range, labelled with z.
    """
    lin = np.asarray(z, dtype=float)
    finite = lin[np.isfinite(lin)]
    top = max(float(np.percentile(finite, 99.5)) if finite.size else 1.0, 1.0)
    zmax = float(np.log10(top))
    ticks = [t for t in MAP_TICKS if MAP_FLOOR - 1e-9 <= np.log10(t) <= zmax + 1e-9]
    return (np.log10(np.maximum(lin, 10.0 ** MAP_FLOOR)), zmax,
            [float(np.log10(t)) for t in ticks], [f"{t:g}" for t in ticks])


def _overlay_traces(fig, overlay, density, pick=()):
    """Add the concentration traces to a sky figure (see sky_figure for their order).

    The map is coloured by log10(map / mean) (map_scale) and its hover shows the linear map / mean
    (customdata, only without a pick mode). pick: the pick modes that are on. With any of them
    the map does not answer the hover with its value; with 'peaks' it answers "click: add peak" (a click on an empty place of the
    map adds a peak), and when the density background is off a fully transparent heatmap on the
    same grid takes its place. With 'paths' the ridges and drawn paths that are on are layout
    shapes (sky_figure), so their traces are left empty.
    """
    pick = tuple(pick)
    logz, zmax, tickvals, ticktext = map_scale(overlay["map_z"])
    if density:
        hover = (dict(hovertemplate=CLICK_HOVER) if "peaks" in pick else
                 dict(hoverinfo="skip") if pick else dict(
                     customdata=np.round(np.asarray(overlay["map_z"], dtype=float), 3),
                     hovertemplate="RA %{x:.2f}  Dec %{y:.2f}<br>map/mean "
                                   "%{customdata:.2f}<extra></extra>"))
        fig.add_trace(go.Heatmap(
            x=overlay["map_ra"], y=overlay["map_dec"], z=logz, name="concentration",
            colorscale="Inferno", zmin=MAP_FLOOR, zmax=zmax, opacity=1.0, hoverongaps=False,
            colorbar=dict(title=dict(text="map / mean", side="bottom"), orientation="h",
                          thickness=10, len=0.5, y=-0.3, yanchor="top", tickmode="array",
                          tickvals=tickvals, ticktext=ticktext),
            zsmooth=False, **hover))
    elif "peaks" in pick:
        fig.add_trace(go.Heatmap(
            x=overlay["map_ra"], y=overlay["map_dec"], z=logz, name="click map",
            colorscale=CLEAR, showscale=False, showlegend=False, hoverongaps=False,
            hovertemplate=CLICK_HOVER, zsmooth=False))
    fig.add_trace(go.Scattergl(
        x=overlay["hull_x"], y=overlay["hull_y"], mode="lines", name="survey area",
        line=dict(color=HULL, width=1.0), hoverinfo="skip"))
    fig.add_trace(go.Scattergl(
        x=overlay["contour_x"], y=overlay["contour_y"], mode="lines", name="contours (2σ, 3σ, ...)",
        line=dict(color=CONTOUR, width=1.0), opacity=0.8, hoverinfo="skip"))
    fig.add_trace(go.Scattergl(
        x=overlay["contour1_x"], y=overlay["contour1_y"], mode="lines", name="1σ contour",
        line=dict(color=CONTOUR, width=2.8), hoverinfo="skip"))
    groups = overlay["ridge_groups"]
    for key, name, colour, width, dash, opacity in RIDGE_STYLES[:2]:
        fig.add_trace(go.Scattergl(
            x=groups[key]["x"], y=groups[key]["y"], mode="lines", name=name + " halo",
            line=dict(color=HALO, width=width + 3.0, dash=dash), hoverinfo="skip",
            showlegend=False, legendgroup=name))
    editing = "paths" in pick
    for key, name, colour, width, dash, opacity in RIDGE_STYLES:
        group = NO_LINES if editing and key != "off" else groups[key]
        fig.add_trace(_structure_trace(group, name, colour, width, dash, opacity))
    drawn = overlay["drawn"]
    fig.add_trace(_structure_trace(NO_LINES if editing else drawn["on"], "drawn paths", DRAWN, 2.0,
                                   "dash"))
    fig.add_trace(_structure_trace(drawn["off"], "drawn paths off", OFF, 1.5, "dot", 0.6))
    model = overlay.get("model")
    if model is not None:
        fig.add_trace(go.Scattergl(
            x=model["band_x"], y=model["band_y"], mode="lines", name="model: 1σ band",
            line=dict(color=MODEL, width=1.0, dash="dash"), hoverinfo="skip",
            legendgroup="model: 1σ band"))
        fig.add_trace(go.Scattergl(
            x=model["circle_x"], y=model["circle_y"], mode="lines", name="model: peak circles",
            line=dict(color=MODEL, width=1.5), hoverinfo="skip", showlegend=False,
            legendgroup="model: 1σ band"))
        fig.add_trace(go.Scattergl(
            x=model["path_x"], y=model["path_y"], mode="lines", name="model: fitted paths",
            line=dict(color=MODEL, width=3.0), hoverinfo="skip"))
    # an SVG trace (legend entry and hover, click through customdata); it lies under the WebGL
    # canvas of the other traces, so the stars that show are the annotations of peak_annotations
    on = overlay["peak_on"]
    fig.add_trace(go.Scatter(
        x=overlay["peak_ra"], y=overlay["peak_dec"], mode="markers", name="peaks",
        marker=dict(symbol="star", size=18, color=[PEAK if k else OFF for k in on],
                    line=dict(width=1.0, color="#111")),
        customdata=[["peak", i] for i in overlay["peak_id"]], text=overlay["peak_text"],
        hovertemplate="%{text}<extra></extra>"))
    user = overlay.get("user_peak_id") or []
    if user:
        fig.add_trace(go.Scatter(
            x=overlay["user_peak_ra"], y=overlay["user_peak_dec"], mode="markers",
            name="peaks added by hand",
            marker=dict(symbol="star", size=18, color=DRAWN, line=dict(width=1.0, color="#111")),
            customdata=[["user_peak", i] for i in user],
            text=[f"user peak {i} (click to remove)" for i in user],
            hovertemplate="%{text}<extra></extra>"))


def _svg_path(ra, dec):
    """'M x,y L x,y ...' of a polyline in data coordinates (a plotly path shape)."""
    return "M" + " L".join(f"{float(a):.5f},{float(b):.5f}" for a, b in zip(ra, dec))


def structure_shapes(overlay):
    """Layout shapes of the ridges and drawn paths that are on, for the Select maximums mode.

    One editable path shape per item of overlay['shapes'] ({name 'ridge <id>' or 'path <id>',
    kind, open, ra, dec}): cyan (dashed when the ridge runs to the border) for a ridge, magenta
    dashes for a drawn path. The shape names are what struct_update reads back from relayoutData.
    """
    out = []
    for s in overlay.get("shapes") or []:
        if len(s["ra"]) < 2:
            continue
        line = (dict(color=DRAWN, width=2, dash="dash") if s["kind"] == "path" else
                dict(color=RIDGE, width=3.5, dash="dash" if s["open"] else "solid"))
        out.append(dict(type="path", path=_svg_path(s["ra"], s["dec"]), xref="x", yref="y",
                        editable=True, name=s["name"], line=line, layer="above"))
    return out


def peak_annotations(overlay):
    """One layout annotation per peak, drawn above every trace: ★ (closed) or ☆ (open).

    White when the peak is on, smaller and grey (OFF) when it is switched off. The
    annotation captures the click events (a dcc.Graph reports them in clickAnnotationData)
    and is named f"peak {id}". The peaks added by hand (overlay['user_peak_*']) follow, as
    magenta ★ named f"user peak {id}".
    """
    notes = [dict(x=ra, y=dec, xref="x", yref="y", text="☆" if is_open else "★",
                  showarrow=False, font=dict(size=22 if on else 16, color=PEAK if on else OFF),
                  hovertext=text, captureevents=True, name=f"peak {pid}")
             for ra, dec, is_open, text, on, pid in zip(
                 overlay["peak_ra"], overlay["peak_dec"], overlay["peak_open"],
                 overlay["peak_text"], overlay["peak_on"], overlay["peak_id"])]
    notes += [dict(x=ra, y=dec, xref="x", yref="y", text="★", showarrow=False,
                   font=dict(size=22, color=DRAWN), hovertext=f"user peak {pid} (click to remove)",
                   captureevents=True, name=f"user peak {pid}")
              for ra, dec, pid in zip(overlay.get("user_peak_ra") or [],
                                      overlay.get("user_peak_dec") or [],
                                      overlay.get("user_peak_id") or [])]
    return notes


def sky_figure(d, color_label="env", clip=(2.0, 98.0), density=True, colorscale="Viridis",
               uirevision="sky", bins=60, overlay=None, pick=(), editrevision=None):
    """RA vs Dec (RA increasing to the left), Scattergl, box / lasso selectable.

    Traces, in order: 'density' (Heatmap of the 2D histogram of d, if density; with an
    overlay it is replaced by the 'concentration' Heatmap, or, without density but with the
    pick mode 'peaks', by a fully transparent 'click map' Heatmap on the same grid that only
    catches the clicks), then, if overlay is given,
    'survey area' (the hull of the catalogue, thin grey), 'contours (2σ, 3σ, ...)',
    '1σ contour' (heavier), the dark halos 'ridges halo', 'ridges to the border halo'
    (wide lines under the cyan ones, so that they show on every colour of the map; a halo
    shares the legend entry of its line), the ridge segments that are on: 'ridges' (cyan,
    solid) and 'ridges to the border' (cyan, dashed: the segment reaches the border), then
    'ridges off' (grey dots: switched off), 'drawn paths' (magenta dashes) and 'drawn paths
    off' (grey dots), then, when overlay['model'] is set, 'model: 1σ band' (the edges, dashed),
    'model: peak circles' (same legend entry as the band) and 'model: fitted paths' (green),
    then 'peaks' (legend, hover and click; the stars drawn on the map are layout annotations,
    ★ and ☆ for peaks whose region reaches the border, white when on, smaller and grey when off,
    rebuilt with every figure because an SVG trace would lie under the WebGL canvas), then
    'peaks added by hand' (only when there are some; magenta ★ annotations named 'user peak
    <id>'); then, last so that they are on top, 'spec-z', 'photo-z', 'no value' (those with
    data).
    customdata[0] of every point of the galaxy traces is the row of the catalogue (read back
    from selectedData); a vertex of a ridge, drawn path or peak carries customdata [kind, id]
    with kind 'ridge', 'path', 'peak' or 'user_peak' (read back from clickData); a point of
    the map Heatmap has no customdata (clickData gives its x, y, z).

    overlay: dict with map_ra, map_dec, map_z (2D, the map over its mean, NaN outside the
    survey area; used if density or with 'peaks'; the Heatmap shows log10 of it, from 0.1 x the
    mean to the 99.5th percentile, see map_scale, and its hover the linear value), hull_x, hull_y, contour_x, contour_y,
    contour1_x, contour1_y (lists of degrees, None separating the polylines),
    ridge_groups {strong, open, off: {x, y, customdata, text}}, drawn {on, off: same},
    shapes ([{name, kind 'ridge' | 'path', open, ra, dec}]: the ridges and drawn paths that
    are on, for the layout shapes), model (None or path_x, path_y, band_x, band_y, circle_x,
    circle_y) and peak_ra, peak_dec, peak_text, peak_open, peak_on, peak_id, optionally
    user_peak_ra, user_peak_dec, user_peak_id.

    pick: the pick modes that are on ('peaks', 'paths'). With any of them the galaxies do not
    answer the hover (a click lands on a structure) and the map does not show its value; with
    'peaks' the map answers "click: add peak". With 'paths' the drag mode is 'drawopenpath' (a
    freehand path) and the ridges and drawn paths that are on are editable layout shapes named
    'ridge <id>' and 'path <id>' (their traces are left empty; the user activates one by a click
    and removes it with the eraser button of the modebar), else the drag mode is 'select' and
    layout.shapes is empty. A freehand path becomes a trace or shape after the callbacks read
    it; editrevision (changed after every drawing and with the mode) makes the browser drop the
    shapes it still holds.
    """
    if len(d) == 0:
        return empty_figure("No galaxies in the selected sample")
    pick = tuple(pick or ())
    d = _scaled_sizes(d, (3.0, 10.0), clip)
    fig = go.Figure()
    if overlay is not None:
        _overlay_traces(fig, overlay, density, pick)
    elif density:
        counts, xe, ye = np.histogram2d(d["ra"], d["dec"], bins=bins)
        fig.add_trace(go.Heatmap(
            x=(xe[:-1] + xe[1:]) / 2, y=(ye[:-1] + ye[1:]) / 2, z=counts.T, name="density",
            colorscale=[[0.0, "rgba(138,143,152,0)"], [1.0, "rgba(138,143,152,0.55)"]],
            showscale=False, hoverinfo="skip", zsmooth=False))
    for name, kind, symbol, opacity, base, sub, coloured in _groups(d):
        marker = dict(symbol=symbol, opacity=opacity, size=_sizes(sub, 4.0 * base))
        if overlay is not None:
            marker["line"] = dict(width=0.5, color="rgba(255,255,255,0.7)")
        if coloured:
            marker.update(color=sub["color"].to_numpy(float), coloraxis="coloraxis")
        else:
            marker["color"] = GREY
        hover = dict(hoverinfo="skip") if pick else dict(hovertemplate=_hover(kind))
        fig.add_trace(go.Scattergl(
            x=sub["ra"].to_numpy(float), y=sub["dec"].to_numpy(float), mode="markers",
            name=name, marker=marker, customdata=_customdata(sub), text=sub["id"].to_numpy(),
            **hover))
    fig.update_layout(
        template="plotly_white", font=FONT, hovermode="closest",
        dragmode="drawopenpath" if "paths" in pick else "select",
        newshape=dict(line=dict(color=DRAWN, width=2), opacity=1),
        shapes=structure_shapes(overlay) if overlay is not None and "paths" in pick else [],
        xaxis=dict(title="RA [deg]", autorange="reversed", zeroline=False),
        yaxis=dict(title="Dec [deg]", zeroline=False),
        coloraxis=_coloraxis(d, color_label, clip, colorscale), uirevision=uirevision,
        margin=dict(l=55, r=10, t=10, b=45 if overlay is None or not density else 160),
        legend=dict(orientation="h", y=1.08, x=0),
        annotations=[] if overlay is None else peak_annotations(overlay))
    if editrevision is not None:
        fig.update_layout(editrevision=editrevision)
    return fig


def _rgba(hex_colour, alpha):
    h = hex_colour.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def trend_groups(trend):
    """Ordered [(mass_bin, z_type, sub-table, colour, dash)] of a trend table.

    Colour follows the mass bin (palette in bin order, cycled past 8 bins),
    spec-z is solid, photo-z dashed, a table without z_type split solid.
    """
    labels = list(dict.fromkeys(trend["mass_bin"]))
    out = []
    for i, label in enumerate(labels):
        for ztype in ("spec", "phot", "all"):
            sub = trend[(trend["mass_bin"] == label) & (trend["z_type"] == ztype)]
            if len(sub):
                out.append((label, ztype, sub.sort_values("bin"), PALETTE[i % len(PALETTE)],
                            "dash" if ztype == "phot" else "solid"))
    return out


def trend_figure(trend, x_label="env", y_label="y"):
    """Trend lines per mass bin (colour) and z_type (spec solid, photo-z dashed).

    Per group: a band trace (16-84 percentiles, or the Wilson interval for a
    fraction; omitted when fewer than two bins are valid) then a line trace
    through the bin medians. Bins below the minimum count are gaps.
    """
    if len(trend) == 0:
        return empty_figure("No data in the selection")
    fraction = (trend["kind"] == "fraction").any()
    fig = go.Figure()
    for label, ztype, sub, colour, dash in trend_groups(trend):
        name = label if ztype == "all" else f"{label} ({'spec-z' if ztype == 'spec' else 'photo-z'})"
        group = f"{label}|{ztype}"
        ok = sub["y"].notna().to_numpy()
        if ok.sum() >= 2:
            s = sub[ok]
            fig.add_trace(go.Scatter(
                x=np.concatenate([s["x"], s["x"][::-1]]),
                y=np.concatenate([s["y_hi"], s["y_lo"][::-1]]), mode="lines", fill="toself",
                fillcolor=_rgba(colour, 0.15), line=dict(width=0), hoverinfo="skip",
                showlegend=False, legendgroup=group, name=name + " band"))
        fig.add_trace(go.Scatter(
            x=sub["x"], y=sub["y"], mode="lines+markers", name=name, legendgroup=group,
            line=dict(color=colour, dash=dash, width=2), marker=dict(size=7, color=colour),
            customdata=sub["n"].to_numpy(),
            hovertemplate=f"{name}<br>{x_label} %{{x:.2f}}<br>{y_label} %{{y:.3f}}"
                          "<br>N %{customdata}<extra></extra>"))
    fig.update_layout(
        template="plotly_white", font=FONT, hovermode="closest", margin=dict(l=60, r=10, t=10, b=45),
        xaxis=dict(title=x_label), yaxis=dict(title=y_label, range=[0, 1] if fraction else None),
        legend=dict(title=dict(text="log M* bin"), orientation="v"))
    return fig


def hist_figure(selected, full, label="env", bins=30):
    """Histogram (density) of a column for the selection and for the full sample.

    Two Bar traces ('full sample', 'selection') on common bins; a trace with
    no finite values is omitted.
    """
    sel = np.asarray(selected, dtype=float)
    sel = sel[np.isfinite(sel)]
    allv = np.asarray(full, dtype=float)
    allv = allv[np.isfinite(allv)]
    if allv.size == 0 and sel.size == 0:
        return empty_figure("No values")
    edges = np.histogram_bin_edges(allv if allv.size else sel, bins=bins)
    centres, width = (edges[:-1] + edges[1:]) / 2, np.diff(edges)
    fig = go.Figure()
    for name, values, colour in (("full sample", allv, GREY), ("selection", sel, PALETTE[0])):
        if values.size:
            dens, _ = np.histogram(values, bins=edges, density=True)
            fig.add_trace(go.Bar(x=centres, y=dens, width=width, name=name, opacity=0.65,
                                 marker=dict(color=colour, line=dict(width=0)),
                                 hovertemplate=f"{name}<br>{label} %{{x:.2f}}<br>density "
                                               "%{y:.3f}<extra></extra>"))
    fig.update_layout(template="plotly_white", font=FONT, barmode="overlay",
                      margin=dict(l=60, r=10, t=10, b=45), xaxis=dict(title=label),
                      yaxis=dict(title="density"), legend=dict(orientation="h", y=1.1, x=0))
    return fig


def counts_text(trend):
    """Plain text: N per bin of every group (one line per mass bin and z_type)."""
    if len(trend) == 0:
        return "No data in the selection."
    lines = []
    for label, ztype, sub, _, _ in trend_groups(trend):
        n = " ".join(f"{int(v)}" if ok else f"{int(v)}*"
                     for v, ok in zip(sub["n"], sub["y"].notna()))
        tag = "all" if ztype == "all" else ztype
        lines.append(f"log M* {label:>6} {tag:>4}: N/bin = {n}")
    lines.append("* below the minimum count: bin left empty")
    return "\n".join(lines)


def trend_matplotlib(trend, x_label="env", y_label="y", title=None):
    """The trend plot as PDF bytes, rendered server side with matplotlib (no pyplot state)."""
    from matplotlib.backends.backend_pdf import FigureCanvasPdf
    from matplotlib.figure import Figure

    fig = Figure(figsize=(5.2, 4.0))
    FigureCanvasPdf(fig)
    ax = fig.add_subplot(111)
    fraction = len(trend) > 0 and (trend["kind"] == "fraction").any()
    for label, ztype, sub, colour, dash in trend_groups(trend):
        name = label if ztype == "all" else f"{label} ({'spec-z' if ztype == 'spec' else 'photo-z'})"
        ok = sub["y"].notna().to_numpy()
        s = sub[ok]
        if len(s) >= 2:
            ax.fill_between(s["x"], s["y_lo"], s["y_hi"], color=colour, alpha=0.15, linewidth=0)
        ax.plot(sub["x"], sub["y"], color=colour, ls="--" if dash == "dash" else "-", marker="o",
                ms=4, lw=1.5, label=name)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if fraction:
        ax.set_ylim(0, 1)
    if len(trend):
        ax.legend(title="log M* bin", fontsize=7, title_fontsize=8, frameon=False)
    if title:
        ax.set_title(title, fontsize=9)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="pdf")
    return buf.getvalue()
