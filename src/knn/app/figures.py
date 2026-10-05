"""Plotly and matplotlib figures of the explorer, as pure functions (data in, figure out).

The view frame 'd' taken by sky_figure and scatter3d_figure is a DataFrame
with one row per plotted galaxy and the columns

    row, id, ra, dec, z, z_type ('spec' | 'phot'), logm, env,
    cx, cy, cz (comoving Mpc, for the 3D view),
    color (float, NaN = no value), size (float, optional, NaN = smallest),
    selected (bool, optional: highlighted in the 3D view)

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
FONT = dict(family="system-ui, -apple-system, Segoe UI, Roboto, sans-serif", size=12,
            color="#222")
CONFIG_FORMATS = ("png", "svg")


def graph_config(fmt="png", name="knn"):
    """dcc.Graph config: modebar image button saves PNG or SVG client side."""
    if fmt not in CONFIG_FORMATS:
        raise ValueError(f"image format must be one of {CONFIG_FORMATS}")
    return {"displaylogo": False, "responsive": True,
            "toImageButtonOptions": {"format": fmt, "filename": name, "scale": 2},
            "modeBarButtonsToRemove": ["sendDataToCloud"]}


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


def sky_figure(d, color_label="env", clip=(2.0, 98.0), density=True, colorscale="Viridis",
               uirevision="sky", bins=60):
    """RA vs Dec (RA increasing to the left), Scattergl, box / lasso selectable.

    Traces, in order: 'density' (Heatmap of the 2D histogram of d, if density),
    'spec-z', 'photo-z', 'no value' (those with data). customdata[0] of every
    point is the row of the catalogue (read back from selectedData).
    """
    if len(d) == 0:
        return empty_figure("No galaxies in the selected sample")
    d = _scaled_sizes(d, (3.0, 10.0), clip)
    fig = go.Figure()
    if density:
        counts, xe, ye = np.histogram2d(d["ra"], d["dec"], bins=bins)
        fig.add_trace(go.Heatmap(
            x=(xe[:-1] + xe[1:]) / 2, y=(ye[:-1] + ye[1:]) / 2, z=counts.T, name="density",
            colorscale=[[0.0, "rgba(138,143,152,0)"], [1.0, "rgba(138,143,152,0.55)"]],
            showscale=False, hoverinfo="skip", zsmooth=False))
    for name, kind, symbol, opacity, base, sub, coloured in _groups(d):
        marker = dict(symbol=symbol, opacity=opacity, size=_sizes(sub, 4.0 * base))
        if coloured:
            marker.update(color=sub["color"].to_numpy(float), coloraxis="coloraxis")
        else:
            marker["color"] = GREY
        fig.add_trace(go.Scattergl(
            x=sub["ra"].to_numpy(float), y=sub["dec"].to_numpy(float), mode="markers",
            name=name, marker=marker, customdata=_customdata(sub), text=sub["id"].to_numpy(),
            hovertemplate=_hover(kind)))
    fig.update_layout(
        template="plotly_white", font=FONT, dragmode="select", hovermode="closest",
        xaxis=dict(title="RA [deg]", autorange="reversed"), yaxis=dict(title="Dec [deg]"),
        coloraxis=_coloraxis(d, color_label, clip, colorscale), uirevision=uirevision,
        margin=dict(l=55, r=10, t=10, b=45), legend=dict(orientation="h", y=1.08, x=0))
    return fig


def scatter3d_figure(d, coords="radecz", color_label="env", clip=(2.0, 98.0),
                     colorscale="Viridis", uirevision="3d"):
    """Scatter3d (WebGL) map; coords 'radecz' (RA, Dec, z) or 'xyz' (comoving Mpc).

    Traces: 'spec-z', 'photo-z', 'no value' (those with data), then
    'selection' (rows with d['selected']) drawn as larger open red markers.
    """
    if coords not in ("radecz", "xyz"):
        raise ValueError("coords must be 'radecz' or 'xyz'")
    if len(d) == 0:
        return empty_figure("No galaxies in the selected sample")
    d = _scaled_sizes(d, (2.0, 7.0), clip)
    cx, cy, cz = (("ra", "dec", "z") if coords == "radecz" else ("cx", "cy", "cz"))
    fig = go.Figure()
    for name, kind, symbol, opacity, base, sub, coloured in _groups(d):
        marker = dict(symbol=symbol, opacity=opacity, size=_sizes(sub, 2.5 * base),
                      line=dict(width=0))
        if coloured:
            marker.update(color=sub["color"].to_numpy(float), coloraxis="coloraxis")
        else:
            marker["color"] = GREY
        fig.add_trace(go.Scatter3d(
            x=sub[cx].to_numpy(float), y=sub[cy].to_numpy(float), z=sub[cz].to_numpy(float),
            mode="markers", name=name, marker=marker, customdata=_customdata(sub),
            text=sub["id"].to_numpy(), hovertemplate=_hover(kind)))
    if "selected" in d.columns and d["selected"].any():
        sel = d[d["selected"].to_numpy(bool)]
        fig.add_trace(go.Scatter3d(
            x=sel[cx].to_numpy(float), y=sel[cy].to_numpy(float), z=sel[cz].to_numpy(float),
            mode="markers", name="selection", hoverinfo="skip",
            marker=dict(symbol="circle-open", size=6, color="#e34948", line=dict(width=2))))
    if coords == "radecz":
        scene = dict(xaxis=dict(title="RA [deg]", autorange="reversed"),
                     yaxis=dict(title="Dec [deg]"), zaxis=dict(title="z"), aspectmode="cube")
    else:
        scene = dict(xaxis=dict(title="x [Mpc]"), yaxis=dict(title="y [Mpc]"),
                     zaxis=dict(title="z [Mpc]"), aspectmode="data")
    fig.update_layout(template="plotly_white", font=FONT, scene=scene, uirevision=uirevision,
                      coloraxis=_coloraxis(d, color_label, clip, colorscale),
                      margin=dict(l=0, r=0, t=10, b=0),
                      legend=dict(orientation="h", y=1.05, x=0))
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
