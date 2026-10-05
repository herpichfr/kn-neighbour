import numpy as np
import pandas as pd
import pytest

from knn.app.figures import (PALETTE, clip_range, counts_text, empty_figure, graph_config,
                             hist_figure, marker_sizes, scatter3d_figure, sky_figure,
                             trend_figure, trend_matplotlib, trend_groups)
from knn.explorer import trend_table


def _view(n=300, seed=0, nan_color=0, size=False, selected=0):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({
        "row": np.arange(n), "id": [str(i) for i in range(n)],
        "ra": rng.uniform(180, 181, n), "dec": rng.uniform(0, 1, n),
        "z": rng.uniform(0.01, 0.03, n),
        "z_type": np.where(np.arange(n) % 4 == 0, "phot", "spec"),
        "logm": rng.normal(9, 1, n), "env": rng.normal(0, 1, n),
        "cx": rng.normal(size=n), "cy": rng.normal(size=n), "cz": rng.normal(size=n),
        "color": rng.normal(size=n)})
    d.loc[:nan_color - 1, "color"] = np.nan
    if size:
        d["size"] = rng.uniform(0, 5, n)
    d["selected"] = np.arange(n) < selected
    return d


def _types(fig):
    return [type(t).__name__ for t in fig.data]


def test_sky_figure_traces_and_axes():
    fig = sky_figure(_view())
    assert _types(fig) == ["Heatmap", "Scattergl", "Scattergl"]
    assert [t.name for t in fig.data] == ["density", "spec-z", "photo-z"]
    assert fig.layout.xaxis.autorange == "reversed"             # RA increases to the left
    assert fig.layout.dragmode == "select"
    assert fig.data[1].marker.symbol == "circle" and fig.data[2].marker.symbol == "diamond"
    assert fig.data[1].marker.opacity > fig.data[2].marker.opacity
    assert fig.data[1].customdata.shape == (225, 6)             # row, z, logm, env, ra, dec
    np.testing.assert_array_equal(fig.data[1].customdata[:, 0],
                                  np.flatnonzero(_view()["z_type"] == "spec"))
    assert len(fig.data[1].x) + len(fig.data[2].x) == 300
    assert "id %{text}" in fig.data[1].hovertemplate


def test_sky_figure_options():
    assert _types(sky_figure(_view(), density=False)) == ["Scattergl", "Scattergl"]
    fig = sky_figure(_view(nan_color=20))
    assert [t.name for t in fig.data][-1] == "no value" and fig.data[-1].marker.color == "#8a8f98"
    only_spec = _view()
    only_spec["z_type"] = "spec"
    assert [t.name for t in sky_figure(only_spec).data] == ["density", "spec-z"]
    fig = sky_figure(_view(size=True), clip=(5, 95))
    assert fig.data[1].marker.size.min() >= 3 and fig.data[1].marker.size.max() <= 10
    lo, hi = np.percentile(_view()["color"], [5, 95])
    assert fig.layout.coloraxis.cmin == pytest.approx(lo)
    assert fig.layout.coloraxis.cmax == pytest.approx(hi)
    assert fig.layout.uirevision == "sky"
    assert sky_figure(_view(), uirevision="sky3").layout.uirevision == "sky3"


def test_scatter3d_figure_traces():
    fig = scatter3d_figure(_view())
    assert _types(fig) == ["Scatter3d", "Scatter3d"]
    assert [t.name for t in fig.data] == ["spec-z", "photo-z"]
    d = _view()
    np.testing.assert_allclose(fig.data[0].x, d.loc[d.z_type == "spec", "ra"])
    assert fig.layout.scene.xaxis.title.text == "RA [deg]"
    xyz = scatter3d_figure(_view(selected=10), coords="xyz")
    assert [t.name for t in xyz.data] == ["spec-z", "photo-z", "selection"]
    d = _view(selected=10)
    np.testing.assert_allclose(xyz.data[0].x, d.loc[d.z_type == "spec", "cx"])
    assert xyz.layout.scene.xaxis.title.text == "x [Mpc]"
    assert len(xyz.data[2].x) == 10
    assert [t.name for t in scatter3d_figure(_view(nan_color=5)).data][-1] == "no value"
    with pytest.raises(ValueError):
        scatter3d_figure(_view(), coords="polar")


def test_empty_frames_give_an_annotated_figure():
    for fig in (sky_figure(_view(0)), scatter3d_figure(_view(0)), trend_figure(
            trend_table([], [], None, None, None)), empty_figure("x")):
        assert len(fig.data) == 0 and len(fig.layout.annotations) == 1


def _trend(boolean=False, seed=0, n=3000):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = (rng.random(n) < 0.5) if boolean else x + rng.normal(0, 0.3, n)
    return trend_table(x, y, None, rng.uniform(8, 11, n),
                       np.where(rng.random(n) < 0.8, "spec", "phot"), [8, 9, 10, 11], nbins=6)


def test_trend_figure_has_a_band_and_a_line_per_group():
    t = _trend()
    fig = trend_figure(t, "env", "y")
    groups = trend_groups(t)
    assert len(groups) == 6                                    # 3 mass bins x spec / photo-z
    assert _types(fig) == ["Scatter", "Scatter"] * 6
    lines = [tr for tr in fig.data if tr.mode == "lines+markers"]
    assert len(lines) == 6
    assert {tr.line.dash for tr in lines} == {"solid", "dash"}
    spec = [tr for tr in lines if "spec-z" in tr.name]
    phot = [tr for tr in lines if "photo-z" in tr.name]
    assert all(tr.line.dash == "solid" for tr in spec) and all(tr.line.dash == "dash" for tr in phot)
    assert [tr.line.color for tr in spec] == PALETTE[:3]       # colour follows the mass bin
    bands = [tr for tr in fig.data if tr.fill == "toself"]
    assert len(bands) == 6 and not any(tr.showlegend for tr in bands)
    assert fig.layout.yaxis.range is None


def test_trend_figure_fraction_axis_and_gaps():
    t = _trend(boolean=True)
    fig = trend_figure(t, "env", "quenched")
    assert tuple(fig.layout.yaxis.range) == (0, 1)
    sparse = t.copy()
    sparse.loc[sparse["z_type"] == "phot", ["y", "y_lo", "y_hi"]] = np.nan
    fig = trend_figure(sparse)
    assert _types(fig).count("Scatter") == 3 * 2 + 3           # bands only for spec-z
    assert counts_text(sparse).count("*") >= 3


def test_hist_and_counts_text():
    rng = np.random.default_rng(0)
    full = rng.normal(size=1000)
    fig = hist_figure(full[:100], full, "env")
    assert _types(fig) == ["Bar", "Bar"] and [t.name for t in fig.data] == ["full sample", "selection"]
    np.testing.assert_allclose(sum(fig.data[0].y * fig.data[0].width), 1.0)
    assert [t.name for t in hist_figure([], full).data] == ["full sample"]
    assert len(hist_figure([], []).data) == 0
    text = counts_text(_trend())
    assert "8-9" in text and "spec" in text and "phot" in text and "N/bin" in text
    assert counts_text(trend_table([], [], None, None, None)) == "No data in the selection."


def test_trend_matplotlib_pdf():
    pdf = trend_matplotlib(_trend(boolean=True), "log(1+delta)", "quenched fraction", title="t")
    assert pdf.startswith(b"%PDF") and len(pdf) > 3000
    assert trend_matplotlib(trend_table([], [], None, None, None)).startswith(b"%PDF")


def test_helpers():
    assert clip_range([1, 2, 3, np.nan, 100], (0, 100)) == (1.0, 100.0)
    lo, hi = clip_range([5.0, 5.0])
    assert hi - lo == pytest.approx(1.0)
    assert clip_range([np.nan]) == (0.0, 1.0)
    s = marker_sizes([0, 5, 10, np.nan], (3, 9), (0, 100))
    assert s[0] == 3 and s[2] == 9 and s[3] == 3 and 3 < s[1] < 9
    cfg = graph_config("svg", "name")
    assert cfg["toImageButtonOptions"]["format"] == "svg"
    assert cfg["toImageButtonOptions"]["filename"] == "name"
    with pytest.raises(ValueError):
        graph_config("gif")


def test_trend_bands_have_no_markers():
    """Band traces are filled outlines only; markers would take default colours."""
    fig = trend_figure(_trend())
    bands = [t for t in fig.data if t.fill == "toself"]
    assert bands
    assert all(t.mode == "lines" for t in bands)
