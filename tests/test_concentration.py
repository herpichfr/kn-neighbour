import json

import numpy as np
import pandas as pd
import pytest
from scipy import ndimage
from scipy.spatial import cKDTree

from knn import concentration as conc
from knn.config import ConfigError, validate_config
from knn.explorer import FILTER_KEYS
from synthetic import BUILD_CONFIG, COSMO, CONC_TRUTH, bridge_curve, conc_frame

Z_CL = CONC_TRUTH["z"]
D_A = float(COSMO.angular_diameter_distance(Z_CL).value)
BLOBS = [(-5.0, 0.0, 0.8, 500), (5.0, 0.0, 0.8, 400)]
# kNN scale about 1.2 Mpc on these fields; the ridge floor is the 1 sigma level (so that the
# structures that the tests plant are joined) and a peak must rise 6 sigma above its saddle (the
# kNN map has noise bumps of its own inside a clump)
K_TEST, PROMINENCE = 20, 6.0


def _run(df, mask=None, section=None, **overrides):
    overrides = {"weight": "count", "k": K_TEST, "ridge_floor_sigma": 1.0,
                 "min_prominence_sigma": PROMINENCE, **overrides}
    settings = conc.conc_settings({"concentration": section or {}}, **overrides)
    sample = np.ones(len(df), dtype=bool) if mask is None else mask
    return conc.compute_concentration(df, sample, settings, COSMO, Z_CL)


def _cell_xy(result):
    """(X, Y) of the cell centres (Mpc), same shape as the map."""
    g = result["grid"]
    ny, nx = g["shape"]
    return np.meshgrid(g["x0"] + (np.arange(nx) + 0.5) * g["pixel"],
                       g["y0"] + (np.arange(ny) + 0.5) * g["pixel"])


def _border_distance(result, points):
    """Distance (Mpc) of points inside the survey hull to its border (< 0 outside)."""
    eq = result["hull"]["equations"]
    return -(np.atleast_2d(points) @ eq[:, :2].T + eq[:, 2]).max(axis=1)


def _near(peaks, x, y, radius):
    return [p for p in peaks if np.hypot(p["x"] - x, p["y"] - y) < radius]


def _sky(x, y):
    """(ra, dec) of tangent-plane positions in Mpc."""
    return conc.inverse_tangent_plane(np.asarray(x, dtype=float) / D_A,
                                      np.asarray(y, dtype=float) / D_A, CONC_TRUTH["ra"],
                                      CONC_TRUTH["dec"])


def _truth(ra, dec):
    """(n, 2) positions (Mpc) of sky positions in the frame of the planted field."""
    xi, eta = conc.tangent_plane(ra, dec, CONC_TRUTH["ra"], CONC_TRUTH["dec"])
    return np.column_stack([np.atleast_1d(xi) * D_A, np.atleast_1d(eta) * D_A])


def _path_spec(points, label="drawn 1"):
    ra, dec = _sky(np.asarray(points)[:, 0], np.asarray(points)[:, 1])
    return {"label": label, "ra": ra.tolist(), "dec": dec.tolist()}


def _connected(result):
    """True when the segments of the network join all its nodes (union-find on node ids)."""
    parent = {n["id"]: n["id"] for n in result["nodes"]}

    def find(a):
        while parent[a] != a:
            a = parent[a]
        return a

    for r in result["ridges"]:
        parent[find(r["nodes"][0])] = find(r["nodes"][1])
    return len({find(a) for a in parent}) == 1


def _frame_xy(df, result):
    """(n, 2) positions (Mpc) of the rows of df in the tangent plane of the map."""
    g = result["grid"]
    xi, eta = conc.tangent_plane(df["ra"].to_numpy(float), df["dec"].to_numpy(float), g["ra0"],
                                 g["dec0"])
    return np.column_stack([xi * g["d_a"], eta * g["d_a"]])


def _main(result, x, y, radius=1.6):
    """The highest significant peak within radius of (x, y): the peak of a planted clump (the kNN
    map has noise bumps of its own inside a clump)."""
    found = _near(result["peaks"], x, y, radius)
    assert found, f"no significant peak within {radius} Mpc of ({x}, {y})"
    return max(found, key=lambda p: p["height"])


def _joined(result, a, b):
    """True when the network joins the peaks a and b (dicts of result['peaks'])."""
    node = {n["peak"]: n["id"] for n in result["nodes"] if n["peak"] is not None}
    parent = {n["id"]: n["id"] for n in result["nodes"]}

    def find(i):
        while parent[i] != i:
            i = parent[i]
        return i

    for r in result["ridges"]:
        parent[find(r["nodes"][0])] = find(r["nodes"][1])
    return a["id"] in node and b["id"] in node and find(node[a["id"]]) == find(node[b["id"]])


@pytest.fixture(scope="module")
def bridged():
    df = conc_frame(blobs=BLOBS, bridges=[(-5.0, 0.0, 5.0, 0.0, 2.0, 0.5, 500)])
    return (df,) + _run(df)


@pytest.fixture(scope="module")
def valley():
    df = conc_frame(blobs=[(-7.0, 0.0, 0.8, 500), (7.0, 0.0, 0.8, 400)])
    return (df,) + _run(df)


def test_cells_far_from_the_slice_are_dark_but_positive_and_cells_outside_the_hull_are_nan():
    # the galaxies of the slice (ra < 181) fill half of the disc; the other half of the
    # catalogue is at another redshift and only sets the footprint
    df = conc_frame(n_field=3000, seed=3)
    sample = (df["ra"] < CONC_TRUTH["ra"]).to_numpy()
    df.loc[~sample, "z_best"] = 0.03
    _, result = _run(df, sample)
    dens, footprint = result["dens"], result["footprint"]
    X, Y = _cell_xy(result)
    far = footprint & (X > 5.0)                          # farther than 5 Mpc from every galaxy of the slice
    assert far.sum() > 100
    assert np.isfinite(dens[footprint]).all() and (dens[footprint] > 0).all()   # a k-th neighbour always exists
    assert dens[far].mean() < 0.2 * dens[footprint & (X < -3.0)].mean()          # the darkest cells
    assert (result["rk"][far] > 5.0).all()
    assert np.isnan(dens[~footprint]).all() and (~footprint).sum() > 100
    assert np.isnan(result["rk"][~footprint]).all() and np.isnan(result["cover"][~footprint]).all()
    assert np.isnan(dens[0, 0]) and np.isnan(dens[-1, -1]) and not footprint[0].any()
    # the footprint is the hull of all the rows: the same for another slice of the same table
    # (the cell is h / 4 and h depends on the slice, so give both the same cell)
    pixel = result["grid"]["pixel"]
    _, same = _run(df, sample, pixel_mpc=pixel)
    _, other = _run(df, ~sample, pixel_mpc=pixel)
    np.testing.assert_array_equal(other["footprint"], same["footprint"])
    np.testing.assert_array_equal(same["footprint"], footprint)
    np.testing.assert_array_equal(other["hull"]["vertices"], result["hull"]["vertices"])
    assert other["info"]["hull_area_mpc2"] == result["info"]["hull_area_mpc2"]
    assert result["info"]["n_rows"] == len(df) and result["info"]["n_sample"] == sample.sum()
    assert footprint.sum() * result["grid"]["pixel"] ** 2 == pytest.approx(
        result["info"]["hull_area_mpc2"], rel=0.03)
    rows, cols = np.nonzero(footprint)
    assert conc.in_hull(result["hull"], X[rows, cols], Y[rows, cols]).all()
    assert not conc.in_hull(result["hull"], X[~footprint], Y[~footprint]).any()
    # the mean and the standard deviation are those over the footprint
    assert result["mean"] == pytest.approx(dens[footprint].mean())
    assert result["sigma"] == pytest.approx(dens[footprint].std())


def test_density_is_flat_up_to_the_border_of_the_hull():
    rng = np.random.default_rng(3)
    n, radius, k = 6000, 10.0, 20
    r, t = radius * np.sqrt(rng.uniform(0, 1, n)), rng.uniform(0, 2 * np.pi, n)
    x, y = r * np.cos(t), r * np.sin(t)
    hull = conc.survey_hull(x, y)
    grid = conc.knn_map(x, y, np.ones(n), hull, k, 0.25)
    dens, footprint = grid["dens"], grid["footprint"]
    level = n / hull["area"] * k / (k - 1.0)              # k / (pi r_k^2) is high by k / (k - 1)
    X, Y = np.meshgrid(grid["x0"] + (np.arange(dens.shape[1]) + 0.5) * 0.25,
                       grid["y0"] + (np.arange(dens.shape[0]) + 0.5) * 0.25)
    border = _border_distance({"hull": hull}, np.column_stack([X[footprint], Y[footprint]]))
    values, cover = dens[footprint], grid["cover"][footprint]
    assert values[border > 3.0].mean() == pytest.approx(level, rel=0.05)
    near = (border > 0.5) & (border < 1.5)               # within one neighbour distance of the border
    assert values[near].mean() == pytest.approx(level, rel=0.1)
    assert values[border <= 0.5].mean() == pytest.approx(level, rel=0.15)
    # without the correction the disc cut by the border would count too few galaxies
    assert (values * cover)[border <= 0.5].mean() < 0.8 * level
    assert (cover[border > 3.0] == 1.0).all() and cover[border <= 0.5].mean() < 0.8
    # a thin sliver: the cut disc is floored, the map stays finite and at most 5 times the
    # uncorrected one
    sx, sy = rng.uniform(0, 40, 4000), rng.uniform(-0.25, 0.25, 4000)
    sliver = conc.knn_map(sx, sy, np.ones(4000), conc.survey_hull(sx, sy), 10, 0.5)
    assert conc.COVER_FLOOR == 0.2
    fp = sliver["footprint"]
    assert np.isfinite(sliver["dens"][fp]).all()
    plain = 10 / (np.pi * sliver["rk"][fp] ** 2)
    assert (sliver["dens"][fp] <= plain / conc.COVER_FLOOR * (1 + 1e-9)).all()
    assert (sliver["dens"][fp] >= plain * (1 - 1e-9)).all()


def test_the_disc_fraction_inside_the_hull_matches_the_circular_segment():
    square = np.array([[-50.0, -50.0], [50.0, -50.0], [50.0, 50.0], [-50.0, 50.0]])
    hull = conc.survey_hull(square[:, 0], square[:, 1])
    r = 3.0
    for d in (0.0, 1.0, 2.5, 3.5, 20.0):                 # distance of the centre to the right edge
        f = conc.disc_hull_fraction(hull, np.array([50.0 - d]), np.array([0.0]), np.array([r]))[0]
        cut = r ** 2 * np.arccos(d / r) - d * np.sqrt(r ** 2 - d ** 2) if d < r else 0.0
        assert f == pytest.approx(1.0 - cut / (np.pi * r ** 2), abs=0.02)
    corner = conc.disc_hull_fraction(hull, np.array([50.0]), np.array([50.0]), np.array([r]))
    assert corner[0] == pytest.approx(0.25, abs=0.02)
    # a disc whose centre is farther than r from every edge is not sampled: exactly 1
    inside = conc.disc_hull_fraction(hull, np.array([0.0, 40.0]), np.array([0.0, 0.0]),
                                     np.array([r, r]))
    np.testing.assert_array_equal(inside, [1.0, 1.0])


def test_the_knn_map_is_the_weight_of_the_k_nearest_over_the_disc_area(monkeypatch):
    monkeypatch.setattr(conc, "SUBSAMPLES", 1)            # a cell is its centre: the exact formula
    df = conc_frame(blobs=[(2.0, 1.0, 0.6, 500)], seed=7)
    k = 15
    sample = np.ones(len(df), dtype=bool)
    for weight in ("count", "mass"):
        _, result = _run(df, k=k, weight=weight)
        w = conc.galaxy_weights(df, sample, weight, 95.0)[0]
        xy = _frame_xy(df, result)
        X, Y = _cell_xy(result)
        fp = result["footprint"]
        cells = np.column_stack([X[fp], Y[fp]])
        dist, idx = cKDTree(xy).query(cells, k=k)
        rk = dist[:, -1]
        np.testing.assert_allclose(result["rk"][fp], rk)
        away = _border_distance(result, cells) > rk       # the disc is inside the survey
        assert away.sum() > 1000 and (result["cover"][fp][away] == 1.0).all()
        np.testing.assert_allclose(result["dens"][fp][away],
                                   w[idx].sum(axis=1)[away] / (np.pi * rk[away] ** 2), rtol=1e-12)
        if weight == "count":
            np.testing.assert_allclose(result["dens"][fp][away], k / (np.pi * rk[away] ** 2),
                                       rtol=1e-12)
        # near the border: corrected by the fraction of the disc inside the hull, at most 5 times
        cut = ~away
        assert cut.sum() > 100 and (result["cover"][fp][cut] < 1.0).any()
        floor = np.maximum(result["cover"][fp][cut], conc.COVER_FLOOR)
        np.testing.assert_allclose(result["dens"][fp][cut],
                                   w[idx].sum(axis=1)[cut] / (np.pi * rk[cut] ** 2 * floor),
                                   rtol=1e-12)
        ratio = result["dens"][fp] * np.pi * rk ** 2 / w[idx].sum(axis=1)
        assert ratio.max() <= 1.0 / conc.COVER_FLOOR + 1e-9 and ratio.min() >= 1.0 - 1e-9
        info = result["info"]
        h = float(np.median(cKDTree(xy[sample]).query(xy[sample], k=k + 1)[0][:, k]))
        assert info["k"] == k and info["knn_scale_mpc"] == pytest.approx(h)
        assert info["pixel_mpc"] == pytest.approx(h / conc.PIXELS_PER_SCALE)
        assert conc.PIXELS_PER_SCALE == 2.0 and info["subsamples"] == 1
        assert (info["grid_ny"], info["grid_nx"]) == result["dens"].shape
        assert result["k"] == k and result["knn_scale"] == pytest.approx(h)
        # the dense clump is the highest peak, near its centre
        top = result["peaks"][0]
        assert np.hypot(top["x"] - 2.0, top["y"] - 1.0) < 1.0 and top["n_sigma"] > 6.0
        assert result["top"] == pytest.approx(top["height"])
    # the kNN scale depends on the positions only, not on the weight
    _, mass = _run(df, k=k, weight="mass")
    _, count = _run(df, k=k, weight="count")
    assert mass["info"]["knn_scale_mpc"] == count["info"]["knn_scale_mpc"]
    assert not np.allclose(mass["dens"] / mass["mean"], count["dens"] / count["mean"],
                           equal_nan=True)


def _random_disc(n=4000, radius=10.0, seed=3):
    rng = np.random.default_rng(seed)
    r, t = radius * np.sqrt(rng.uniform(0, 1, n)), rng.uniform(0, 2 * np.pi, n)
    return r * np.cos(t), r * np.sin(t)


def test_a_cell_is_the_mean_of_the_density_at_its_sub_points_inside_the_hull():
    x, y = _random_disc(2500, seed=4)
    hull = conc.survey_hull(x, y)
    w, k, pixel, n = np.ones(len(x)), 8, 0.9, 3
    grid = conc.knn_map(x, y, w, hull, k, pixel)             # the default is SUBSAMPLES = 3
    assert conc.SUBSAMPLES == 3 and grid["subsamples"] == 3
    centre = conc.knn_map(x, y, w, hull, k, pixel, subsamples=1)
    np.testing.assert_array_equal(grid["footprint"], centre["footprint"])
    tree = cKDTree(np.column_stack([x, y]))
    ny, nx = grid["dens"].shape
    off = ((np.arange(n) + 0.5) / n - 0.5) * pixel
    rows, cols = np.nonzero(grid["footprint"])
    border = _border_distance({"hull": hull}, np.column_stack([
        grid["x0"] + (cols + 0.5) * pixel, grid["y0"] + (rows + 0.5) * pixel]))
    edge = np.flatnonzero(border < 0.3 * pixel)[:15]         # sub-points on both sides of the border
    pick = np.concatenate([edge, np.arange(0, len(rows), len(rows) // 15)])
    assert len(edge) > 5
    for i in pick:
        cx, cy = grid["x0"] + (cols[i] + 0.5) * pixel, grid["y0"] + (rows[i] + 0.5) * pixel
        px, py = (a.ravel() for a in np.meshgrid(cx + off, cy + off))
        inside = conc.in_hull(hull, px, py)
        dist, idx = tree.query(np.column_stack([px[inside], py[inside]]), k=k)
        r = dist[:, -1]
        f = np.maximum(conc.disc_hull_fraction(hull, px[inside], py[inside], r), conc.COVER_FLOOR)
        expected = np.mean(w[idx].sum(axis=1) / (np.pi * r ** 2 * f))
        assert grid["dens"][rows[i], cols[i]] == pytest.approx(expected, rel=1e-12)
        assert grid["rk"][rows[i], cols[i]] == pytest.approx(np.mean(r), rel=1e-12)
    assert np.isfinite(grid["dens"][grid["footprint"]]).all() and np.isnan(
        grid["dens"][~grid["footprint"]]).all()
    # the average does not change the level of the map, only its roughness
    assert np.nanmean(grid["dens"]) == pytest.approx(np.nanmean(centre["dens"]), rel=0.03)
    # a cell whose centre is inside the hull but none of whose sub-points is takes the value at its
    # centre: the tip of a thin triangle (lattice centre at y = 0 and 0.05 from the tip)
    rng = np.random.default_rng(5)
    u, v = rng.uniform(0, 1, 600), rng.uniform(-1, 1, 600)
    tx, ty = np.r_[u, 0, 0, 1.0], np.r_[0.05 * (1 - u) * v, -0.05, 0.05, 0.0]
    tip = conc.survey_hull(tx, ty)
    two = conc.knn_map(tx, ty, np.ones(len(tx)), tip, 5, 0.1, subsamples=2)
    one = conc.knn_map(tx, ty, np.ones(len(tx)), tip, 5, 0.1, subsamples=1)
    rows, cols = np.nonzero(two["footprint"])
    px = two["x0"] + (cols[:, None] + 0.5) * 0.1 + np.array([-0.025, 0.025, -0.025, 0.025])
    py = two["y0"] + (rows[:, None] + 0.5) * 0.1 + np.array([-0.025, -0.025, 0.025, 0.025])
    none = ~conc.in_hull(tip, px, py).any(axis=1)
    assert none.any() and (~none).any()
    assert np.isfinite(two["dens"][two["footprint"]]).all()
    np.testing.assert_allclose(two["dens"][rows[none], cols[none]],
                               one["dens"][rows[none], cols[none]])


def test_averaging_the_sub_points_smooths_the_map_at_the_same_pixel(monkeypatch):
    x, y = _random_disc()
    hull = conc.survey_hull(x, y)
    maps = {n: conc.knn_map(x, y, np.ones(len(x)), hull, 10, 0.4, subsamples=n) for n in (1, 3)}

    def roughness(grid):          # the spread of the difference between neighbouring cells
        d, ok = grid["dens"], grid["footprint"]
        pair = ok[:, 1:] & ok[:, :-1]
        return np.std((d[:, 1:] - d[:, :-1])[pair])

    assert roughness(maps[3]) < 0.9 * roughness(maps[1])
    assert np.nanstd(maps[3]["dens"]) < np.nanstd(maps[1]["dens"])
    # compute_concentration uses the constant, and refuses too many evaluations
    df = conc_frame(blobs=BLOBS)
    _, three = _run(df)
    monkeypatch.setattr(conc, "SUBSAMPLES", 1)
    _, one = _run(df)
    assert three["info"]["subsamples"] == 3 and one["info"]["subsamples"] == 1
    monkeypatch.setattr(conc, "SUBSAMPLES", 3)
    monkeypatch.setattr(conc, "MAX_EVALUATIONS", 1000)
    with pytest.raises(conc.ConcentrationError, match="density evaluations"):
        _run(df)
    with pytest.raises(conc.ConcentrationError, match="subsamples"):
        conc.knn_map(x, y, np.ones(len(x)), hull, 10, 0.4, subsamples=0)


def test_k_is_an_integer_between_2_and_the_sample_size_minus_1():
    df = conc_frame(n_field=400, seed=5)
    n = len(df)
    assert conc.CONC_DEFAULTS["k"] == 10 and conc.CONC_DEFAULTS["ridge_floor_sigma"] == 5.0
    assert "bandwidth_mpc" not in conc.CONC_DEFAULTS
    _, result = _run(df, k=10.0)                         # a whole float is accepted
    assert result["info"]["k"] == 10 and isinstance(result["info"]["settings"]["k"], int)
    _run(df, k=2)
    _run(df, k=n - 1, pixel_mpc=2.0)
    for bad in (1, 0, -3, n, n + 5, 7.5, "ten", True):
        with pytest.raises(conc.ConcentrationError, match="k "):
            _run(df, k=bad)
    sub = np.arange(n) < 40
    with pytest.raises(conc.ConcentrationError, match="between 2 and 39"):
        _run(df, sub, k=40)
    _, ok = _run(df, sub, k=39)
    assert ok["info"]["n_sample"] == 40


def test_a_contour_that_reaches_the_border_stays_open():
    df = conc_frame(blobs=[(-3.0, 0.0, 1.0, 500), (14.0, 0.0, 2.0, 700)])
    columns, result = _run(df)
    pixel = result["grid"]["pixel"]
    first = result["contours"][0]
    assert first["n_sigma"] == 1.0 and first["level"] == pytest.approx(
        result["mean"] + result["sigma"])
    assert len(first["lines"]) == len(first["closed"])
    open_lines = [ln for ln, closed in zip(first["lines"], first["closed"]) if not closed]
    closed_lines = [ln for ln, closed in zip(first["lines"], first["closed"]) if closed]
    assert len(open_lines) >= 1 and len(closed_lines) >= 1
    for line in open_lines:                              # ends on the border, never along it
        ends = _border_distance(result, np.array([line[0], line[-1]]))
        assert (ends >= -1e-9).all() and (ends <= 1.5 * pixel).all()
        assert np.hypot(*(line[0] - line[-1])) > 1.0
        assert abs(np.mean([line[:, 0].max(), line[:, 0].min()]) - 12.0) < 4.0   # the edge blob
    for line in closed_lines:                            # a loop stays inside the survey
        assert np.allclose(line[0], line[-1]) and _border_distance(result, line).min() > 0
    assert any(np.mean(line[:, 0]) < 0.0 for line in closed_lines)      # the central blob
    for c in result["contours"]:                         # no line leaves the survey area
        for line in c["lines"]:
            assert conc.in_hull(result["hull"], line[:, 0], line[:, 1]).all()
    # the same holds on the sky, and the edge peak is flagged open
    sky = conc.contour_table(result)[0]["lines"]
    assert [ln["closed"] for ln in sky] == first["closed"]
    assert _main(result, 13.5, 0.0, 3.0)["open"]
    assert not _main(result, -3.0, 0.0)["open"]


def test_two_peaks_joined_by_a_bridge_give_one_ridge_along_its_crest(bridged):
    df, columns, result = bridged
    assert len(result["peaks"]) == 2
    left, right = _main(result, -5.0, 0.0), _main(result, 5.0, 0.0)
    assert [p["id"] for p in result["peaks"]] == [1, 2] and left["height"] > right["height"]
    assert len(result["ridges"]) == 1
    ridge = result["ridges"][0]
    assert ridge["kind"] == "peak" and not ridge["open"]
    assert ridge["peaks"] == [2, 1]                      # lower end first
    assert ridge["saddle"] >= result["level"] and ridge["saddle"] >= result["floor_level"]
    assert ridge["saddle_n_sigma"] == pytest.approx(
        (ridge["saddle"] - result["mean"]) / result["sigma"])
    assert ridge["component"] == 1 and result["info"]["n_ridge_components"] == 1
    assert [n["kind"] for n in result["nodes"]] == ["peak", "peak"] and ridge["nodes"] == [2, 1]
    assert [p["degree"] for p in result["peaks"]] == [1, 1] and ridge["cells"][0] == right["cell"]
    pts = ridge["points"]
    arc = bridge_curve(-5.0, 0.0, 5.0, 0.0, 2.0, np.linspace(0, 1, 2000))
    gap = np.array([np.hypot(*(arc - p).T).min() for p in pts])
    assert gap.max() < 1.0 and np.median(gap) < 0.5      # on the crest, not on the straight line
    assert np.abs(pts[:, 1]).max() > 1.2                 # the bridge bows by 2 Mpc, the ridge follows
    assert pts[:, 0].min() < -4.0 and pts[:, 0].max() > 4.0
    ends = np.array([pts[0], pts[-1]])
    assert min(np.hypot(*(ends - [left["x"], left["y"]]).T)) < 1e-9
    assert min(np.hypot(*(ends - [right["x"], right["y"]]).T)) < 1e-9
    assert 10.0 < ridge["length"] < 12.5
    info = result["info"]
    assert info["n_ridges"] == 1 and info["n_ridges_open"] == 0
    assert info["n_junctions"] == 0 and info["n_border_ends"] == 0
    assert info["ridges_total_length_mpc"] == pytest.approx(ridge["length"])
    # the galaxies of the bridge are near the ridge
    assert np.median(columns["conc_ridge_dist_mpc"][-500:]) < 0.6
    table = conc.ridge_table(result)
    assert json.dumps(table) and table[0]["saddle_n_sigma"] >= 1.0 and len(table[0]["ra"]) == len(pts)
    assert table[0]["peaks"] == [2, 1] and table[0]["nodes"] == [2, 1]
    nodes = conc.node_table(result)
    assert json.dumps(nodes) and [n["peak"] for n in nodes] == [1, 2]
    assert [n["degree"] for n in nodes] == [1, 1]
    assert {"id", "kind", "peak", "ra", "dec", "x_mpc", "y_mpc", "n_sigma", "degree"} == set(nodes[0])


def test_two_peaks_separated_by_a_deep_valley_are_not_joined(valley):
    df, columns, result = valley
    left, right = _main(result, -7.0, 0.0), _main(result, 7.0, 0.0)
    assert not _joined(result, left, right)
    # they are in separate 1 sigma regions, so no ridge above a floor of 1 sigma or more runs
    # from one to the other
    assert result["info"]["n_regions"] == 2
    assert left["height"] > result["level"] and right["height"] > result["level"]
    assert left["prominence_sigma"] > 1.0 and right["prominence_sigma"] > 1.0
    for ridge in result["ridges"]:
        x = ridge["points"][:, 0]
        assert (x < 0).all() or (x > 0).all()
    assert result["info"]["n_ridge_components"] <= 2


def test_a_bump_below_the_minimum_prominence_is_not_a_significant_peak():
    # the kNN map of a broad cluster has noise bumps: local maxima above the 1 sigma level that
    # rise only slightly above their saddle
    df = conc_frame(blobs=[(0.0, 0.0, 2.5, 1500)])
    _, result = _run(df, min_prominence_sigma=1.0)
    bumps = [p for p in result["all_peaks"] if not p["significant"] and p["height"] >= result["level"]]
    assert bumps and all(p["prominence_sigma"] < 1.0 and p["id"] == 0 for p in bumps)
    assert all(p["id"] > 0 for p in result["peaks"])
    assert result["peaks"][0]["id"] == 1 and result["peaks"][0]["height"] == result["top"]
    assert result["info"]["n_peaks"] == len(result["peaks"]) < result["info"]["n_local_maxima"]
    assert all(p["prominence_sigma"] >= 1.0 for p in result["peaks"][1:])
    # with no prominence required they are marked
    _, loose = _run(df, min_prominence_sigma=0.0)
    assert len(loose["peaks"]) >= len(result["peaks"]) + len(bumps)
    # and with a high requirement only the global maximum is left, whatever its prominence
    _, strict = _run(df, min_prominence_sigma=100.0)
    assert len(strict["peaks"]) == 1 and strict["peaks"][0]["height"] == result["top"]


def test_peak_tree_prominence_saddles_and_tree_edges():
    row = np.array([9.0, 5.0, 4.0, 6.0, 3.0, 8.0, 1.0])
    dens = np.pad(row[None, :], 1, constant_values=np.nan)
    footprint = np.pad(np.ones((1, 7), dtype=bool), 1, constant_values=False)
    tree = conc.merge_tree(dens, footprint, level=0.0)
    np.testing.assert_array_equal(tree["height"], [9.0, 8.0, 6.0])
    np.testing.assert_array_equal(tree["parent"], [-1, 0, 0])
    np.testing.assert_array_equal(tree["saddle"][1:], [3.0, 4.0])
    assert np.isnan(tree["saddle"][0])
    np.testing.assert_array_equal(tree["prominence"], [8.0, 5.0, 2.0])   # 9 - 1, 8 - 3, 6 - 4
    assert len(tree["edges"][0]) == 6                    # 7 cells in one tree
    cols = (tree["cell"] % dens.shape[1]) - 1
    np.testing.assert_array_equal(cols, [0, 5, 3])
    # the tree edges join neighbouring cells only
    a, b = tree["edges"]
    assert (np.abs(a - b) == 1).all()
    with pytest.raises(ValueError, match="edge of the array"):
        conc.merge_tree(row[None, :], np.ones((1, 7), dtype=bool), 0.0)
    # 8-connected: two cells touching at a corner are one component
    dens = np.full((6, 6), np.nan)
    foot = np.zeros((6, 6), dtype=bool)
    dens[2, 2], dens[3, 3] = 5.0, 4.0
    foot[2, 2] = foot[3, 3] = True
    tree = conc.merge_tree(dens, foot, 0.0)
    assert len(tree["height"]) == 1 and tree["height"][0] == 5.0


def test_peak_flag_open_follows_the_region_reaching_the_border():
    yy, xx = np.mgrid[0:13, 0:13]
    dens = np.exp(-0.5 * ((yy - 6.0) ** 2 + (xx - 6.0) ** 2) / 4.0)
    footprint = np.zeros(dens.shape, dtype=bool)
    footprint[1:-1, 1:-1] = True                         # the border of the footprint is its outer ring
    dens = np.where(footprint, dens, np.nan)
    closed = conc.merge_tree(dens, footprint, level=0.5)
    assert closed["open"].tolist() == [False]            # the region above 0.5 is a disc inside
    reaching = conc.merge_tree(dens, footprint, level=0.0)
    assert reaching["open"].tolist() == [True]           # it covers the whole footprint
    # a second peak: its region ends at a saddle above the level, so its own region is small
    dens2 = dens.copy()
    dens2[6, 10] = 0.9
    tree = conc.merge_tree(dens2, footprint, level=0.0)
    assert tree["open"].tolist() == [True, False] and tree["saddle"][1] > 0.0


def test_contour_levels_step_and_thinning():
    n, levels = conc.contour_levels(10.0, 2.0, 10.0 + 2.0 * 5.5, 1.0)
    np.testing.assert_allclose(n, [1, 2, 3, 4, 5])
    np.testing.assert_allclose(levels, 10.0 + 2.0 * n)
    n, _ = conc.contour_levels(0.0, 1.0, 3.2, 0.5)
    np.testing.assert_allclose(n, [1.0, 1.5, 2.0, 2.5, 3.0])
    n, _ = conc.contour_levels(0.0, 1.0, 1.0, 1.0)
    np.testing.assert_allclose(n, [1.0])                 # the top exactly at the 1 sigma level
    n, levels = conc.contour_levels(0.0, 1.0, 0.9, 1.0)
    assert len(n) == 0 and len(levels) == 0              # a map that never reaches 1 sigma
    n, _ = conc.contour_levels(0.0, 1.0, 30.5, 1.0)      # 30 levels thinned to 12
    assert len(n) == conc.MAX_LEVELS == 12 and n[0] == 1.0 and n[-1] == 30.0
    assert (np.diff(n) >= 2.0).all() and (np.diff(n) <= 3.0).all()   # evenly spread
    n, _ = conc.contour_levels(0.0, 1.0, 12.5, 1.0)
    assert len(n) == 12                                  # exactly 12: all kept
    n, _ = conc.contour_levels(0.0, 1.0, 13.5, 1.0)
    assert len(n) == 12 and n[0] == 1.0 and n[-1] == 13.0


def test_a_sharp_cluster_gets_at_most_twelve_levels_that_include_the_first_and_last():
    df = conc_frame(blobs=[(0.0, 0.0, 0.3, 1500)])
    _, result = _run(df, section={"contour_step_sigma": 1.0})
    info = result["info"]
    assert info["top_n_sigma"] > 14.0
    n = np.array(info["contour_levels_n_sigma"])
    assert len(n) == 12 and n[0] == 1.0 and (np.diff(n) > 0).all()
    assert n[-1] == np.floor(info["top_n_sigma"])        # the highest level below the maximum
    assert len(result["contours"]) == 12 and info["level_1sigma"] == result["contours"][0]["level"]
    _, coarse = _run(df, contour_step_sigma=5.0)
    np.testing.assert_allclose(coarse["info"]["contour_levels_n_sigma"][:3], [1.0, 6.0, 11.0])


def test_inside_flag_is_the_density_at_or_above_the_1_sigma_level(bridged):
    df, columns, result = bridged
    level = result["info"]["level_1sigma"]
    assert level == pytest.approx(result["mean"] + result["sigma"])
    density, inside = columns["conc_density"], columns["conc_inside"]
    np.testing.assert_array_equal(inside, density >= level)
    assert inside.dtype == bool and 0.05 < inside.mean() < 0.5
    np.testing.assert_allclose(columns["conc_nsigma"], (density - result["mean"]) / result["sigma"])
    assert (columns["conc_nsigma"][inside] >= 1.0 - 1e-12).all()
    assert (columns["conc_nsigma"][~inside] < 1.0).all()
    region = columns["conc_region"]
    assert ((region >= 0) == inside).all() and region.dtype.kind == "i"
    assert set(np.unique(region[inside])) == set(range(result["info"]["n_regions"]))
    # region 0 holds the highest peak; the bridge joins both blobs in it
    assert (region[-500:][inside[-500:]] == 0).all() and result["info"]["n_regions"] >= 1
    assert np.isfinite(density).all() and np.isfinite(columns["conc_ridge_dist_mpc"]).all()
    assert set(columns) == set(conc.COLUMNS) and all(len(v) == len(df) for v in columns.values())
    c = result["info"]["counts"]
    assert c["inside_spec"] == inside.sum() and c["inside_phot"] == 0
    assert c["outside_spec"] == (~inside).sum()


def test_columns_cover_the_rows_outside_the_sample(bridged):
    df = bridged[0]
    half = np.arange(len(df)) % 2 == 0
    columns, result = _run(df, half)
    assert result["info"]["n_sample"] == half.sum()
    assert np.isfinite(columns["conc_density"]).all()           # every row has a map value
    inside = columns["conc_inside"]
    assert result["info"]["counts"]["inside_spec"] == (inside & half).sum()
    assert result["info"]["counts"]["outside_spec"] == (~inside & half).sum()


def test_no_ridge_runs_below_the_floor_and_a_valley_below_it_cuts_the_network(bridged):
    df, columns, base = bridged
    saddle = base["ridges"][0]["saddle_n_sigma"]          # the crest of the bridge, in sigma
    assert 1.0 <= saddle < 4.0
    left, right = _main(base, -5.0, 0.0), _main(base, 5.0, 0.0)
    for floor in (1.0, saddle - 0.05):
        _, result = _run(df, ridge_floor_sigma=floor)
        level = result["mean"] + floor * result["sigma"]
        assert result["floor_level"] == pytest.approx(level)
        assert result["info"]["ridge_floor_sigma"] == floor
        assert result["info"]["ridge_floor_level"] == pytest.approx(level)
        assert len(result["ridges"]) == 1
        for ridge in result["ridges"]:
            assert (result["dens"].ravel()[ridge["cells"]] >= level - 1e-9).all()
            assert ridge["saddle"] >= level - 1e-9 and ridge["saddle_n_sigma"] >= floor - 1e-9
        assert all(n["height"] >= level - 1e-9 for n in result["nodes"])
    # two peaks that are both above the floor but joined only through ground below it: no ridge
    for floor in (saddle + 0.05, 3.0, 5.0):
        _, high = _run(df, ridge_floor_sigma=floor)
        level = high["mean"] + floor * high["sigma"]
        assert _main(high, -5.0, 0.0)["height"] > level and _main(high, 5.0, 0.0)["height"] > level
        assert high["ridges"] == [] and high["nodes"] == []
        assert all(p["degree"] == 0 for p in high["peaks"])
        assert high["info"]["n_ridges"] == 0 and high["info"]["n_ridge_components"] == 0
        assert high["info"]["n_peaks_above_floor"] >= 2
        assert np.isnan(_run(df, ridge_floor_sigma=floor)[0]["conc_ridge_dist_mpc"]).all()
    # a drawn path in the model is not subject to the floor: it follows the crest below it
    spec = {"paths": [_path_spec(bridge_curve(-5.0, 0.0, 5.0, 0.0, 2.0, np.linspace(0, 1, 11)))]}
    path = conc.compute_model(df, high, spec)[1]["paths"][0]
    assert path["crest_n_sigma_min"] < 5.0 and path["length_mpc"] > 9.0


def test_the_network_is_a_forest_one_tree_per_piece_of_ground_above_the_floor():
    df = conc_frame(blobs=[(-8.0, -6.0, 0.8, 500), (6.0, -6.0, 0.8, 450), (-8.0, 7.0, 0.8, 500),
                           (6.0, 7.0, 0.8, 450), (0.0, 0.5, 0.8, 400)],
                    bridges=[(-8.0, -6.0, 6.0, -6.0, 0.0, 0.4, 800),
                             (-8.0, 7.0, 6.0, 7.0, 0.0, 0.4, 800)])
    _, result = _run(df)
    a, b = _main(result, -8.0, -6.0), _main(result, 6.0, -6.0)
    c, d = _main(result, -8.0, 7.0), _main(result, 6.0, 7.0)
    lone = _main(result, 0.0, 0.5)
    assert _joined(result, a, b) and _joined(result, c, d)
    assert not _joined(result, a, c) and not _joined(result, b, d) and not _joined(result, a, d)
    assert result["info"]["n_ridge_components"] == 2 and len(result["ridges"]) == 2
    assert sorted(r["component"] for r in result["ridges"]) == [1, 2]
    assert lone["degree"] == 0 and lone["cell"] not in {n["cell"] for n in result["nodes"]}
    assert len(result["ridges"]) == len(result["nodes"]) - 2       # a forest of two trees
    # no ridge from or to the peak that is alone above the floor, whatever the others do
    assert all(lone["id"] not in r["peaks"] for r in result["ridges"])


def test_a_border_end_is_the_highest_cell_of_a_border_run_at_or_above_the_floor():
    df = conc_frame(blobs=[(-5.0, 0.0, 0.8, 500)], bridges=[(-5.0, 0.0, 15.5, 0.0, 0.0, 0.4, 1000)])
    _, low = _run(df, ridge_floor_sigma=1.0)
    ends = [n for n in low["nodes"] if n["kind"] == "border"]
    assert len(ends) == 1 and ends[0]["height"] >= low["floor_level"]
    border = low["footprint"] & ~ndimage.binary_erosion(low["footprint"], structure=np.ones((3, 3)),
                                                      border_value=0)
    labels, n = ndimage.label(border & (np.where(low["footprint"], low["dens"], -np.inf)
                                        >= low["floor_level"]), structure=np.ones((3, 3)))
    row, col = divmod(ends[0]["cell"], low["dens"].shape[1])
    assert labels[row, col] > 0
    assert ends[0]["height"] == pytest.approx(np.max(low["dens"][labels == labels[row, col]]))
    # above the crest of the bridge at the border no border cell is at the floor: no border end
    # and no ridge, though the peak is far above the floor
    _, high = _run(df, ridge_floor_sigma=ends[0]["n_sigma"] + 0.5)
    assert high["info"]["n_border_ends"] == 0 and high["ridges"] == []
    assert high["peaks"][0]["height"] > high["floor_level"] and high["peaks"][0]["degree"] == 0


def test_the_floor_defaults_to_5_sigma_and_must_be_at_least_1_sigma():
    df = conc_frame(blobs=BLOBS, bridges=[(-5.0, 0.0, 5.0, 0.0, 2.0, 0.5, 500)])
    _, result = _run(df, ridge_floor_sigma=None)           # None = the default
    assert result["info"]["ridge_floor_sigma"] == 5.0
    assert result["floor_level"] == pytest.approx(result["mean"] + 5.0 * result["sigma"])
    assert conc.conc_settings({})["ridge_floor_sigma"] == 5.0
    assert conc.conc_settings({"concentration": {"ridge_floor_sigma": 2}})["ridge_floor_sigma"] == 2
    assert conc.conc_settings({}, ridge_floor_sigma=3.0)["ridge_floor_sigma"] == 3.0
    for bad in (0.5, 0.0, -1.0, float("nan")):
        with pytest.raises(conc.ConcentrationError, match="ridge_floor_sigma"):
            _run(df, ridge_floor_sigma=bad)
    _run(df, ridge_floor_sigma=1.0)
    assert "ridges (floor 5 sigma, " in conc.describe(result["info"])[2]


def test_a_ridge_from_a_peak_to_the_border_is_open():
    df = conc_frame(blobs=[(6.0, 0.0, 1.0, 700)], bridges=[(6.0, 0.0, 15.5, 0.0, 0.0, 0.6, 300)])
    columns, result = _run(df)
    peak = result["peaks"][0]
    assert peak["open"] and len(result["peaks"]) == 1
    assert [r["kind"] for r in result["ridges"]] == ["boundary"]
    ridge = result["ridges"][0]
    assert ridge["open"] and ridge["peaks"] == [None, 1] and ridge["saddle"] >= result["level"]
    assert [n["kind"] for n in result["nodes"]] == ["peak", "border"] and ridge["nodes"] == [2, 1]
    assert result["info"]["n_border_ends"] == 1 and result["info"]["n_junctions"] == 0
    pts = ridge["points"]                                # from the border end to the peak
    assert np.hypot(pts[-1, 0] - peak["x"], pts[-1, 1] - peak["y"]) < 1e-9
    assert _border_distance(result, pts[0])[0] <= 1.5 * result["grid"]["pixel"]
    assert pts[0, 0] > 12.0 and np.abs(pts[:, 1]).max() < 1.0       # along the bridge, to the border
    # the end is the highest cell of its component of the border at or above the 1 sigma level
    info = result["info"]
    assert info["n_ridges"] == 1 and info["n_ridges_open"] == 1
    assert info["peaks"][0]["open"] is True
    assert np.nanmedian(columns["conc_ridge_dist_mpc"][-300:]) < 0.6
    assert conc.ridge_table(result)[0]["open"] is True and peak["degree"] == 1
    end = result["nodes"][1]
    assert end["peak"] is None and end["degree"] == 1 and end["height"] >= result["level"]
    # no peak within BORDER_PEAK_CELLS of the end (otherwise the peak would be the open end)
    reach = conc.BORDER_PEAK_CELLS * result["grid"]["pixel"]
    assert max(abs(end["x"] - peak["x"]), abs(end["y"] - peak["y"])) > reach


def test_a_field_without_structure_has_a_lower_top_than_a_cluster():
    df = conc_frame(n_field=3000, seed=6)
    _, result = _run(df, k=40)
    cluster = _run(conc_frame(blobs=[(0.0, 0.0, 0.3, 1500)]), k=40)[1]
    assert result["info"]["top_n_sigma"] < 0.5 * cluster["info"]["top_n_sigma"]   # noise only
    assert result["peaks"][0]["id"] == 1 and result["peaks"][0]["height"] == result["top"]
    assert "Peaks: " in conc.describe(result["info"])[2]
    # only the global maximum marked: the network joins the high border cells of the noise
    # that lie in the same piece of ground above the floor
    _, single = _run(df, k=40, min_prominence_sigma=100.0)
    assert len(single["peaks"]) == 1
    assert all(r["open"] == (r["kind"] == "boundary") for r in single["ridges"])
    pieces = single["info"]["n_ridge_components"]
    assert len(single["ridges"]) == len(single["nodes"]) - pieces      # a forest
    assert single["info"]["n_border_ends"] >= 2 * pieces


def test_a_single_peak_without_a_border_end_has_no_ridge():
    df = conc_frame(blobs=[(0.0, 0.0, 1.0, 600)])
    columns, single = _run(df)
    assert len(single["peaks"]) == 1 and single["ridges"] == [] and single["peaks"][0]["degree"] == 0
    assert single["nodes"] == []                         # a peak with no ridge is not a node
    assert np.isnan(columns["conc_ridge_dist_mpc"]).all()
    info = single["info"]
    assert info["n_ridges"] == 0 and info["ridges_total_length_mpc"] == 0.0
    assert info["n_junctions"] == 0 and info["n_border_ends"] == 0 and info["n_ridge_components"] == 0
    assert "0 segments in 0 pieces, 0 branch points, 0 open" in conc.describe(info)[2]
    assert conc.ridge_table(single) == [] and conc.node_table(single) == []


def test_peaks_in_separate_regions_get_no_ridge():
    # four separate clumps, no bridges: the saddles are far below the floor (1 sigma or more)
    df = conc_frame(blobs=[(-8.0, 0.0, 0.8, 500), (8.0, 0.0, 0.8, 400), (0.0, 9.0, 0.8, 350),
                           (0.0, -9.0, 0.8, 450)])
    _, result = _run(df)
    assert len(result["peaks"]) == 4 and result["info"]["n_regions"] == 4
    assert result["ridges"] == [] and result["nodes"] == []
    assert all(p["degree"] == 0 for p in result["peaks"])
    assert result["info"]["n_ridge_components"] == 0 and result["info"]["n_border_ends"] == 0


def test_a_ridge_branches_at_a_junction_or_at_a_peak():
    # a bridge from a third clump meets the bridge between two others: a branch point in the
    # crest that is not a peak, joining three peaks
    df = conc_frame(blobs=[(-8.0, 0.0, 0.8, 500), (8.0, 0.0, 0.8, 450), (0.0, 8.0, 0.8, 400)],
                    bridges=[(-8.0, 0.0, 8.0, 0.0, 0.0, 0.4, 800), (0.0, 8.0, 0.0, 0.0, 0.0, 0.4, 500)])
    _, result = _run(df)
    assert len(result["peaks"]) == 3 and [p["degree"] for p in result["peaks"]] == [1, 1, 1]
    junctions = [n for n in result["nodes"] if n["kind"] == "junction"]
    assert len(junctions) == 1 and junctions[0]["degree"] == 3 and junctions[0]["peak"] is None
    assert abs(junctions[0]["x"]) < 2.0 and abs(junctions[0]["y"]) < 3.0   # near the middle
    assert _connected(result) and len(result["ridges"]) == 3
    assert sorted(r["kind"] for r in result["ridges"]) == ["junction"] * 3
    assert all(None in r["peaks"] and junctions[0]["id"] in r["nodes"] for r in result["ridges"])
    assert result["info"]["n_junctions"] == 1 and result["info"]["n_ridge_components"] == 1
    assert "1 branch points" in conc.describe(result["info"])[2]
    # three clumps in a row: one network joins the three of them
    df = conc_frame(blobs=[(-8.0, 0.0, 0.8, 400), (0.0, 0.0, 0.8, 600), (8.0, 0.0, 0.8, 400)],
                    bridges=[(-8.0, 0.0, 0.0, 0.0, 0.0, 0.4, 500), (0.0, 0.0, 8.0, 0.0, 0.0, 0.4, 500)])
    _, row = _run(df)
    left, middle, right = (_main(row, x, 0.0, 1.5) for x in (-8.0, 0.0, 8.0))
    assert _joined(row, left, middle) and _joined(row, middle, right) and _joined(row, left, right)
    assert row["info"]["n_ridge_components"] == 1 and middle["degree"] >= 1


def test_a_bridge_that_leaves_the_map_ends_at_a_border_node_unless_a_peak_is_there():
    df = conc_frame(blobs=[(-5.0, 0.0, 0.8, 500)], bridges=[(-5.0, 0.0, 15.5, 0.0, 0.0, 0.4, 1000)])
    _, result = _run(df)
    pixel = result["grid"]["pixel"]
    border = [n for n in result["nodes"] if n["kind"] == "border"]
    assert len(border) == 1 and result["info"]["n_border_ends"] == 1 and len(result["peaks"]) == 1
    assert _border_distance(result, [[border[0]["x"], border[0]["y"]]])[0] <= 1.5 * pixel
    assert border[0]["x"] > 12.0 and [r["kind"] for r in result["ridges"]] == ["boundary"]
    # a clump in the very corner of the map: its peak is within BORDER_PEAK_CELLS of the border
    # component, it is the open end and no border node is added
    df = conc_frame(blobs=[(-3.0, 0.0, 1.0, 500), (14.8, 0.0, 1.0, 400)],
                    bridges=[(-3.0, 0.0, 14.8, 0.0, 0.0, 0.4, 1000)])
    _, edge = _run(df)
    peak = _main(edge, 13.5, 0.0, 3.0)
    assert peak["open"] and edge["info"]["n_border_ends"] == 0
    assert [r["kind"] for r in edge["ridges"]] == ["peak"] and peak["degree"] == 1


def test_weight_switch_changes_the_map_not_the_knn_scale():
    df = conc_frame(blobs=BLOBS)
    df["flag"] = np.where(df["ra"] > df["ra"].median(), 2.0, 1.0)
    _, mass = _run(df, weight="mass")
    _, count = _run(df, weight="count")
    _, column = _run(df, weight="flag")
    _, as_count = _run(df.assign(one=1.0), weight="one")
    assert mass["info"]["knn_scale_mpc"] == count["info"]["knn_scale_mpc"] == column["info"][
        "knn_scale_mpc"]
    np.testing.assert_allclose(as_count["dens"], count["dens"], equal_nan=True)
    assert not np.allclose(mass["dens"] / mass["mean"], count["dens"] / count["mean"],
                           equal_nan=True)
    assert mass["info"]["settings"]["weight"] == "mass"
    assert column["info"]["settings"]["weight"] == "flag"
    assert mass["mean"] > 1e6 > count["mean"]            # M_sun / Mpc^2 against galaxies / Mpc^2


def test_weights_and_the_cap():
    df = pd.DataFrame({"logmstar": [8.0, 9.0, 10.0, np.nan, 12.0], "p_in_window": [1, 0.5, 1, 1, 1],
                       "x": [-1.0, 0.0, 1.0, np.nan, 3.0], "z_type": "spec"})
    sample = np.array([True, True, True, True, False])
    w, info = conc.galaxy_weights(df, sample, "count")
    np.testing.assert_array_equal(w, [1, 1, 1, 1, 0])
    w, info = conc.galaxy_weights(df, sample, "mass")
    np.testing.assert_allclose(w, [1e8, 0.5e9, 1e10, 0.0, 0.0])
    assert info["clip_value"] is None and info["n_eff"] == pytest.approx(
        (1e8 + 0.5e9 + 1e10) ** 2 / (1e16 + 0.25e18 + 1e20))
    capped, info = conc.galaxy_weights(df, sample, "mass", clip=50.0)
    assert info["clip_value"] == pytest.approx(0.5e9) and capped.max() == pytest.approx(0.5e9)
    # a column: NaN count 0, negative values are shifted by the sample minimum
    w, _ = conc.galaxy_weights(df, sample, "x")
    np.testing.assert_allclose(w, [0, 1, 2, 0, 0])
    w, _ = conc.galaxy_weights(df, sample, "ignored", values=np.array([1.0, 2, 3, 4, 5]))
    np.testing.assert_allclose(w, [1, 2, 3, 4, 0])
    with pytest.raises(conc.ConcentrationError, match="not in the table"):
        conc.galaxy_weights(df, sample, "nope")
    with pytest.raises(conc.ConcentrationError, match="logmstar"):
        conc.galaxy_weights(df.drop(columns="logmstar"), sample, "mass")


def test_the_cap_stops_a_single_massive_galaxy_from_making_the_peak():
    df = conc_frame(n_field=2500, seed=4)
    df.loc[0, "logmstar"] = 13.0                          # one galaxy 10^4 times the typical mass
    top = {}
    for clip in (None, 95.0):
        _, result = _run(df, section={"weight_clip": clip}, weight="mass")
        top[clip] = result["info"]["top_n_sigma"]
    assert top[None] > top[95.0] + 5.0 and top[95.0] < 8.0


def test_the_knn_scale_is_the_median_distance_to_the_kth_neighbour_without_the_galaxy_itself():
    rng = np.random.default_rng(2)
    x, y = rng.normal(0.0, 5.0, 300), rng.normal(0.0, 3.0, 300)
    d = np.sort(np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :]), axis=1)
    for k in (2, 5, 17):
        assert conc.knn_scale(x, y, k) == pytest.approx(np.median(d[:, k]))   # d[:, 0] is the galaxy
    assert conc.knn_scale(x, y, 5) < conc.knn_scale(x, y, 17)
    df = conc_frame(blobs=BLOBS)
    _, result = _run(df)
    assert result["info"]["pixel_mpc"] == pytest.approx(
        result["info"]["knn_scale_mpc"] / conc.PIXELS_PER_SCALE)
    _, fixed = _run(df, pixel_mpc=0.4)
    assert fixed["info"]["pixel_mpc"] == 0.4 and fixed["grid"]["pixel"] == 0.4
    assert fixed["info"]["knn_scale_mpc"] == result["info"]["knn_scale_mpc"]
    assert "bandwidth" not in result["info"]
    assert not hasattr(conc, "scott_bandwidth") and not hasattr(conc, "density_map")


def test_sample_mask_defaults_and_errors(bridged):
    df = bridged[0].assign(z_type=np.where(np.arange(len(bridged[0])) % 3 == 0, "phot", "spec"))
    s = conc.conc_settings({"concentration": {"z_range": [0.019, 0.021], "sample": "spec"}})
    keep = conc.default_sample(df, s)
    z = df["z_best"].to_numpy()
    np.testing.assert_array_equal(keep, (z >= 0.019) & (z <= 0.021) & (df["z_type"] == "spec"))
    assert conc.default_sample(df, conc.conc_settings({})).all()
    assert conc.conc_settings({}, k=None, weight="count")["weight"] == "count"
    assert conc.conc_settings({}, k=None)["k"] == 10 and conc.conc_settings({}, k=7)["k"] == 7
    assert conc.conc_settings({"concentration": {"weight": "count"}})["weight"] == "count"
    with pytest.raises(KeyError, match="unknown concentration setting"):
        conc.conc_settings({}, speed=3)
    with pytest.raises(conc.ConcentrationError, match="only 3 galaxies"):
        _run(df, np.arange(len(df)) < 3)
    with pytest.raises(conc.ConcentrationError, match="all weights are zero"):
        _run(df.assign(logmstar=np.nan), weight="mass")
    with pytest.raises(conc.ConcentrationError, match="contour_step_sigma"):
        _run(df, contour_step_sigma=0.0)
    with pytest.raises(conc.ConcentrationError, match="cells"):
        _run(df, pixel_mpc=0.01)


def test_tables_sky_grid_hull_and_describe(bridged):
    df, columns, result = bridged
    contours = conc.contour_table(result)
    json.dumps(contours), json.dumps(result["info"]), json.dumps(conc.peak_table(result))
    assert [c["n_sigma"] for c in contours] == list(result["info"]["contour_levels_n_sigma"])
    assert all(len(ln["ra"]) == len(ln["dec"]) >= 4 for c in contours for ln in c["lines"])
    ra, dec, z = conc.sky_grid(result)
    assert z.shape == (len(dec), len(ra)) and np.isnan(z).any() and np.isfinite(z).any()
    assert np.nanmax(z) == pytest.approx(result["top"] / result["mean"], rel=0.1)
    assert np.nanmin(z) >= 0.0 and ra.min() > 160.0 and ra.max() < 200.0
    # NaN outside the hull only: the corners are NaN, the centre is not
    assert np.isnan(z[0, 0]) and np.isfinite(z[len(dec) // 2, len(ra) // 2])
    hull_ra, hull_dec = conc.hull_sky(result)
    assert hull_ra[0] == pytest.approx(hull_ra[-1]) and hull_dec[0] == pytest.approx(hull_dec[-1])
    xi, eta = conc.tangent_plane(hull_ra, hull_dec, result["grid"]["ra0"], result["grid"]["dec0"])
    assert conc.in_hull(result["hull"], xi * D_A, eta * D_A, tol=1e-6).all()
    lines = conc.describe(result["info"])
    assert lines[0].startswith(f"Map of {len(df)} galaxies") and "hull of" in lines[0]
    assert f"k = {K_TEST} nearest neighbours, kNN scale " in lines[0]
    assert "contour levels" in lines[1] and "Peaks: 2 significant" in lines[2]
    assert ("ridges (floor 1 sigma, 2 peaks above it): 1 segments in 1 pieces, 0 branch points, "
            "0 open to the border") in lines[2]
    assert "Inside the 1 sigma contour" in lines[3]
    # the coordinate transforms
    assert conc.inverse_tangent_plane(0.0, 0.0, 181.0, 2.0) == (181.0, 2.0)
    xi, eta = conc.tangent_plane(181.01, 2.0, 181.0, 2.0)
    assert xi > 0 and abs(eta) < 1e-7                           # east = increasing RA
    back = conc.inverse_tangent_plane(*conc.tangent_plane([170.0, 191.0], [-8.0, 10.0], 181.0, 2.0),
                                      181.0, 2.0)
    np.testing.assert_allclose(back, ([170.0, 191.0], [-8.0, 10.0]), atol=1e-9)


def _config(**section):
    cfg = json.loads(json.dumps(BUILD_CONFIG))
    cfg["concentration"] = section
    return cfg


def test_config_validation():
    validate_config(_config())
    validate_config(_config(sample="spec", z_range=[0.017, 0.024], weight="count",
                            weight_clip=None, k=12, pixel_mpc=0.3, contour_step_sigma=0.5,
                            min_prominence_sigma=0, ridge_floor_sigma=3.0))
    validate_config(_config(weight="log1p_delta_sigma2d_k5"))
    bad = [({"sample": "phot"}, "sample"), ({"z_range": [0.03, 0.02]}, "z_range"),
           ({"weight": ""}, "weight"), ({"weight": 3}, "weight"),
           ({"weight_clip": 0}, "weight_clip"), ({"weight_clip": 101}, "weight_clip"),
           ({"k": 1}, "concentration.k"), ({"k": 2.5}, "concentration.k"),
           ({"k": True}, "concentration.k"), ({"k": None}, "concentration.k"),
           ({"ridge_floor_sigma": 0.5}, "ridge_floor_sigma"),
           ({"ridge_floor_sigma": "5"}, "ridge_floor_sigma"),
           ({"bandwidth_mpc": 2.0}, "unknown key"), ({"pixel_mpc": "x"}, "pixel_mpc"),
           ({"contour_step_sigma": 0}, "contour_step_sigma"),
           ({"min_prominence_sigma": -1}, "min_prominence_sigma"),
           ({"footprint_mpc": 1.0}, "unknown key"), ({"sigma_low": 2}, "unknown key"),
           ({"speed": 1}, "unknown key")]
    for section, text in bad:
        with pytest.raises(ConfigError, match=text):
            validate_config(_config(**section))
    no_phot = _config(weight="mass")
    del no_phot["photometry"]
    no_phot["environment"] = [e for e in no_phot["environment"] if "weight" not in e]
    with pytest.raises(ConfigError, match="photometry"):
        validate_config(no_phot)
    del no_phot["concentration"]["weight"]
    no_phot["concentration"]["weight"] = "count"
    validate_config(no_phot)                                    # count needs no masses
    no_z = _config()
    no_z["cluster"] = {"ra": 180.5, "dec": 0.5}
    with pytest.raises(ConfigError, match="cluster.z"):
        validate_config(no_z)
    with pytest.raises(ConfigError, match="mapping"):
        validate_config(_config() | {"concentration": [1]})


SIGMA_B = 0.6


@pytest.fixture(scope="module")
def straight():
    # two clumps joined by a straight bridge with a Gaussian cross-section of SIGMA_B
    df = conc_frame(blobs=[(-6.0, 0.0, 0.8, 500), (6.0, 0.0, 0.8, 400)],
                    bridges=[(-6.0, 0.0, 6.0, 0.0, 0.0, SIGMA_B, 700)])
    return (df,) + _run(df)


def test_a_path_drawn_beside_a_bridge_is_snapped_to_its_crest():
    df = conc_frame(blobs=[(-6.0, 0.0, 0.8, 500), (6.0, 0.0, 0.8, 400)],
                    bridges=[(-6.0, 0.0, 6.0, 0.0, 1.0, SIGMA_B, 700)])
    _, result = _run(df)
    h = result["info"]["knn_scale_mpc"]
    truth = bridge_curve(-6.0, 0.0, 6.0, 0.0, 1.0, np.linspace(0.0, 1.0, 4000))
    for side in (0.6, -0.6):                             # drawn 0.6 h to either side of the bridge
        drawn = bridge_curve(-6.0, 0.0, 6.0, 0.0, 1.0, np.linspace(0.0, 1.0, 13)) \
            + [0.0, side * h]
        _, model = conc.compute_model(df, result, {"paths": [_path_spec(drawn)]})
        path = model["paths"][0]
        pts = _truth(path["ra"], path["dec"])
        gap = np.array([np.hypot(*(truth - p).T).min() for p in pts])
        interior = (pts[:, 0] > -4.0) & (pts[:, 0] < 4.0)
        assert np.median(gap[interior]) < 0.25 * h < 0.6 * h
        assert path["shift_max_mpc"] > 0.5 * h and path["n_iter"] > 1
        assert path["label"] == "drawn 1" and not path["open"]
        assert path["clamped_fraction"] < 0.1          # a crest within reach nearly all along
        assert path["shift_max_mpc"] <= conc.SNAP_MAX * h + 1e-6
        assert len(path["input_ra"]) == 13 and path["crest_n_sigma_mean"] > 1.0
    assert model["skipped"] == [] and model["info"]["n_paths"] == 1


def test_the_width_of_a_bridge_is_that_of_the_gaussian_fitted_to_its_kNN_cross_section(straight):
    df, columns, result = straight
    h = result["info"]["knn_scale_mpc"]
    t = np.linspace(-5.0, 5.0, 11)
    spec = {"paths": [_path_spec(np.column_stack([t, np.full_like(t, 0.3)]), "ridge A")]}
    _, model = conc.compute_model(df, result, spec)
    path = model["paths"][0]
    # the kNN map is not a convolution with a kernel: the fitted width is that of the planted
    # cross-section, a little wider (the neighbour disc blurs it), never wider than the kNN scale
    # added in quadrature; no intrinsic width is defined
    assert 0.8 * SIGMA_B < path["width_median_mpc"] < np.hypot(SIGMA_B, h)
    assert path["fit"]["ok"] and 0.8 * SIGMA_B < path["fit"]["s"] < np.hypot(SIGMA_B, h)
    assert "width_intrinsic_median_mpc" not in path
    assert len(path["width_mpc"]) == len(path["x_mpc"]) and min(path["width_mpc"]) > 0.5 * SIGMA_B
    assert path["length_mpc"] == pytest.approx(10.0, abs=1.0)
    assert np.abs(_truth(path["ra"], path["dec"])[:, 1]).max() < SIGMA_B     # on the straight crest
    # the band edges are the path moved by the width on either side
    g = result["grid"]
    centre = np.column_stack(conc.tangent_plane(path["ra"], path["dec"], g["ra0"], g["dec0"])) * D_A
    for side in ("left", "right"):
        edge = np.column_stack(conc.tangent_plane(path[side + "_ra"], path[side + "_dec"],
                                                  g["ra0"], g["dec0"])) * D_A
        np.testing.assert_allclose(np.hypot(*(edge - centre).T), path["width_mpc"], rtol=1e-3)
    assert len(path["left_ra"]) == len(path["right_dec"]) == len(path["ra"])


def test_a_selected_peak_is_refined_to_the_blob_and_gets_a_fitted_width():
    df = conc_frame(blobs=[(2.0, 1.0, 0.8, 800)])
    _, result = _run(df)
    h, pixel = result["info"]["knn_scale_mpc"], result["grid"]["pixel"]
    assert len(result["peaks"]) == 1
    ra, dec = _sky(2.0 + 0.5 * h, 1.0)                  # given 0.5 h off the blob
    columns, model = conc.compute_model(df, result, {"peaks": [[float(ra), float(dec)]]})
    peak = model["peaks"][0]
    assert peak["label"] == "peak 1" and model["paths"] == [] and model["skipped"] == []
    assert np.hypot(*(_truth(peak["ra"], peak["dec"])[0] - [2.0, 1.0])) < pixel     # within a cell
    assert 0.2 * h < peak["shift_mpc"] < 0.8 * h                  # moved from 0.5 h off the blob
    assert peak["height"] == pytest.approx(result["top"], rel=0.15) and peak["n_sigma"] > 5.0
    assert 0.4 * 0.8 < peak["width_mpc"] < np.hypot(0.8, h) and peak["fit_ok"]
    assert "width_intrinsic_mpc" not in peak
    assert peak["A"] > 0 and peak["B"] >= 0
    # the circle of the 1 sigma disc is closed and CIRCLE_POINTS segments around the peak
    assert len(peak["circle_ra"]) == conc.CIRCLE_POINTS + 1
    assert peak["circle_ra"][0] == pytest.approx(peak["circle_ra"][-1])
    cx, cy = conc.tangent_plane(peak["circle_ra"], peak["circle_dec"], result["grid"]["ra0"],
                                result["grid"]["dec0"])
    np.testing.assert_allclose(np.hypot(cx * D_A - peak["x"], cy * D_A - peak["y"]),
                               peak["width_mpc"], rtol=1e-3)
    # the galaxies of the blob are inside the disc of one sigma radius
    assert columns["model_structure"][columns["model_inside"]][0] == "peak 1"
    assert columns["model_dist_sigma"][np.argmin(columns["model_dist_mpc"])] < 0.2


def test_model_membership_columns(straight):
    df, columns, result = straight
    t = np.linspace(-5.0, 5.0, 11)
    spec = {"peaks": [list(map(float, _sky(-6.0, 0.0))), list(map(float, _sky(6.0, 0.0)))],
            "paths": [_path_spec(np.column_stack([t, np.zeros_like(t)]), "bridge")]}
    cols, model = conc.compute_model(df, result, spec)
    assert set(cols) == set(conc.MODEL_COLUMNS) and all(len(v) == len(df) for v in cols.values())
    dist, sig, inside, who = (cols[k] for k in conc.MODEL_COLUMNS)
    np.testing.assert_array_equal(inside, sig <= 1.0)
    assert inside.dtype == bool and dist.shape == sig.shape and (sig >= 0).all()
    assert (dist >= 0).all()
    assert set(who) <= {"peak 1", "peak 2", "bridge"}
    assert (who[-700:][inside[-700:]] == "bridge").mean() > 0.5       # the bridge galaxies: its band
    assert inside[-700:].mean() > 0.6                                  # mostly inside
    assert inside[:2500].mean() < 0.15                                 # the far field: outside
    assert model["info"]["counts"]["inside_spec"] == inside.sum()
    assert model["info"]["counts"]["outside_spec"] == (~inside).sum()
    assert model["info"]["n_peaks"] == 2 and model["info"]["n_paths"] == 1
    assert json.dumps(model["info"]) and set(model["info"]["constants"]) >= {"PEAK_SEARCH", "WINDOW"}
    np.testing.assert_array_equal(model["inside"], inside)
    # a row without a position has NaN, False and no structure
    broken = df.copy()
    broken.loc[3, "ra"] = np.nan
    cols2, _ = conc.compute_model(broken, result, spec)
    assert np.isnan(cols2["model_dist_mpc"][3]) and np.isnan(cols2["model_dist_sigma"][3])
    assert not cols2["model_inside"][3] and cols2["model_structure"][3] == ""
    assert (cols2["model_inside"][:3] == inside[:3]).all()
    lines = conc.describe_model(model)
    assert lines[0].startswith("Model: 2 peaks and 1 paths")
    assert lines[-1].startswith("Inside the 1 sigma band")
    assert any(line.startswith("bridge: ") for line in lines)
    half = np.arange(len(df)) % 2 == 0
    spec_part = conc.describe_model(model, half, df)[-1]
    assert f"{int((inside & half).sum())} (" in spec_part


def test_a_path_outside_the_survey_is_skipped_and_a_bad_spec_is_refused(straight):
    df, columns, result = straight
    far = _path_spec(np.array([[60.0, 0.0], [70.0, 0.0], [80.0, 0.0]]), "far")
    cols, model = conc.compute_model(df, result, {"paths": [far]})
    assert [s["label"] for s in model["skipped"]] == ["far"] and model["paths"] == []
    assert "inside the survey" in model["skipped"][0]["reason"] and model["info"]["n_skipped"] == 1
    assert not cols["model_inside"].any() and (cols["model_structure"] == "").all()
    assert np.isnan(cols["model_dist_mpc"]).all()
    one = _path_spec(np.array([[0.0, 0.0], [70.0, 0.0]]), "one point inside")
    skipped = conc.compute_model(df, result, {"paths": [one]})[1]["skipped"]
    assert skipped[0]["label"] == "one point inside"
    ra, dec = _sky(0.0, 0.0)
    _, mixed = conc.compute_model(df, result, {"peaks": [[float(ra), float(dec)]], "paths": [far]})
    assert len(mixed["peaks"]) == 1 and len(mixed["skipped"]) == 1
    for spec in ({}, {"peaks": [], "paths": []}, {"peaks": None}):
        with pytest.raises(conc.ConcentrationError, match="select at least one peak or path"):
            conc.compute_model(df, result, spec)
    for spec in ({"peaks": [[1.0]]}, {"peaks": [[np.nan, 2.0]]},
                 {"paths": [{"ra": [1, 2], "dec": [1]}]}, {"paths": [{"label": "x"}]}, "peaks",
                 None, {"peaks": [["a", "b"]]}):
        with pytest.raises(conc.ConcentrationError, match="spec"):
            conc.compute_model(df, result, spec)
    with pytest.raises(ValueError):                     # ConcentrationError is a ValueError
        conc.compute_model(df, result, {"peaks": 3})


def test_default_spec_selects_every_peak_and_segment_and_round_trips(bridged):
    df, columns, result = bridged
    spec = conc.default_spec(result)
    assert len(spec["peaks"]) == len(result["peaks"]) == 2
    assert [p["label"] for p in spec["paths"]] == ["ridge 1"]
    again = json.loads(json.dumps(spec))
    assert again == spec
    cols, model = conc.compute_model(df, result, again)
    assert model["skipped"] == [] and [p["label"] for p in model["peaks"]] == ["peak 1", "peak 2"]
    pixel = result["grid"]["pixel"]
    for p, q in zip(model["peaks"], result["peaks"]):
        assert np.hypot(p["x"] - q["x"], p["y"] - q["y"]) < pixel and p["shift_mpc"] < pixel
    path = model["paths"][0]
    ridge = result["ridges"][0]
    pts = np.column_stack([path["x_mpc"], path["y_mpc"]])
    assert np.hypot(*(pts[0] - ridge["points"][0])).max() < 1.0 and path["length_mpc"] > 9.0
    assert 9.0 < path["length_mpc"] < 13.0 and path["crest_n_sigma_min"] > 0.0
    assert json.dumps(model["info"]) and np.isfinite(cols["model_dist_sigma"]).all()
    # the bridge galaxies are inside the model: the band covers the crest
    assert cols["model_inside"][-500:].mean() > 0.7
    assert (cols["model_dist_sigma"][-500:] < cols["model_dist_sigma"][:2500].mean()).mean() > 0.9
    # the ends of the path are pinned to the refined peaks
    ends = np.array([pts[0], pts[-1]])
    for p in model["peaks"]:
        assert min(np.hypot(*(ends - [p["x"], p["y"]]).T)) < 1e-9


def test_explorer_model_is_cached_and_filters_the_selection(conc_explorer):
    ex = conc_explorer
    base = ex.select({"z_range": [0.014, 0.028]})
    _, result = ex.concentration(base)
    spec = conc.default_spec(result)
    first = ex.model(base, spec)
    again = ex.model(base, json.loads(json.dumps(spec)))
    assert again is first                                 # same objects: cached on the spec
    assert ex.model(base, None) is first                  # None: the default spec
    columns, model = first
    assert set(conc.MODEL_COLUMNS) <= set(columns) and model["info"]["n_peaks"] == len(result["peaks"])
    other = {**spec, "peaks": spec["peaks"][:1]}
    assert ex.model(base, other) is not first
    # select: conc_region follows the model band when conc_model is given
    filters = {"z_range": [0.014, 0.028], "conc_region": "inside", "conc_model": spec}
    inside = ex.select(filters)
    np.testing.assert_array_equal(inside, base & columns["model_inside"])
    outside = ex.select({**filters, "conc_region": "outside"})
    np.testing.assert_array_equal(outside, base & ~columns["model_inside"])
    assert inside.sum() > 0 and outside.sum() > 0 and not (inside & outside).any()
    contour = ex.select({"z_range": [0.014, 0.028], "conc_region": "inside"})
    assert not np.array_equal(contour, inside)            # the model is another region
    np.testing.assert_array_equal(
        ex.select({**filters, "conc_model": {}}), contour)    # an empty model: the 1 sigma contour
    np.testing.assert_array_equal(ex.select({**filters, "conc_region": "any"}), base)
    np.testing.assert_array_equal(
        ex.select({**filters, "conc_params": {"k": 12}}),
        base & ex.model(base, spec, k=12)[0]["model_inside"])
    assert "conc_model" in FILTER_KEYS
    with pytest.raises(ValueError):
        ex.model(base, {"peaks": [[1.0]]})
    with pytest.raises(ValueError):                       # a spec that is not a dict is refused
        ex.select({**filters, "conc_model": [[181.0, 2.0]]})
    with pytest.raises(ValueError, match="JSON"):
        ex.model(base, {"peaks": [[object(), 1.0]]})
    # mass_table with the model membership
    conc_columns = ex.concentration(base)[0]
    table = ex.mass_table(conc_columns, base, inside=columns["model_inside"])
    n = table[(table["stat"] == "n") & (table["z_type"] == "all")].set_index("region")["value"]
    assert n["inside"] == (base & columns["model_inside"]).sum()
    assert n["outside"] == (base & ~columns["model_inside"]).sum()
    default = ex.mass_table(conc_columns, base)
    n0 = default[(default["stat"] == "n") & (default["z_type"] == "all")].set_index("region")["value"]
    assert n0["inside"] == (base & conc_columns["conc_inside"]).sum()


def test_a_path_drawn_far_from_any_crest_stays_within_snap_max_of_the_drawing(straight):
    df, columns, result = straight
    h = result["info"]["knn_scale_mpc"]
    assert conc.SNAP_MAX == 1.0
    t = np.linspace(-3.0, 3.0, 7)
    drawn = np.column_stack([t, np.full_like(t, 2.0 * h)])       # 2 h beside the bridge
    _, model = conc.compute_model(df, result, {"paths": [_path_spec(drawn)]})
    path = model["paths"][0]
    assert path["shift_max_mpc"] <= conc.SNAP_MAX * h + 1e-6
    assert path["shift_max_mpc"] > 0.5 * conc.SNAP_MAX * h       # it went a good way
    assert path["clamped_fraction"] >= 0.0
    assert model["info"]["constants"]["SNAP_MAX"] == conc.SNAP_MAX
    pts = _truth(path["ra"], path["dec"])
    assert (pts[:, 1] > 0.9 * h).all()                           # still on the drawn side
    line = [x for x in conc.describe_model(model) if x.startswith("drawn 1")][0]
    assert f"no crest within {conc.SNAP_MAX:g} h of the drawn path along" in line
    # a path on the crest has no such note
    on_crest = np.column_stack([t, np.zeros_like(t)])
    quiet = conc.compute_model(df, result, {"paths": [_path_spec(on_crest)]})[1]
    assert quiet["paths"][0]["clamped_fraction"] == 0.0
    assert "no crest within" not in " ".join(conc.describe_model(quiet))


def test_band_edges_and_circles_outside_the_survey_are_nan_and_the_path_stays_inside():
    df = conc_frame(blobs=[(-3.0, 0.0, 1.0, 500), (14.8, 0.0, 1.0, 400)])
    _, result = _run(df)
    g, hull = result["grid"], result["hull"]

    def project(ra, dec):
        return np.column_stack(conc.tangent_plane(ra, dec, g["ra0"], g["dec0"])) * D_A

    # a clump at the edge: part of the circle of its 1 sigma radius is outside the hull
    spec = conc.default_spec(result)
    _, model = conc.compute_model(df, result, {"peaks": spec["peaks"]})
    peak = max(model["peaks"], key=lambda p: p["x"])
    assert peak["x"] > 12.0
    angle = np.linspace(0.0, 2.0 * np.pi, conc.CIRCLE_POINTS + 1)
    circle = np.column_stack([peak["x"] + peak["width_mpc"] * np.cos(angle),
                              peak["y"] + peak["width_mpc"] * np.sin(angle)])
    outside = ~conc.in_hull(hull, circle[:, 0], circle[:, 1])
    ra, dec = np.array(peak["circle_ra"]), np.array(peak["circle_dec"])
    assert outside.any() and not outside.all()
    np.testing.assert_array_equal(np.isnan(ra), outside)
    np.testing.assert_array_equal(np.isnan(dec), outside)
    assert conc.in_hull(hull, *project(ra[~outside], dec[~outside]).T, tol=1e-6).all()
    inner = [p for p in model["peaks"] if p is not peak][0]       # a clump well inside: no NaN
    assert np.isfinite(inner["circle_ra"]).all() and np.isfinite(inner["circle_dec"]).all()
    # a path 0.4 Mpc inside the longest edge of the hull: its band reaches out of the survey
    vertices = hull["vertices"]
    edges = np.roll(vertices, -1, axis=0) - vertices
    i = int(np.hypot(*edges.T).argmax())
    direction = edges[i] / np.hypot(*edges[i])
    inward = np.array([-direction[1], direction[0]])                # hull vertices run counter-clockwise
    pts = np.array([vertices[i] + t * edges[i] for t in np.linspace(0.1, 0.9, 6)]) + 0.4 * inward
    ra, dec = conc.inverse_tangent_plane(pts[:, 0] / D_A, pts[:, 1] / D_A, g["ra0"], g["dec0"])
    _, model = conc.compute_model(df, result, {"paths": [{"label": "edge", "ra": ra.tolist(),
                                                           "dec": dec.tolist()}]})
    path = model["paths"][0]
    fitted = project(path["ra"], path["dec"])
    assert np.isfinite(fitted).all() and conc.in_hull(hull, *fitted.T).all()   # inside, by clipping
    normals = conc._unit_normals(fitted)
    width = np.array(path["width_mpc"])
    seen_nan = seen_finite = False
    for side, sign in (("left", 1.0), ("right", -1.0)):
        edge = fitted + sign * width[:, None] * normals
        gone = ~conc.in_hull(hull, edge[:, 0], edge[:, 1])
        np.testing.assert_array_equal(np.isnan(path[side + "_ra"]), gone)
        np.testing.assert_array_equal(np.isnan(path[side + "_dec"]), gone)
        seen_nan |= bool(gone.any())
        seen_finite |= bool((~gone).any())
    assert seen_nan and seen_finite
    assert json.dumps(model["info"])


def test_a_path_drawn_up_a_slope_does_not_retract_and_reports_its_true_shift():
    # a broad clump with a bridge leaving it; the path is drawn along the bridge from the clump,
    # 0.5 h beside it. Points move along their normals only, so the path cannot slide back up
    # the slope of the clump
    df = conc_frame(blobs=[(-3.0, 0.0, 1.5, 900)],
                    bridges=[(-3.0, 0.0, 10.0, 0.0, 1.0, SIGMA_B, 250)])
    _, result = _run(df)
    h = result["info"]["knn_scale_mpc"]
    drawn = bridge_curve(-3.0, 0.0, 10.0, 0.0, 1.0, np.linspace(0.0, 1.0, 14)) + [0.0, 0.5 * h]
    _, model = conc.compute_model(df, result, {"paths": [_path_spec(drawn)]})
    path = model["paths"][0]
    fitted = _truth(path["ra"], path["dec"])
    assert path["length_mpc"] >= 0.85 * np.hypot(*np.diff(drawn, axis=0).T).sum()
    far = np.hypot(*(fitted[-1] - drawn[-1]))
    assert far <= conc.SNAP_MAX * h + 1e-6                        # the far end stays near its place
    assert fitted[:, 0].max() > drawn[-1, 0] - 0.5 * h
    assert fitted[:, 0].min() < drawn[0, 0] + 0.5 * h
    truth = bridge_curve(-3.0, 0.0, 10.0, 0.0, 1.0, np.linspace(0.0, 1.0, 4000))
    gap = np.array([np.hypot(*(truth - p).T).min() for p in fitted])
    assert np.median(gap) < 0.3 * h
    # the reported shift is the true displacement of the ends and of the points
    assert path["shift_max_mpc"] + 1e-6 >= far
    assert path["shift_max_mpc"] + 1e-6 >= np.hypot(*(fitted[0] - drawn[0]))
    assert path["shift_max_mpc"] <= conc.SNAP_MAX * h + 1e-6 and path["end_pin_mpc"] == 0.0
    assert path["n_iter"] == 2


def test_a_path_along_a_crest_stays_put_and_an_end_near_a_peak_is_pinned(straight):
    df, columns, result = straight
    h = result["info"]["knn_scale_mpc"]
    t = np.linspace(-4.0, 4.0, 9)
    _, model = conc.compute_model(df, result, {"paths": [_path_spec(np.column_stack(
        [t, np.zeros_like(t)]))]})
    path = model["paths"][0]
    assert path["shift_max_mpc"] < 0.25 * h and path["clamped_fraction"] == 0.0
    assert path["end_pin_mpc"] == 0.0
    # an end within ENDPOINT_SNAP of a selected peak goes to the refined peak (drawn 0.5 h off it)
    ra, dec = _sky(-6.0, 0.0)
    peak = conc.compute_model(df, result, {"peaks": [[float(ra), float(dec)]]})[1]["peaks"][0]
    start = _truth(*conc.inverse_tangent_plane(peak["x"] / D_A, peak["y"] / D_A,
                                               result["grid"]["ra0"], result["grid"]["dec0"]))[0]
    t = np.linspace(0.0, 4.0, 5)
    drawn = np.column_stack([start[0] + 0.5 * h + t, np.zeros_like(t)])
    spec = {"peaks": [[float(ra), float(dec)]], "paths": [_path_spec(drawn)]}
    _, model = conc.compute_model(df, result, spec)
    path, peak = model["paths"][0], model["peaks"][0]
    assert 0.0 < path["end_pin_mpc"] <= conc.ENDPOINT_SNAP * h
    assert np.hypot(path["x_mpc"][0] - peak["x"], path["y_mpc"][0] - peak["y"]) < 1e-6
    assert path["shift_max_mpc"] <= conc.SNAP_MAX * h + 1e-6      # the pinned end is not counted
