import json

import numpy as np
import pandas as pd
import pytest
from scipy.stats import gaussian_kde

from knn import concentration as conc
from knn.config import ConfigError, validate_config
from knn.explorer import FILTER_KEYS
from synthetic import BUILD_CONFIG, COSMO, CONC_TRUTH, bridge_curve, conc_frame

Z_CL = CONC_TRUTH["z"]
D_A = float(COSMO.angular_diameter_distance(Z_CL).value)
BLOBS = [(-5.0, 0.0, 0.8, 500), (5.0, 0.0, 0.8, 400)]


def _run(df, mask=None, section=None, **overrides):
    overrides = {"weight": "count", "bandwidth_mpc": 1.0, **overrides}
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


@pytest.fixture(scope="module")
def bridged():
    df = conc_frame(blobs=BLOBS, bridges=[(-5.0, 0.0, 5.0, 0.0, 2.0, 0.5, 500)])
    return (df,) + _run(df)


@pytest.fixture(scope="module")
def valley():
    df = conc_frame(blobs=[(-7.0, 0.0, 0.8, 500), (7.0, 0.0, 0.8, 400)])
    return (df,) + _run(df)


def test_empty_cells_inside_the_hull_are_zero_and_cells_outside_are_nan():
    # the galaxies of the slice (ra < 181) fill half of the disc; the other half of the
    # catalogue is at another redshift and only sets the footprint
    df = conc_frame(n_field=3000, seed=3)
    sample = (df["ra"] < CONC_TRUTH["ra"]).to_numpy()
    df.loc[~sample, "z_best"] = 0.03
    _, result = _run(df, sample)
    dens, footprint = result["dens"], result["footprint"]
    X, Y = _cell_xy(result)
    far = footprint & (X > 5.0)                          # farther than 4 bandwidths from the slice
    assert far.sum() > 100 and (dens[far] == 0.0).all()
    assert np.isfinite(dens[footprint]).all() and (dens[footprint] >= 0).all()
    assert np.isnan(dens[~footprint]).all() and (~footprint).sum() > 100
    assert np.isnan(dens[0, 0]) and np.isnan(dens[-1, -1]) and not footprint[0].any()
    # the footprint is the hull of all the rows: the same for another slice of the same table
    _, other = _run(df, ~sample)
    np.testing.assert_array_equal(other["footprint"], footprint)
    assert other["info"]["hull_area_mpc2"] == result["info"]["hull_area_mpc2"]
    assert result["info"]["n_rows"] == len(df) and result["info"]["n_sample"] == sample.sum()
    assert footprint.sum() * result["grid"]["pixel"] ** 2 == pytest.approx(
        result["info"]["hull_area_mpc2"], rel=0.03)
    rows, cols = np.nonzero(footprint)
    assert conc.in_hull(result["hull"], X[rows, cols], Y[rows, cols]).all()
    assert not conc.in_hull(result["hull"], X[~footprint], Y[~footprint]).any()
    # empty cells count in the statistics: the mean is the mean over the footprint
    assert result["mean"] == pytest.approx(dens[footprint].mean())
    assert result["sigma"] == pytest.approx(dens[footprint].std())


def test_density_is_flat_up_to_the_border_of_the_hull():
    rng = np.random.default_rng(3)
    n, radius = 6000, 10.0
    r, t = radius * np.sqrt(rng.uniform(0, 1, n)), rng.uniform(0, 2 * np.pi, n)
    x, y = r * np.cos(t), r * np.sin(t)
    hull = conc.survey_hull(x, y)
    grid = conc.density_map(x, y, np.ones(n), hull, 1.0, 0.25)
    dens, footprint = grid["dens"], grid["footprint"]
    level = n / hull["area"]
    X, Y = np.meshgrid(grid["x0"] + (np.arange(dens.shape[1]) + 0.5) * 0.25,
                       grid["y0"] + (np.arange(dens.shape[0]) + 0.5) * 0.25)
    border = _border_distance({"hull": hull}, np.column_stack([X[footprint], Y[footprint]]))
    values = dens[footprint]
    assert values[border > 3.0].mean() == pytest.approx(level, rel=0.05)
    near = (border > 0.5) & (border < 1.5)               # within one kernel of the border
    assert values[near].mean() > 0.9 * level             # the plain smoothing gives about 0.8
    assert values[border <= 0.5].mean() > 0.85 * level   # and 0.5 at the border itself
    # a thin sliver: the divisor is floored, the map stays finite and bounded
    sx, sy = rng.uniform(0, 40, 4000), rng.uniform(-0.25, 0.25, 4000)
    sliver = conc.density_map(sx, sy, np.ones(4000), conc.survey_hull(sx, sy), 2.0, 0.5)
    assert conc.COVER_FLOOR == 0.2
    assert np.isfinite(sliver["dens"][sliver["footprint"]]).all()
    assert sliver["dens"][sliver["footprint"]].max() < 3.0 * 4000 / (40.0 * 0.5)


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
    for line in closed_lines:                            # the central blob is a closed loop
        assert np.allclose(line[0], line[-1]) and _border_distance(result, line).min() > 0
        assert np.mean(line[:, 0]) < 0.0
    for c in result["contours"]:                         # no line leaves the survey area
        for line in c["lines"]:
            assert conc.in_hull(result["hull"], line[:, 0], line[:, 1]).all()
    # the same holds on the sky, and the edge peak is flagged open
    sky = conc.contour_table(result)[0]["lines"]
    assert [ln["closed"] for ln in sky] == first["closed"]
    edge_peak = _near(result["peaks"], 13.5, 0.0, 2.0)
    assert len(edge_peak) == 1 and edge_peak[0]["open"]
    assert not _near(result["peaks"], -3.0, 0.0, 1.5)[0]["open"]


def test_two_peaks_joined_by_a_bridge_give_one_ridge_along_its_crest(bridged):
    df, columns, result = bridged
    assert len(result["peaks"]) == 2
    left, right = _near(result["peaks"], -5.0, 0.0, 1.0)[0], _near(result["peaks"], 5.0, 0.0, 1.0)[0]
    assert [p["id"] for p in result["peaks"]] == [1, 2] and left["height"] > right["height"]
    assert len(result["ridges"]) == 1
    ridge = result["ridges"][0]
    assert ridge["kind"] == "peak" and not ridge["open"]
    assert ridge["peaks"] == [2, 1]                      # lower end first
    assert ridge["saddle"] >= result["level"]            # one 1 sigma region
    assert ridge["saddle_n_sigma"] == pytest.approx(
        (ridge["saddle"] - result["mean"]) / result["sigma"])
    assert [n["kind"] for n in result["nodes"]] == ["peak", "peak"] and ridge["nodes"] == [2, 1]
    assert [p["degree"] for p in result["peaks"]] == [1, 1] and ridge["cells"][0] == right["cell"]
    pts = ridge["points"]
    arc = bridge_curve(-5.0, 0.0, 5.0, 0.0, 2.0, np.linspace(0, 1, 2000))
    gap = np.array([np.hypot(*(arc - p).T).min() for p in pts])
    assert gap.max() < 0.8 and np.median(gap) < 0.4      # on the crest, not on the straight line
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


def test_two_peaks_separated_by_a_deep_valley_are_still_joined_by_a_low_ridge(valley):
    df, columns, result = valley
    assert len(result["peaks"]) == 2 and len(result["ridges"]) == 1       # no saddle threshold
    ridge = result["ridges"][0]
    assert ridge["kind"] == "peak" and not ridge["open"] and ridge["peaks"] == [2, 1]
    assert ridge["saddle_n_sigma"] < 1.0 and ridge["saddle"] < result["level"]
    assert result["info"]["n_ridges"] == 1 and np.isfinite(columns["conc_ridge_dist_mpc"]).all()
    pts = ridge["points"]                                # no crest: the path winds through the field
    left = _near(result["peaks"], -7.0, 0.0, 1.0)[0]
    right = _near(result["peaks"], 7.0, 0.0, 1.0)[0]
    assert np.hypot(*(pts[0] - [right["x"], right["y"]])) < 1e-9
    assert np.hypot(*(pts[-1] - [left["x"], left["y"]])) < 1e-9
    assert ridge["length"] > 14.0                        # at least the distance between the peaks
    assert pts[:, 0].min() < -6.5 and pts[:, 0].max() > 6.5      # the highest ground, not a line
    # they are in separate 1 sigma regions
    assert result["info"]["n_regions"] == 2
    assert left["height"] > result["level"] and right["height"] > result["level"]
    assert left["prominence_sigma"] > 1.0 and right["prominence_sigma"] > 1.0


def test_a_bump_below_the_minimum_prominence_is_not_a_significant_peak():
    # a small clump on the flank of a broad cluster: a local maximum above the 1 sigma level
    # that rises only slightly above its saddle
    df = conc_frame(blobs=[(0.0, 0.0, 2.5, 1500), (4.5, 0.0, 0.4, 100)])
    _, result = _run(df)
    bump = [p for p in result["all_peaks"] if np.hypot(p["x"] - 4.0, p["y"]) < 1.5]
    assert len(bump) == 1
    bump = bump[0]
    assert bump["height"] >= result["level"] and bump["prominence_sigma"] < 1.0
    assert not bump["significant"] and bump["id"] == 0
    assert len(result["peaks"]) == 1 and not _near(result["peaks"], 4.0, 0.0, 1.5)
    assert result["peaks"][0]["id"] == 1 and result["peaks"][0]["height"] == result["top"]
    assert result["info"]["n_peaks"] == 1 and result["info"]["n_local_maxima"] > 1
    # with no prominence required it is marked
    _, loose = _run(df, min_prominence_sigma=0.0)
    assert len(_near(loose["peaks"], 4.0, 0.0, 1.5)) == 1 and len(loose["peaks"]) >= 2
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


def test_a_field_without_structure_has_no_strong_peak():
    df = conc_frame(n_field=3000, seed=6)
    _, result = _run(df, bandwidth_mpc=2.0)
    assert result["info"]["top_n_sigma"] < 4.0           # noise only: no mountain
    assert result["peaks"][0]["id"] == 1 and result["peaks"][0]["height"] == result["top"]
    assert "Peaks: " in conc.describe(result["info"])[2]
    # only the global maximum marked: the network joins it to the high border cells of the noise
    _, single = _run(df, bandwidth_mpc=2.0, min_prominence_sigma=100.0)
    assert len(single["peaks"]) == 1 and single["info"]["n_border_ends"] >= 1
    assert single["peaks"][0]["degree"] >= 1 and _connected(single)
    assert all(r["open"] == (r["kind"] == "boundary") for r in single["ridges"])
    assert len(single["ridges"]) == len(single["nodes"]) - 1           # a tree


def test_a_single_peak_without_a_border_end_has_no_ridge():
    df = conc_frame(blobs=[(0.0, 0.0, 1.0, 600)])
    columns, single = _run(df)
    assert len(single["peaks"]) == 1 and single["ridges"] == [] and single["peaks"][0]["degree"] == 0
    assert [n["kind"] for n in single["nodes"]] == ["peak"] and single["nodes"][0]["degree"] == 0
    assert np.isnan(columns["conc_ridge_dist_mpc"]).all()
    info = single["info"]
    assert info["n_ridges"] == 0 and info["ridges_total_length_mpc"] == 0.0
    assert info["n_junctions"] == 0 and info["n_border_ends"] == 0
    assert "ridges: 0 segments, 0 branch points, 0 open" in conc.describe(info)[2]
    assert conc.ridge_table(single) == [] and conc.node_table(single)[0]["degree"] == 0


def test_every_significant_peak_is_connected_to_the_others():
    # four separate clumps, no bridges: the saddles are far below the 1 sigma level
    df = conc_frame(blobs=[(-8.0, 0.0, 0.8, 500), (8.0, 0.0, 0.8, 400), (0.0, 9.0, 0.8, 350),
                           (0.0, -9.0, 0.8, 300)])
    _, result = _run(df)
    assert len(result["peaks"]) == 4 and result["info"]["n_regions"] == 4
    assert all(p["degree"] >= 1 for p in result["peaks"])
    assert _connected(result) and len(result["ridges"]) == 3 and result["info"]["n_border_ends"] == 0
    assert all(r["kind"] == "peak" and r["saddle_n_sigma"] < 1.0 for r in result["ridges"])
    assert [r["saddle"] for r in result["ridges"]] == sorted((r["saddle"] for r in result["ridges"]),
                                                              reverse=True)
    assert [r["id"] for r in result["ridges"]] == [1, 2, 3]
    assert sum(n["degree"] for n in result["nodes"]) == 2 * len(result["ridges"])


def test_a_ridge_branches_at_a_junction_or_at_a_peak():
    # a bridge from a third clump meets the bridge between two others: a branch point in the
    # crest that is not a peak, joining three peaks
    df = conc_frame(blobs=[(-8.0, 0.0, 0.8, 500), (8.0, 0.0, 0.8, 450), (0.0, 8.0, 0.8, 400)],
                    bridges=[(-8.0, 0.0, 8.0, 0.0, 0.0, 0.5, 300), (0.0, 8.0, 0.0, 0.0, 0.0, 0.5, 300)])
    _, result = _run(df)
    assert len(result["peaks"]) == 3 and [p["degree"] for p in result["peaks"]] == [1, 1, 1]
    junctions = [n for n in result["nodes"] if n["kind"] == "junction"]
    assert len(junctions) == 1 and junctions[0]["degree"] == 3 and junctions[0]["peak"] is None
    assert abs(junctions[0]["x"]) < 2.0 and abs(junctions[0]["y"]) < 3.0   # near the middle
    assert _connected(result) and len(result["ridges"]) == 3
    assert sorted(r["kind"] for r in result["ridges"]) == ["junction"] * 3
    assert all(None in r["peaks"] and junctions[0]["id"] in r["nodes"] for r in result["ridges"])
    assert result["info"]["n_junctions"] == 1
    assert "1 branch points" in conc.describe(result["info"])[2]
    # three clumps in a row: the middle one is a branch point itself (a peak of degree 2)
    df = conc_frame(blobs=[(-8.0, 0.0, 0.8, 400), (0.0, 0.0, 0.8, 600), (8.0, 0.0, 0.8, 400)],
                    bridges=[(-8.0, 0.0, 0.0, 0.0, 0.0, 0.5, 300), (0.0, 0.0, 8.0, 0.0, 0.0, 0.5, 300)])
    _, row = _run(df)
    middle = _near(row["peaks"], 0.0, 0.0, 1.0)[0]
    assert middle["id"] == 1 and middle["degree"] == 2 and _connected(row)
    assert [p["degree"] for p in row["peaks"] if p is not middle] == [1, 1]
    assert row["info"]["n_junctions"] == 0 and len(row["ridges"]) == 2


def test_a_bridge_that_leaves_the_map_ends_at_a_border_node_unless_a_peak_is_there():
    df = conc_frame(blobs=[(-5.0, 0.0, 0.8, 500)], bridges=[(-5.0, 0.0, 15.5, 0.0, 0.0, 0.6, 400)])
    _, result = _run(df)
    pixel = result["grid"]["pixel"]
    border = [n for n in result["nodes"] if n["kind"] == "border"]
    assert len(border) == 1 and result["info"]["n_border_ends"] == 1 and len(result["peaks"]) == 1
    assert _border_distance(result, [[border[0]["x"], border[0]["y"]]])[0] <= 1.5 * pixel
    assert border[0]["x"] > 12.0 and [r["kind"] for r in result["ridges"]] == ["boundary"]
    # a clump in the very corner of the map: its peak is within BORDER_PEAK_CELLS of the border
    # component, it is the open end and no border node is added
    df = conc_frame(blobs=[(-3.0, 0.0, 1.0, 500), (14.8, 0.0, 1.0, 400)])
    _, edge = _run(df)
    peak = _near(edge["peaks"], 14.4, 0.0, 1.0)[0]
    assert peak["open"] and edge["info"]["n_border_ends"] == 0
    assert [r["kind"] for r in edge["ridges"]] == ["peak"] and peak["degree"] == 1


def test_weight_switch_changes_the_map_not_the_bandwidth():
    df = conc_frame(blobs=BLOBS)
    df["flag"] = np.where(df["ra"] > df["ra"].median(), 2.0, 1.0)
    _, mass = _run(df, weight="mass", bandwidth_mpc=None)
    _, count = _run(df, weight="count", bandwidth_mpc=None)
    _, column = _run(df, weight="flag", bandwidth_mpc=None)
    _, as_count = _run(df.assign(one=1.0), weight="one", bandwidth_mpc=None)
    assert mass["info"]["bandwidth"]["mpc"] == count["info"]["bandwidth"]["mpc"]
    assert mass["info"]["bandwidth"]["source"].startswith("Scott")
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


def test_bandwidth_is_scotts_rule_of_the_positions():
    rng = np.random.default_rng(2)
    x, y = rng.normal(0.0, 5.0, 800), rng.normal(0.0, 3.0, 800)
    kde = gaussian_kde(np.vstack([x, y]))
    assert conc.scott_bandwidth(x, y) == pytest.approx(np.sqrt(np.std(x) * np.std(y)) * kde.factor)
    df = conc_frame(blobs=BLOBS)
    _, result = _run(df, bandwidth_mpc=None)
    assert result["info"]["bandwidth"]["mpc"] == pytest.approx(
        result["info"]["bandwidth"]["scott_mpc"])
    _, fixed = _run(df, bandwidth_mpc=1.25)
    assert fixed["info"]["bandwidth"]["mpc"] == 1.25 and fixed["info"]["pixel_mpc"] == pytest.approx(
        1.25 / conc.PIXELS_PER_BANDWIDTH)


def test_sample_mask_defaults_and_errors(bridged):
    df = bridged[0].assign(z_type=np.where(np.arange(len(bridged[0])) % 3 == 0, "phot", "spec"))
    s = conc.conc_settings({"concentration": {"z_range": [0.019, 0.021], "sample": "spec"}})
    keep = conc.default_sample(df, s)
    z = df["z_best"].to_numpy()
    np.testing.assert_array_equal(keep, (z >= 0.019) & (z <= 0.021) & (df["z_type"] == "spec"))
    assert conc.default_sample(df, conc.conc_settings({})).all()
    assert conc.conc_settings({}, bandwidth_mpc=None, weight="count")["weight"] == "count"
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
        _run(df, bandwidth_mpc=0.01)


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
    assert "contour levels" in lines[1] and "Peaks: 2 significant" in lines[2]
    assert "ridges: 1 segments, 0 branch points, 0 open to the border" in lines[2]
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
                            weight_clip=None, bandwidth_mpc=2.0, pixel_mpc=0.3,
                            contour_step_sigma=0.5, min_prominence_sigma=0))
    validate_config(_config(weight="log1p_delta_sigma2d_k5"))
    bad = [({"sample": "phot"}, "sample"), ({"z_range": [0.03, 0.02]}, "z_range"),
           ({"weight": ""}, "weight"), ({"weight": 3}, "weight"),
           ({"weight_clip": 0}, "weight_clip"), ({"weight_clip": 101}, "weight_clip"),
           ({"bandwidth_mpc": 0}, "bandwidth_mpc"), ({"pixel_mpc": "x"}, "pixel_mpc"),
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
    h = result["info"]["bandwidth"]["mpc"]
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
        assert path["clamped_fraction"] == 0.0         # a crest within reach all along
        assert path["shift_max_mpc"] <= conc.SNAP_MAX * h + 1e-6
        assert len(path["input_ra"]) == 13 and path["crest_n_sigma_mean"] > 1.0
    assert model["skipped"] == [] and model["info"]["n_paths"] == 1


def test_the_width_of_a_bridge_is_its_cross_section_convolved_with_the_bandwidth(straight):
    df, columns, result = straight
    h = result["info"]["bandwidth"]["mpc"]
    t = np.linspace(-5.0, 5.0, 11)
    spec = {"paths": [_path_spec(np.column_stack([t, np.full_like(t, 0.3)]), "ridge A")]}
    _, model = conc.compute_model(df, result, spec)
    path = model["paths"][0]
    expected = np.sqrt(SIGMA_B ** 2 + h ** 2)
    assert path["width_median_mpc"] == pytest.approx(expected, rel=0.2)
    assert path["fit"]["ok"] and path["fit"]["s"] == pytest.approx(expected, rel=0.2)
    assert path["width_intrinsic_median_mpc"] == pytest.approx(SIGMA_B, abs=0.3)
    assert len(path["width_mpc"]) == len(path["x_mpc"]) and min(path["width_mpc"]) > 0.5 * h
    assert path["length_mpc"] == pytest.approx(10.0, abs=1.0)
    assert np.abs(_truth(path["ra"], path["dec"])[:, 1]).max() < 0.1 * h   # on the straight crest
    # the band edges are the path moved by the width on either side
    g = result["grid"]
    centre = np.column_stack(conc.tangent_plane(path["ra"], path["dec"], g["ra0"], g["dec0"])) * D_A
    for side in ("left", "right"):
        edge = np.column_stack(conc.tangent_plane(path[side + "_ra"], path[side + "_dec"],
                                                  g["ra0"], g["dec0"])) * D_A
        np.testing.assert_allclose(np.hypot(*(edge - centre).T), path["width_mpc"], rtol=1e-3)
    assert len(path["left_ra"]) == len(path["right_dec"]) == len(path["ra"])


def test_a_selected_peak_is_refined_to_the_blob_and_its_width_is_the_blob_convolved():
    df = conc_frame(blobs=[(2.0, 1.0, 0.8, 800)])
    _, result = _run(df)
    h, pixel = result["info"]["bandwidth"]["mpc"], result["grid"]["pixel"]
    assert len(result["peaks"]) == 1
    ra, dec = _sky(2.0 + 0.5 * h, 1.0)                  # given 0.5 h off the blob
    columns, model = conc.compute_model(df, result, {"peaks": [[float(ra), float(dec)]]})
    peak = model["peaks"][0]
    assert peak["label"] == "peak 1" and model["paths"] == [] and model["skipped"] == []
    assert np.hypot(*(_truth(peak["ra"], peak["dec"])[0] - [2.0, 1.0])) < pixel     # within a cell
    assert 0.2 * h < peak["shift_mpc"] < 0.8 * h                  # moved from 0.5 h off the blob
    assert peak["height"] == pytest.approx(result["top"], rel=0.05) and peak["n_sigma"] > 5.0
    expected = np.sqrt(0.8 ** 2 + h ** 2)
    assert peak["width_mpc"] == pytest.approx(expected, rel=0.2) and peak["fit_ok"]
    assert peak["width_intrinsic_mpc"] == pytest.approx(0.8, abs=0.35)
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
        ex.select({**filters, "conc_params": {"bandwidth_mpc": 1.5}}),
        base & ex.model(base, spec, bandwidth_mpc=1.5)[0]["model_inside"])
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
    h = result["info"]["bandwidth"]["mpc"]
    assert conc.SNAP_MAX == 1.0
    t = np.linspace(-3.0, 3.0, 7)
    drawn = np.column_stack([t, np.full_like(t, 2.0 * h)])       # 2 h beside the bridge
    _, model = conc.compute_model(df, result, {"paths": [_path_spec(drawn)]})
    path = model["paths"][0]
    assert path["shift_max_mpc"] <= conc.SNAP_MAX * h + 1e-6
    assert path["shift_max_mpc"] > 0.9 * conc.SNAP_MAX * h       # it went as far as it could
    assert path["clamped_fraction"] > 0.0
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
    h = result["info"]["bandwidth"]["mpc"]
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
    h = result["info"]["bandwidth"]["mpc"]
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
