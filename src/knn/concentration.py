"""Projected concentration map of a redshift slice: density contours, peaks and ridges.

compute_concentration(df, sample, settings, cosmo, z_cluster) returns the per-galaxy
columns of the build (prefix conc_) and a result dictionary (map, contours, peaks,
ridge network, summary); contour_table, peak_table, ridge_table, node_table, hull_sky
and sky_grid turn the result into the polylines and the colour map of the web app.
compute_model(df, result, spec) fits the density profile around peaks and paths chosen
by the user (default_spec: all of them) and returns the model_ columns.

The equivalent of the legacy plot (a scipy gaussian_kde of RA, Dec over the whole
field, src/calculate_kn_neighbour.py plot_kn_neighbour), with a weight per galaxy.
The map is read like a mountain range: the centre of the cluster is the highest peak,
the contours are levels above the mean, and the ridges join the peaks along the
higher ground between them (every peak is connected, a ridge can branch and can run
off the map).

Map
    Galaxies of the sample are projected on the tangent plane (gnomonic) at the
    mean position of the catalogue and scaled to proper Mpc at the cluster distance
    D_A(z_cl); every galaxy is assumed to lie at that distance, so sizes are right
    only for a narrow redshift slice. A weighted 2D histogram is smoothed with a
    Gaussian of standard deviation h (the bandwidth); the result is a weight surface
    density (M_sun/Mpc^2 for the mass weight). The default h is Scott's rule as in
    scipy.stats.gaussian_kde, h = sigma n^(-1/(d+4)) with d = 2, n the number of
    galaxies and sigma the geometric mean of the standard deviations of their
    coordinates; it depends on the positions only, not on the weight.

Footprint
    The survey area is the convex hull of ALL the galaxies of the table (every row,
    any redshift) in the tangent plane, not of the galaxies of the slice. A cell
    whose centre is inside the hull is part of the map and is never masked: with no
    galaxy of the slice within reach it has density 0 (the darkest colour of a
    linear scale from 0), not NaN. A cell outside the hull is NaN. The smoothed
    weights are divided by the same smoothing of the footprint indicator (normalised
    convolution), which removes most of the dilution at the border of the hull; the
    divisor is floored at COVER_FLOOR = 0.2 so that the correction cannot blow up at
    a sharp corner of the hull (it can raise the density there at most 5 times).
    Weight that falls in a cell whose centre is outside the hull (a galaxy within
    half a cell of the hull) is smoothed like the rest.

Weight
    count: 1; mass: 10**logmstar x p_in_window (0 without a mass); any other
    name: that numeric column (values below 0 are shifted by the sample
    minimum; NaN count 0). weights are capped at the weight_clip percentile of
    the positive weights of the sample (null: no cap): the stellar mass of
    MKW4 is so skewed that the 10 most massive galaxies change the raw mass
    map by up to 1.6 times the median level (0.6 with the 95th percentile cap).

Levels
    mean and sigma are the mean and the standard deviation of the map over the
    footprint cells. The contour levels are mean + n sigma with n = 1, 1 + step,
    1 + 2 step, ... (step = contour_step_sigma, default 1) up to the map maximum, so
    the 1 sigma level always exists. If that gives more than MAX_LEVELS = 12 levels,
    the 1 sigma level and the highest one are kept and the others are thinned evenly
    (indices round(linspace(0, n - 1, 12))). Contour lines are made by contourpy on
    the grid with the cells outside the hull masked: a contour that reaches the
    border of the hull is an open line that ends there (within one cell), it is
    never closed along the border, as it would continue with data outside.

Peaks
    A peak is a local maximum of the map (8-connected). Its height is the map value;
    its prominence is the height minus the highest saddle that connects it to a
    higher peak, from the merge tree of the cells sorted by decreasing density
    (union-find, 8-connected; when two components meet at a cell, the one with the
    lower peak ends there). A peak is significant if its height is at least the 1
    sigma level and its prominence at least min_prominence_sigma x sigma (default 1:
    it has a contour of its own); the global maximum is always significant. A
    significant peak is open if its own region touches the border of the footprint:
    the region is the connected set of cells above the saddle where it ends (the 1
    sigma region for the global maximum and for a saddle below the 1 sigma level).

Ridges
    The merge tree is also the maximum spanning tree (MST) of the 8-connected graph of the
    cells with edge weight = the lower density of the two cells, so the path between two
    cells in it is the maximin (bottleneck) path, the one whose lowest point (the saddle)
    is highest: the crest between them. The union of the MST paths between all pairs of a
    set of cells is the Steiner subtree of the MST that spans them; the ridge network is
    that subtree for the terminals below, cut into segments. There is no level threshold:
    a segment is drawn whatever its saddle (kept, in n sigma, to style it), so every
    significant peak is connected to the rest, a ridge can pass through a branch point
    (a local maximum along the crest) to more than one peak, and a ridge can run off the
    map. With one significant peak and no border end there is no segment.
    Terminals: every significant peak, and the border ends. The border is the cells of
    the footprint with a neighbour outside it; its cells at or above the 1 sigma level
    are grouped in 8-connected components, and the highest cell of a component is a
    border end, unless a significant peak lies within BORDER_PEAK_CELLS = 2 cells
    (Chebyshev distance) of any cell of the component: that peak is then the open end.
    Nodes: the terminals and the junctions, the cells of the subtree (rooted at the
    global maximum) with two or more branches. A segment is the path in the subtree from
    a node up to the next node towards the root, both ends included. Its kind is 'peak'
    (both ends are peaks), 'junction' (an end is a junction, none a border end) or
    'boundary' (an end is a border end; the segment is open). Segments are lines, smoothed
    with a 5-point mean (ends fixed), without a width: the model has the widths.

Inside
    A galaxy is inside when the map value at its position (bilinear, the map
    extended outside the hull by the nearest cell) is at least the 1 sigma level.
    Columns: conc_density (map value at the galaxy), conc_nsigma ((density - mean)
    / sigma), conc_inside, conc_region (id of the connected 1 sigma region, 0 for the
    one with the highest map value, -1 outside), conc_ridge_dist_mpc (distance to the
    nearest ridge segment, NaN without one). They are computed for every row of the table, also
    for galaxies outside the sample, against the map of the sample.

Model
    The user picks peaks (positions on the sky) and paths (the segments of the network
    or polylines drawn on the sky); compute_model snaps and fits them, all lengths in
    units of the bandwidth h of the map (the map is the true profile convolved with h).
    Peak: the highest footprint cell within PEAK_SEARCH = 1 h of the given position,
    moved to the vertex of a 3-point parabola in x and in y (at most half a cell). The
    mean map in annuli of one cell out to PROFILE_RADIUS = 4 h is fitted with
    B + A exp(-r^2 / (2 s^2)) (A, B >= 0, half a cell <= s <= 4 h; a failed fit uses the
    half-maximum radius over sqrt(2 ln 2)). The 1 sigma region of the peak is the disc of
    radius s; the intrinsic width is sqrt(s^2 - h^2).
    Path: the points inside the survey area, smoothed lightly (5-point mean) and resampled
    at one cell: points q_k with unit normals n_k that stay fixed. Each point takes the
    profile of the map along n_k within SNAP_MAX = 1 h either side (-inf outside the hull)
    and the local maximum of it nearest to the drawn line (3-point parabola). A profile with
    no interior maximum (the map only rises to one side: no crest within reach) takes its
    highest sample, at most SNAP_MAX h from q_k, and the point is flagged (clamped_fraction
    is the share of flagged points). The offsets d_k are regularised along the path
    (running median over WINDOW = 2 h of arc length, then the 5-point mean), searched again
    within one cell of the result and smoothed once more; the fitted points are
    q_k + d_k n_k, inside the hull. Points never slide along the path, so it cannot
    retract, and shift_max_mpc = max |d_k| <= SNAP_MAX h. The ends within ENDPOINT_SNAP = 1 h
    of a selected peak are pinned to it (end_pin_mpc: the distance from the drawn end; the
    pinned ends are not in shift_max_mpc).
    The profile across the path (to PROFILE_HALF = 3 h each side) averaged
    over the points within WINDOW = 2 h of arc length is fitted with the same Gaussian
    (centred); fits that fail or hit a bound are interpolated along the path and the
    widths smoothed. The 1 sigma band is the set of points within s of the path. The band
    edges and the circles of the peaks are drawn only inside the survey area (NaN outside);
    the fitted path is kept inside it.
    Columns model_dist_mpc (distance to the nearest structure), model_dist_sigma (the
    smallest distance / s of a structure), model_inside (model_dist_sigma <= 1: inside the
    band of any selected structure), model_structure (the label of that structure), for
    every row of the table.
"""

import warnings

import astropy.units as u
import contourpy
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.optimize import OptimizeWarning, curve_fit
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import breadth_first_order
from scipy.spatial import ConvexHull, cKDTree

WEIGHT_SPECIAL = ("count", "mass")
SAMPLES = ("all", "spec")
CONC_DEFAULTS = {"sample": "all", "z_range": None, "weight": "mass", "weight_clip": 95.0,
                 "bandwidth_mpc": None, "pixel_mpc": None, "contour_step_sigma": 1.0,
                 "min_prominence_sigma": 1.0}
PIXELS_PER_BANDWIDTH = 4.0
MARGIN_PIXELS = 3
COVER_FLOOR = 0.2
MAX_LEVELS = 12
MAX_CELLS = 2_000_000
MIN_SAMPLE = 30
RIDGE_POINT_STEP = 0.25
BORDER_PEAK_CELLS = 2
PEAK_SEARCH = 1.0
PROFILE_RADIUS = 4.0
SNAP_MAX = 1.0
ENDPOINT_SNAP = 1.0
PROFILE_HALF = 3.0
WINDOW = 2.0
CIRCLE_POINTS = 64
HULL_EDGE_POINTS = 24
COLUMNS = ("conc_density", "conc_nsigma", "conc_inside", "conc_region", "conc_ridge_dist_mpc")
MODEL_COLUMNS = ("model_dist_mpc", "model_dist_sigma", "model_inside", "model_structure")


class ConcentrationError(ValueError):
    """Raised when the map cannot be built from the inputs."""


def conc_settings(cfg, **overrides):
    """The 'concentration' section with defaults filled in (CONC_DEFAULTS), then overrides.

    Unknown override keys raise KeyError; None overrides are ignored.
    """
    settings = {**CONC_DEFAULTS, **(cfg.get("concentration") or {})}
    for key, value in overrides.items():
        if key not in CONC_DEFAULTS:
            raise KeyError(f"unknown concentration setting {key!r}; known: {tuple(CONC_DEFAULTS)}")
        if value is not None:
            settings[key] = value
    return settings


def default_sample(df, settings):
    """Boolean mask of the build sample: z_best inside z_range, spec-z only for sample 'spec'."""
    keep = np.ones(len(df), dtype=bool)
    if settings["z_range"] is not None:
        z = df["z_best"].to_numpy(float)
        keep &= (z >= settings["z_range"][0]) & (z <= settings["z_range"][1])
    if settings["sample"] == "spec":
        keep &= df["z_type"].to_numpy(object) == "spec"
    return keep


def tangent_plane(ra, dec, ra0, dec0):
    """Gnomonic projection about (ra0, dec0), all in degrees.

    Returns (xi, eta) in radians: xi towards increasing RA (east), eta north.
    """
    ra, dec = np.radians(np.asarray(ra, dtype=float)), np.radians(np.asarray(dec, dtype=float))
    r0, d0 = np.radians(ra0), np.radians(dec0)
    cos_c = np.sin(d0) * np.sin(dec) + np.cos(d0) * np.cos(dec) * np.cos(ra - r0)
    xi = np.cos(dec) * np.sin(ra - r0) / cos_c
    eta = (np.cos(d0) * np.sin(dec) - np.sin(d0) * np.cos(dec) * np.cos(ra - r0)) / cos_c
    return xi, eta


def inverse_tangent_plane(xi, eta, ra0, dec0):
    """(ra, dec) in degrees of tangent-plane coordinates (radians) about (ra0, dec0)."""
    xi, eta = np.asarray(xi, dtype=float), np.asarray(eta, dtype=float)
    r0, d0 = np.radians(ra0), np.radians(dec0)
    rho = np.hypot(xi, eta)
    c = np.arctan(rho)
    ratio = np.where(rho > 0, np.sin(c) / np.where(rho > 0, rho, 1.0), 1.0)
    dec = np.arcsin(np.cos(c) * np.sin(d0) + eta * ratio * np.cos(d0))
    ra = r0 + np.arctan2(xi * np.sin(c), rho * np.cos(d0) * np.cos(c)
                         - eta * np.sin(d0) * np.sin(c))
    return np.degrees(ra) % 360.0, np.degrees(dec)


def galaxy_weights(df, sample, weight, clip=None, values=None):
    """(weights, info): the weight of every row, 0 outside the sample.

    weight 'count' = 1, 'mass' = 10**logmstar x p_in_window (0 without a mass),
    any other name = the numeric column of that name, or values (an array with
    one entry per row) when given; NaN count 0, and if the column has values
    below 0 the minimum over the sample is subtracted. With clip (a percentile,
    0 < clip < 100) the weights are capped at that percentile of the positive
    weights of the sample. info: clip value, n_eff of the sample.
    """
    n = len(df)
    sample = np.asarray(sample, dtype=bool)
    if weight == "count":
        w = np.ones(n)
    elif weight == "mass":
        if "logmstar" not in df.columns:
            raise ConcentrationError("weight 'mass' needs the column 'logmstar' (a "
                                     "'photometry' section in the config)")
        p = (df["p_in_window"].to_numpy(float) if "p_in_window" in df.columns
             else np.ones(n))
        p = np.where(np.isfinite(p), p, 1.0)
        logm = df["logmstar"].to_numpy(float)
        w = np.where(np.isfinite(logm), 10.0 ** np.where(np.isfinite(logm), logm, 0.0), 0.0) * p
    else:
        if values is None:
            if weight not in df.columns:
                raise ConcentrationError(f"weight column {weight!r} not in the table")
            values = df[weight]
        w = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
        if len(w) != n:
            raise ConcentrationError("weight values need one entry per row")
        low = np.nanmin(np.where(sample, w, np.nan)) if (sample & np.isfinite(w)).any() else 0.0
        if low < 0:
            w = w - low
        w = np.where(np.isfinite(w), w, 0.0)
    w = np.where(sample, w, 0.0)
    cap = None
    positive = w[sample & (w > 0)]
    if clip is not None and positive.size:
        cap = float(np.percentile(positive, clip))
        w = np.minimum(w, cap)
    used = w[sample]
    n_eff = float(used.sum() ** 2 / (used ** 2).sum()) if (used ** 2).sum() > 0 else 0.0
    return w, {"clip_value": cap, "n_eff": n_eff}


def scott_bandwidth(x, y):
    """Scott's rule h = sigma n^(-1/6) (scipy gaussian_kde, d = 2).

    sigma is the geometric mean of the standard deviations of x and y.
    """
    n = len(x)
    return float(np.sqrt(np.std(x) * np.std(y)) * n ** (-1.0 / 6.0))


def survey_hull(x, y):
    """Convex hull of the points as a dict: vertices (m, 2) counter-clockwise, equations (m, 3).

    A point p is inside when equations[:, :2] @ p + equations[:, 2] <= 0 for every row;
    area is in the units of x, y squared.
    """
    points = np.column_stack([x, y])
    hull = ConvexHull(points)
    return {"vertices": points[hull.vertices], "equations": hull.equations,
            "area": float(hull.volume)}


def in_hull(hull, px, py, tol=1e-9):
    """Boolean array: the points (px, py) are inside the hull (within tol of its border)."""
    eq = hull["equations"]
    pts = np.column_stack([np.ravel(px), np.ravel(py)])
    inside = np.all(pts @ eq[:, :2].T + eq[:, 2] <= tol, axis=1)
    return inside.reshape(np.shape(px))


def density_map(x, y, w, hull, h, pixel):
    """Edge-corrected Gaussian-smoothed weighted map on a regular grid (Mpc).

    x, y (n,) positions of the galaxies of the sample, w (n,) weights, hull the
    survey hull (survey_hull), h the kernel standard deviation, pixel the cell
    size. The grid covers the hull with MARGIN_PIXELS empty cells around it, so no
    cell of the footprint touches the edge of the array. Returns a dict: dens (ny,
    nx) weight per Mpc^2 (NaN outside the hull, 0 for an empty cell inside), footprint
    (bool: cell centre inside the hull), x0, y0 (lower left corner), pixel.
    """
    vx, vy = hull["vertices"][:, 0], hull["vertices"][:, 1]
    x0, y0 = float(vx.min() - MARGIN_PIXELS * pixel), float(vy.min() - MARGIN_PIXELS * pixel)
    nx = int(np.ceil((vx.max() + MARGIN_PIXELS * pixel - x0) / pixel))
    ny = int(np.ceil((vy.max() + MARGIN_PIXELS * pixel - y0) / pixel))
    if nx * ny > MAX_CELLS:
        raise ConcentrationError(f"the map would have {nx * ny} cells (maximum {MAX_CELLS}): "
                                 "use a larger bandwidth or pixel")
    gx, gy = np.meshgrid(x0 + (np.arange(nx) + 0.5) * pixel, y0 + (np.arange(ny) + 0.5) * pixel)
    footprint = in_hull(hull, gx, gy)
    box = [[y0, y0 + ny * pixel], [x0, x0 + nx * pixel]]
    weighted = np.histogram2d(y, x, bins=[ny, nx], range=box, weights=w)[0]
    sigma = h / pixel
    num = ndimage.gaussian_filter(weighted, sigma, mode="constant")
    cover = ndimage.gaussian_filter(footprint.astype(float), sigma, mode="constant")
    dens = np.maximum(num, 0.0) / np.maximum(cover, COVER_FLOOR) / pixel ** 2
    return {"dens": np.where(footprint, dens, np.nan), "footprint": footprint,
            "x0": x0, "y0": y0, "pixel": float(pixel)}


def contour_levels(mean, sigma, top, step, max_levels=MAX_LEVELS):
    """(n_sigma, levels): the contour levels mean + n sigma up to top, thinned to max_levels.

    n = 1, 1 + step, 1 + 2 step, ... while mean + n sigma <= top (empty if top is below the 1
    sigma level). With more than max_levels, the first and the last are kept and the others
    are the indices round(linspace(0, count - 1, max_levels)).
    """
    count = 0
    if top >= mean + sigma:
        count = int(np.floor(((top - mean) / sigma - 1.0) / step + 1e-9)) + 1
    n_sigma = 1.0 + step * np.arange(max(count, 0))
    if len(n_sigma) > max_levels:
        keep = np.unique(np.round(np.linspace(0, len(n_sigma) - 1, max_levels)).astype(int))
        n_sigma = n_sigma[keep]
    return n_sigma, mean + sigma * n_sigma


def contour_lines(dens, footprint, grid, levels):
    """Contour lines of the map: a list (one per level) of lists of (m, 2) arrays in Mpc.

    contourpy on the grid with the cells outside the footprint masked: a line that reaches
    the border of the footprint ends there (open); a closed line repeats its first point.
    """
    ny, nx = dens.shape
    xs = grid["x0"] + (np.arange(nx) + 0.5) * grid["pixel"]
    ys = grid["y0"] + (np.arange(ny) + 0.5) * grid["pixel"]
    z = np.ma.masked_array(np.where(footprint, dens, 0.0), mask=~footprint)
    gen = contourpy.contour_generator(xs, ys, z, name="serial", line_type="Separate",
                                      corner_mask=True)
    return [[np.asarray(line, dtype=float) for line in gen.lines(float(level))]
            for level in levels]


def is_closed(line):
    """True for a contour line that returns to its first point."""
    return len(line) > 3 and bool(np.allclose(line[0], line[-1]))


def merge_tree(dens, footprint, level):
    """Peaks of the map, their prominence and the maximum spanning tree of the cells.

    dens (ny, nx), footprint (bool, no True on the outermost rows and columns). The cells
    of the footprint are taken by decreasing density (ties in array order) and joined
    to their 8-connected neighbours already taken (union-find). A cell with no such
    neighbour is a peak. When a cell joins several components, the one with the highest
    peak (the earliest on a tie) goes on and each of the others ends there: its saddle
    is the density of the cell and its parent peak that of the winner. Tree edges: the
    cell to its highest neighbour of every component it joins (the maximum spanning
    tree for edge weight = the lower density of the two cells).
    level: the density of the 1 sigma level, for the border flag.

    Returns a dict of arrays, one entry per peak, in order of decreasing height: cell
    (flat index into dens), height, saddle (NaN for a peak that never ends), parent
    (index of the higher peak it ends in, -1), prominence (height - saddle; height - the
    lowest density of the footprint for a peak that never ends), open (its own region
    touches the border of the footprint, see the module docstring); edges (two arrays
    of flat indices, the tree edges).
    """
    ny, nx = dens.shape
    if footprint[0].any() or footprint[-1].any() or footprint[:, 0].any() or footprint[:, -1].any():
        raise ValueError("the footprint must not touch the edge of the array")
    flat = np.where(footprint, dens, -np.inf).ravel()
    cells = np.flatnonzero(footprint.ravel())
    cells = cells[np.argsort(-flat[cells], kind="stable")]
    border = (footprint & ~ndimage.binary_erosion(footprint, structure=np.ones((3, 3)),
                                                  border_value=0)).ravel().tolist()
    value = flat.tolist()
    offsets = (-nx - 1, -nx, -nx + 1, -1, 1, nx - 1, nx, nx + 1)
    uf = list(range(flat.size))
    done = bytearray(flat.size)
    peak_of, touch = {}, {}
    cell, height, saddle, parent, died_open = [], [], [], [], []
    snapshot = None
    edge_a, edge_b = [], []

    def find(a):
        while uf[a] != a:
            uf[a] = uf[uf[a]]
            a = uf[a]
        return a

    for i in cells.tolist():
        v = value[i]
        if snapshot is None and v < level:
            snapshot = [touch[find(c)] for c in cell]
        found = {}
        for d in offsets:
            j = i + d
            if done[j]:
                r = find(j)
                if r not in found or value[j] > value[found[r]]:
                    found[r] = j
        done[i] = 1
        if not found:
            peak_of[i], touch[i] = len(cell), border[i]
            cell.append(i)
            height.append(v)
            saddle.append(np.nan)
            parent.append(-1)
            died_open.append(False)
            continue
        roots = sorted(found, key=lambda r: (-height[peak_of[r]], peak_of[r]))
        win, t = roots[0], border[i]
        for r in roots:
            edge_a.append(i)
            edge_b.append(found[r])
            t = t or touch[r]
            if r != win:
                q = peak_of[r]
                saddle[q], parent[q], died_open[q] = v, peak_of[win], touch[r]
                uf[r] = win
        uf[i] = win
        touch[win] = t
    if snapshot is None:
        snapshot = [touch[find(c)] for c in cell]
    height, saddle = np.array(height), np.array(saddle)
    low = float(flat[cells].min()) if len(cells) else 0.0
    prominence = height - np.where(np.isnan(saddle), low, saddle)
    own = np.array([died_open[p] if saddle[p] >= level else (snapshot[p] if p < len(snapshot)
                                                              else False)
                    for p in range(len(cell))], dtype=bool)
    return {"cell": np.array(cell, dtype=int), "height": height, "saddle": saddle,
            "parent": np.array(parent, dtype=int), "prominence": prominence, "open": own,
            "edges": (np.array(edge_a, dtype=int), np.array(edge_b, dtype=int))}


def _smooth(points):
    """5-point moving average along a polyline; the end points stay fixed."""
    m = len(points)
    out = points.copy()
    for k in range(m):
        half = min(2, k, m - 1 - k)
        if half:
            out[k] = points[k - half:k + half + 1].mean(axis=0)
    return out


def _length(points):
    return float(np.hypot(*np.diff(points, axis=0).T).sum()) if len(points) > 1 else 0.0


def ridge_network(dens, footprint, tree, peaks, level, grid, mean, sigma):
    """Ridge network of the significant peaks: nodes and the segments between them.

    peaks: the list of peak dicts of compute_concentration (cell, id, height; the first is
    the global maximum); level: the 1 sigma level; mean, sigma of the map. Each peak gets
    its 'degree' (number of segments at its node). Returns (ridges, nodes), lists of dicts
    with the keys documented in compute_concentration. See the module docstring for the
    rules: Steiner subtree of the maximum spanning tree over the peaks and the border
    ends, cut at the terminals and the junctions.
    """
    nx = dens.shape[1]
    flat = np.where(footprint, dens, -np.inf).ravel()
    a, b = tree["edges"]
    graph = coo_matrix((np.ones(2 * len(a)), (np.concatenate([a, b]), np.concatenate([b, a]))),
                       shape=(flat.size, flat.size)).tocsr()
    pixel = grid["pixel"]
    border = footprint & ~ndimage.binary_erosion(footprint, structure=np.ones((3, 3)),
                                                 border_value=0)
    labels, n_labels = ndimage.label(border & (np.where(footprint, dens, -np.inf) >= level),
                                     structure=np.ones((3, 3)))
    near = set()
    for p in peaks:
        row, col = divmod(p["cell"], nx)
        window = labels[max(row - BORDER_PEAK_CELLS, 0):row + BORDER_PEAK_CELLS + 1,
                        max(col - BORDER_PEAK_CELLS, 0):col + BORDER_PEAK_CELLS + 1]
        near.update(np.unique(window).tolist())
    free = [lab for lab in range(1, n_labels + 1) if lab not in near]
    kind = {p["cell"]: "peak" for p in peaks}
    if free:
        top = ndimage.maximum_position(np.where(footprint, dens, -np.inf), labels, index=free)
        kind.update({int(r) * nx + int(c): "border" for r, c in top})
    root = peaks[0]["cell"]
    pred = breadth_first_order(graph, root, directed=True, return_predecessors=True)[1]
    marked = {root}
    for cell in kind:
        path = []
        while cell not in marked and pred[cell] >= 0:
            path.append(cell)
            cell = int(pred[cell])
        if cell in marked:
            marked.update(path)
    branches = {}
    for cell in marked - {root}:
        branches[int(pred[cell])] = branches.get(int(pred[cell]), 0) + 1
    for cell in marked:
        if cell not in kind and branches.get(cell, 0) >= 2:
            kind[cell] = "junction"
    node_cells = {cell for cell in marked if cell in kind}
    peak_id = {p["cell"]: p["id"] for p in peaks}
    rank = {"peak": 0, "junction": 1, "border": 2}
    order = sorted(node_cells, key=lambda c: (rank[kind[c]], peak_id.get(c, 0), -flat[c], c))
    nodes = []
    for k, cell in enumerate(order, start=1):
        row, col = divmod(cell, nx)
        height = float(flat[cell])
        nodes.append({"id": k, "kind": kind[cell], "cell": int(cell),
                      "x": float(grid["x0"] + (col + 0.5) * pixel),
                      "y": float(grid["y0"] + (row + 0.5) * pixel), "peak": peak_id.get(cell),
                      "height": height, "n_sigma": (height - mean) / sigma, "degree": 0})
    node_of = {n["cell"]: n for n in nodes}

    def points_of(cells):
        rows, cols = np.divmod(np.array(cells), nx)
        return _smooth(np.column_stack([grid["x0"] + (cols + 0.5) * pixel,
                                        grid["y0"] + (rows + 0.5) * pixel]))

    segments = []
    for cell in sorted(node_cells - {root}):
        cells = [cell]
        while True:
            cells.append(int(pred[cells[-1]]))
            if cells[-1] in node_cells:
                break
        segments.append(cells)
    segments.sort(key=lambda cells: -float(flat[cells].min()))
    ridges = []
    for k, cells in enumerate(segments, start=1):
        low, up = node_of[cells[0]], node_of[cells[-1]]
        kinds = {low["kind"], up["kind"]}
        saddle = float(flat[cells].min())
        pts = points_of(cells)
        ridges.append({"id": k, "kind": ("boundary" if "border" in kinds else
                                         "junction" if "junction" in kinds else "peak"),
                       "open": "border" in kinds, "peaks": [low["peak"], up["peak"]],
                       "nodes": [low["id"], up["id"]], "saddle": saddle,
                       "saddle_n_sigma": (saddle - mean) / sigma, "points": pts,
                       "length": _length(pts), "cells": [int(c) for c in cells]})
        low["degree"] += 1
        up["degree"] += 1
    for p in peaks:
        p["degree"] = node_of[p["cell"]]["degree"] if p["cell"] in node_of else 0
    return ridges, nodes


def one_sigma_regions(dens, footprint, level):
    """(labels, n): the connected (8-neighbour) regions of the footprint at or above level.

    labels (ny, nx) is 0 outside them and 1..n inside, 1 being the region with the highest
    map value.
    """
    labels, n = ndimage.label((np.where(footprint, dens, -np.inf) >= level) & footprint,
                              structure=np.ones((3, 3)))
    if n > 1:
        top = ndimage.maximum(np.where(footprint, dens, -np.inf), labels, index=np.arange(1, n + 1))
        rank = np.empty(n + 1, dtype=int)
        rank[0] = 0
        rank[1 + np.argsort(-np.asarray(top), kind="stable")] = np.arange(1, n + 1)
        labels = rank[labels]
    return labels, n


def _filled(values, footprint):
    """values with the cells outside the footprint set to the nearest footprint cell."""
    idx = ndimage.distance_transform_edt(~footprint, return_distances=False,
                                         return_indices=True)
    return values[tuple(idx)]


def _bilinear(values, grid, px, py):
    """Bilinear value of a map at positions (Mpc); NaN outside the grid."""
    pixel = grid["pixel"]
    col = (px - grid["x0"]) / pixel - 0.5
    row = (py - grid["y0"]) / pixel - 0.5
    inside = np.isfinite(col) & np.isfinite(row) & (col >= 0) & (row >= 0) \
        & (col <= values.shape[1] - 1) & (row <= values.shape[0] - 1)
    out = np.full(len(px), np.nan)
    out[inside] = ndimage.map_coordinates(values, [row[inside], col[inside]], order=1)
    return out


def _densify(points, step):
    """The polyline resampled at (at most) step apart."""
    seg = np.hypot(*np.diff(points, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] <= 0:
        return points[:1]
    t = np.linspace(0.0, s[-1], max(2, int(np.ceil(s[-1] / step)) + 1))
    return np.column_stack([np.interp(t, s, points[:, 0]), np.interp(t, s, points[:, 1])])


def _galaxy_columns(grid, filled, labels, mean, sigma, level, ridges, x, y):
    n = len(x)
    density = _bilinear(filled, grid, x, y)
    with np.errstate(invalid="ignore"):
        inside = density >= level
    cols = {"conc_density": density, "conc_nsigma": (density - mean) / sigma,
            "conc_inside": inside, "conc_region": np.full(n, -1, dtype=int),
            "conc_ridge_dist_mpc": np.full(n, np.nan)}
    pixel = grid["pixel"]
    col = np.floor((x[inside] - grid["x0"]) / pixel - 0.5).astype(int)
    row = np.floor((y[inside] - grid["y0"]) / pixel - 0.5).astype(int)
    best_label = np.zeros(inside.sum(), dtype=int)
    best_value = np.full(inside.sum(), -np.inf)
    for dr in (0, 1):
        for dc in (0, 1):
            r = np.clip(row + dr, 0, filled.shape[0] - 1)
            c = np.clip(col + dc, 0, filled.shape[1] - 1)
            better = (labels[r, c] > 0) & (filled[r, c] > best_value)
            best_label = np.where(better, labels[r, c], best_label)
            best_value = np.where(better, filled[r, c], best_value)
    cols["conc_region"][inside] = np.where(best_label > 0, best_label - 1, -1)
    if ridges:
        pts = np.vstack([_densify(r["points"], RIDGE_POINT_STEP * pixel) for r in ridges])
        good = np.isfinite(x) & np.isfinite(y)
        dist = np.full(n, np.nan)
        dist[good] = cKDTree(pts).query(np.column_stack([x[good], y[good]]))[0]
        cols["conc_ridge_dist_mpc"] = dist
    return cols


def _to_sky(points, grid):
    return inverse_tangent_plane(points[:, 0] / grid["d_a"], points[:, 1] / grid["d_a"],
                                 grid["ra0"], grid["dec0"])


def compute_concentration(df, sample, settings, cosmo, z_cluster, values=None):
    """Map, contours, peaks, ridges and per-galaxy columns for the galaxies of a sample.

    df: table with ra, dec, z_type and, for weight 'mass', logmstar and p_in_window; every
    row of it, any redshift, sets the footprint (the convex hull of the positions).
    sample: boolean mask of the galaxies that make the map (at least MIN_SAMPLE).
    settings: conc_settings(cfg, ...). cosmo: astropy cosmology; z_cluster: redshift
    giving the Mpc per degree. values: array with one entry per row for a weight column
    that is not in df (the live environment). See the module docstring for the method.

    Returns (columns, result). columns (arrays of len(df), for every row, not only the
    sample): COLUMNS. result: dict with grid (x0, y0, pixel, shape, ra0, dec0, d_a),
    dens (ny, nx; NaN outside the hull), filled (dens with the nearest footprint value
    outside the hull), footprint (bool), hull (survey_hull), mean, sigma, level (the 1
    sigma level), top (map maximum), n_sigma and levels (arrays), contours (one dict per
    level: n_sigma, level, lines (list of (m, 2) arrays in Mpc), closed (list of bool)),
    peaks (the significant peaks: id from 1 by height, cell, row, col, x, y in Mpc,
    height, n_sigma, prominence, prominence_sigma, open, significant), all_peaks
    (every local maximum, same keys, id 0 when not significant; the significant ones have
    degree, the number of ridge segments at their node), ridges (the segments of the ridge
    network, id from 1 by decreasing saddle: kind 'peak' | 'junction' | 'boundary', open,
    peaks (ids of the two ends, lower end first, None for a junction or border end), nodes
    (ids of the two ends), saddle, saddle_n_sigma, points ((m, 2) Mpc, smoothed), length,
    cells (flat indices of the unsmoothed path)), nodes (the ends of the segments, id from
    1: kind 'peak' | 'junction' | 'border', cell, x, y (Mpc), peak (id or None), height,
    n_sigma, degree), regions (label image of the 1 sigma regions), info
    (JSON-serialisable summary).
    """
    sample = np.asarray(sample, dtype=bool)
    if int(sample.sum()) < MIN_SAMPLE:
        raise ConcentrationError(f"only {int(sample.sum())} galaxies in the sample (need "
                                 f"{MIN_SAMPLE}): cannot build the map")
    step, min_prom = float(settings["contour_step_sigma"]), float(settings["min_prominence_sigma"])
    if step <= 0 or min_prom < 0:
        raise ConcentrationError("contour_step_sigma must be > 0 and min_prominence_sigma >= 0")
    ra, dec = df["ra"].to_numpy(float), df["dec"].to_numpy(float)
    finite = np.isfinite(ra) & np.isfinite(dec)
    d_a = float(cosmo.angular_diameter_distance(z_cluster).to_value(u.Mpc))
    ra0, dec0 = float(ra[finite].mean()), float(dec[finite].mean())
    xi, eta = tangent_plane(ra, dec, ra0, dec0)
    x, y = xi * d_a, eta * d_a
    hull = survey_hull(x[finite], y[finite])
    w, winfo = galaxy_weights(df, sample, settings["weight"], settings["weight_clip"], values)
    if not (w > 0).any():
        raise ConcentrationError("all weights are zero in the sample")
    use = sample & finite
    h_scott = scott_bandwidth(x[use], y[use])
    h = float(settings["bandwidth_mpc"]) if settings["bandwidth_mpc"] else h_scott
    pixel = float(settings["pixel_mpc"]) if settings["pixel_mpc"] else h / PIXELS_PER_BANDWIDTH
    grid = density_map(x[use], y[use], w[use], hull, h, pixel)
    dens, footprint = grid["dens"], grid["footprint"]
    mean, sigma = float(dens[footprint].mean()), float(dens[footprint].std())
    if not sigma > 0:
        raise ConcentrationError("the map is flat (zero standard deviation)")
    level, top = mean + sigma, float(dens[footprint].max())
    n_sigma, levels = contour_levels(mean, sigma, top, step)
    lines = contour_lines(dens, footprint, grid, levels)
    contours = [{"n_sigma": float(n), "level": float(lv), "lines": ls,
                 "closed": [is_closed(line) for line in ls]}
                for n, lv, ls in zip(n_sigma, levels, lines)]

    tree = merge_tree(dens, footprint, level)
    ny, nx = dens.shape
    rows, cols = np.divmod(tree["cell"], nx)
    all_peaks, next_id = [], 1
    for p in range(len(rows)):
        height, prom = float(tree["height"][p]), float(tree["prominence"][p])
        significant = p == 0 or (height >= level and prom >= min_prom * sigma)
        all_peaks.append({
            "id": 0, "cell": int(tree["cell"][p]), "row": int(rows[p]), "col": int(cols[p]),
            "x": float(grid["x0"] + (cols[p] + 0.5) * pixel),
            "y": float(grid["y0"] + (rows[p] + 0.5) * pixel), "height": height,
            "n_sigma": (height - mean) / sigma, "prominence": prom,
            "prominence_sigma": prom / sigma, "open": bool(tree["open"][p]),
            "significant": bool(significant), "degree": 0})
        if significant:
            all_peaks[-1]["id"] = next_id
            next_id += 1
    peaks = [p for p in all_peaks if p["significant"]]
    regions, n_regions = one_sigma_regions(dens, footprint, level)
    ridges, nodes = ridge_network(dens, footprint, tree, peaks, level, grid, mean, sigma)
    filled = _filled(dens, footprint)
    filled_labels = _filled(regions, footprint)
    columns = _galaxy_columns(grid, filled, filled_labels, mean, sigma, level, ridges, x, y)
    columns["conc_inside"] = np.where(finite, columns["conc_inside"], False).astype(bool)
    grid_info = {"x0": grid["x0"], "y0": grid["y0"], "pixel": pixel, "shape": list(dens.shape),
                 "ra0": ra0, "dec0": dec0, "d_a": d_a}
    result = {"grid": grid_info, "dens": dens, "filled": filled, "footprint": footprint,
              "hull": hull, "mean": mean, "sigma": sigma, "level": float(level), "top": top,
              "n_sigma": n_sigma, "levels": levels, "contours": contours, "peaks": peaks,
              "all_peaks": all_peaks, "ridges": ridges, "nodes": nodes, "regions": regions,
              "info": None}
    spec = df["z_type"].to_numpy(object) == "spec"
    inside = columns["conc_inside"]
    peak_ra, peak_dec = _to_sky(np.array([[p["x"], p["y"]] for p in peaks]), grid_info)
    result["info"] = {
        "settings": {k: (list(v) if isinstance(v, tuple) else v) for k, v in settings.items()},
        "z_cluster": float(z_cluster), "d_a_mpc": d_a, "n_sample": int(sample.sum()),
        "n_weighted": int((sample & (w > 0)).sum()), "n_eff": winfo["n_eff"],
        "weight_clip_value": winfo["clip_value"], "n_rows": int(len(df)),
        "bandwidth": {"mpc": h, "scott_mpc": h_scott,
                      "source": ("config / app bandwidth" if settings["bandwidth_mpc"]
                                 else "Scott's rule on the sample positions")},
        "pixel_mpc": pixel, "cover_floor": COVER_FLOOR,
        "hull_area_mpc2": hull["area"], "map_area_mpc2": float(footprint.sum() * pixel ** 2),
        "mean": mean, "sigma": sigma, "level_1sigma": float(level), "top": top,
        "top_n_sigma": (top - mean) / sigma,
        "contour_levels_n_sigma": [float(v) for v in n_sigma],
        "contour_levels": [float(v) for v in levels],
        "n_peaks": len(peaks), "n_local_maxima": len(all_peaks),
        "peaks": [{"id": p["id"], "ra": float(r), "dec": float(d), "n_sigma": p["n_sigma"],
                   "prominence_sigma": p["prominence_sigma"], "open": p["open"]}
                  for p, r, d in zip(peaks, np.atleast_1d(peak_ra), np.atleast_1d(peak_dec))],
        "n_ridges": len(ridges), "n_ridges_open": sum(r["open"] for r in ridges),
        "ridges_total_length_mpc": float(sum(r["length"] for r in ridges)),
        "n_junctions": sum(n["kind"] == "junction" for n in nodes),
        "n_border_ends": sum(n["kind"] == "border" for n in nodes),
        "n_regions": int(n_regions),
        "counts": {
            "inside_spec": int((inside & spec & sample).sum()),
            "inside_phot": int((inside & ~spec & sample).sum()),
            "outside_spec": int((~inside & spec & sample).sum()),
            "outside_phot": int((~inside & ~spec & sample).sum())},
    }
    return columns, result


def contour_table(result):
    """JSON-serialisable contours of a result: one dict per level with its lines on the sky.

    Keys: n_sigma, level, lines (list of dicts: ra, dec, closed).
    """
    g = result["grid"]
    out = []
    for c in result["contours"]:
        lines = []
        for line, closed in zip(c["lines"], c["closed"]):
            ra, dec = _to_sky(line, g)
            lines.append({"ra": ra.tolist(), "dec": dec.tolist(), "closed": bool(closed)})
        out.append({"n_sigma": c["n_sigma"], "level": c["level"], "lines": lines})
    return out


def peak_table(result):
    """JSON-serialisable significant peaks: id, ra, dec, x_mpc, y_mpc, n_sigma, prominence_sigma,
    open."""
    g = result["grid"]
    out = []
    for p in result["peaks"]:
        ra, dec = _to_sky(np.array([[p["x"], p["y"]]]), g)
        out.append({"id": p["id"], "ra": float(ra[0]), "dec": float(dec[0]), "x_mpc": p["x"],
                    "y_mpc": p["y"], "n_sigma": p["n_sigma"],
                    "prominence_sigma": p["prominence_sigma"], "open": p["open"]})
    return out


def ridge_table(result):
    """JSON-serialisable ridge segments: id, kind, open, peaks (ids), nodes (ids), saddle_n_sigma,
    length_mpc, ra, dec, x_mpc, y_mpc."""
    g = result["grid"]
    out = []
    for r in result["ridges"]:
        ra, dec = _to_sky(r["points"], g)
        out.append({"id": r["id"], "kind": r["kind"], "open": bool(r["open"]),
                    "peaks": list(r["peaks"]), "nodes": list(r["nodes"]),
                    "saddle_n_sigma": r["saddle_n_sigma"],
                    "length_mpc": float(r["length"]), "ra": ra.tolist(), "dec": dec.tolist(),
                    "x_mpc": r["points"][:, 0].tolist(), "y_mpc": r["points"][:, 1].tolist()})
    return out


def node_table(result):
    """JSON-serialisable nodes of the ridge network: id, kind, peak (id or None), ra, dec, x_mpc,
    y_mpc, n_sigma, degree."""
    g = result["grid"]
    out = []
    for n in result["nodes"]:
        ra, dec = _to_sky(np.array([[n["x"], n["y"]]]), g)
        out.append({"id": n["id"], "kind": n["kind"], "peak": n["peak"], "ra": float(ra[0]),
                    "dec": float(dec[0]), "x_mpc": n["x"], "y_mpc": n["y"],
                    "n_sigma": n["n_sigma"], "degree": n["degree"]})
    return out


def hull_sky(result):
    """(ra, dec) of the border of the survey hull, closed, each edge cut into HULL_EDGE_POINTS."""
    v = result["hull"]["vertices"]
    t = np.linspace(0.0, 1.0, HULL_EDGE_POINTS, endpoint=False)[:, None]
    nxt = np.roll(v, -1, axis=0)
    pts = np.vstack([v[i] + t * (nxt[i] - v[i]) for i in range(len(v))] + [v[:1]])
    return _to_sky(pts, result["grid"])


def sky_grid(result, step_deg=None):
    """The map resampled on a regular (RA, Dec) grid for plotting.

    Returns (ra (nx,), dec (ny,), z (ny, nx)) with z = the map over its mean (0 for an
    empty cell, NaN outside the survey hull; the position is tested against the hull,
    not against the cells). The grid step defaults to the cell of the map in degrees.
    """
    g, hull = result["grid"], result["hull"]
    step = step_deg or float(np.degrees(g["pixel"] / g["d_a"]))
    vra, vdec = _to_sky(hull["vertices"], g)
    ra = np.arange(vra.min(), vra.max() + step, step)
    dec = np.arange(vdec.min(), vdec.max() + step, step)
    gra, gdec = np.meshgrid(ra, dec)
    xi, eta = tangent_plane(gra.ravel(), gdec.ravel(), g["ra0"], g["dec0"])
    gx, gy = xi * g["d_a"], eta * g["d_a"]
    inside = in_hull(hull, gx, gy)
    val = ndimage.map_coordinates(result["filled"], [
        (gy - g["y0"]) / g["pixel"] - 0.5, (gx - g["x0"]) / g["pixel"] - 0.5], order=1,
        mode="nearest")
    return ra, dec, np.where(inside, val / result["mean"], np.nan).reshape(gra.shape)


def _gauss(r, a, s, b):
    return b + a * np.exp(-0.5 * (r / s) ** 2)


def _fit_gauss(x, y, s_lo, s_hi, s0, base):
    """(A, s, B, status) of the fit y = B + A exp(-x^2 / (2 s^2)), A, B >= 0, s_lo <= s <= s_hi.

    x, y: the profile (non-finite entries are ignored); s0, base: the starting width and
    background. status: 'ok', 'bound' (s or A at a bound) or 'fail' (too few points, no
    convergence, non-finite result; then s is NaN).
    """
    good = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x)[good], np.asarray(y)[good]
    scale = float(np.abs(y).max()) if len(y) else 0.0
    if len(x) < 4 or not scale > 0:
        return 0.0, np.nan, 0.0, "fail"
    b0 = min(max(base / scale, 0.0), 1.0)
    p0 = (max(float(y.max()) / scale - b0, 1e-3), min(max(s0, s_lo), s_hi), b0)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            p = curve_fit(_gauss, x, y / scale, p0=p0, maxfev=2000,
                          bounds=([0.0, s_lo, 0.0], [np.inf, s_hi, np.inf]))[0]
    except (RuntimeError, ValueError):
        return 0.0, np.nan, 0.0, "fail"
    if not np.isfinite(p).all():
        return 0.0, np.nan, 0.0, "fail"
    at_bound = p[1] <= s_lo * 1.001 or p[1] >= s_hi * 0.999 or p[0] <= 1e-6
    return float(p[0] * scale), float(p[1]), float(p[2] * scale), "bound" if at_bound else "ok"


def _half_max_sigma(r, rho, fallback):
    """Gaussian sigma of the half-maximum radius of a profile above its last value."""
    p = rho - rho[-1]
    top = float(p.max())
    if not top > 0:
        return fallback
    below = np.flatnonzero(p <= 0.5 * top)
    if len(below) == 0:
        half = float(r[-1])
    elif below[0] == 0:
        half = float(r[0])
    else:
        k = below[0]
        half = float(np.interp(0.5 * top, [p[k], p[k - 1]], [r[k], r[k - 1]]))
    return half / np.sqrt(2.0 * np.log(2.0))


def _window(shape, grid, cx, cy, radius):
    """(rows, cols, r): the cells within one cell of radius of (cx, cy) (Mpc) and their distance."""
    pixel = grid["pixel"]
    reach = int(np.ceil(radius / pixel)) + 1
    row0 = int(round((cy - grid["y0"]) / pixel - 0.5))
    col0 = int(round((cx - grid["x0"]) / pixel - 0.5))
    rows = np.arange(max(row0 - reach, 0), min(row0 + reach + 1, shape[0]))
    cols = np.arange(max(col0 - reach, 0), min(col0 + reach + 1, shape[1]))
    r = np.hypot(grid["x0"] + (cols[None, :] + 0.5) * pixel - cx,
                 grid["y0"] + (rows[:, None] + 0.5) * pixel - cy)
    return rows, cols, r


def _refine_peak(dens, footprint, filled, grid, cx, cy, h):
    """(x, y) of the highest footprint cell within PEAK_SEARCH h of (cx, cy), moved to the
    vertex of a separable 3-point parabola (offset clipped to half a cell); None if no
    footprint cell is within reach."""
    rows, cols, r = _window(dens.shape, grid, cx, cy, PEAK_SEARCH * h)
    block = np.where(footprint[np.ix_(rows, cols)] & (r <= PEAK_SEARCH * h),
                     dens[np.ix_(rows, cols)], -np.inf)
    if not np.isfinite(block).any():
        return None
    i, j = np.unravel_index(np.argmax(block), block.shape)
    row, col = int(rows[i]), int(cols[j])

    def vertex(m, c, p):
        den = m - 2.0 * c + p
        return float(np.clip(0.5 * (m - p) / den, -0.5, 0.5)) if den < 0 else 0.0

    c = filled[row, col]
    ox = vertex(filled[row, col - 1], c, filled[row, col + 1])
    oy = vertex(filled[row - 1, col], c, filled[row + 1, col])
    return (float(grid["x0"] + (col + 0.5 + ox) * grid["pixel"]),
            float(grid["y0"] + (row + 0.5 + oy) * grid["pixel"]))


def _radial_fit(dens, footprint, grid, cx, cy, h):
    """(A, s, B, status): Gaussian fit of the mean map in annuli of one cell around (cx, cy).

    The annuli reach PROFILE_RADIUS h and hold footprint cells only. A failed fit falls back
    to the half-maximum radius of the profile converted to a sigma (A, B from the profile).
    """
    pixel = grid["pixel"]
    rows, cols, r = _window(dens.shape, grid, cx, cy, PROFILE_RADIUS * h)
    keep = footprint[np.ix_(rows, cols)] & (r <= PROFILE_RADIUS * h)
    bins = np.floor(r[keep] / pixel).astype(int)
    count = np.bincount(bins)
    used = count > 0
    rho = np.bincount(bins, weights=dens[np.ix_(rows, cols)][keep])[used] / count[used]
    radius = np.bincount(bins, weights=r[keep])[used] / count[used]
    a, s, b, status = _fit_gauss(radius, rho, pixel / 2.0, PROFILE_RADIUS * h, h, float(rho[-1]))
    if status == "fail":
        s = max(_half_max_sigma(radius, rho, h), pixel / 2.0)
        a, b = float(rho.max() - rho[-1]), float(rho[-1])
    return a, s, b, status


def _unit_normals(points):
    """Unit normals (-ty, tx) of a polyline, tangent from the gradient of its points."""
    t = np.gradient(points, axis=0)
    norm = np.hypot(t[:, 0], t[:, 1])
    t = t / np.where(norm > 0, norm, 1.0)[:, None]
    return np.column_stack([-t[:, 1], t[:, 0]])


def _across(filled, hull, grid, points, normals, offsets):
    """The map at points + offset x normal: (len(points), len(offsets)), NaN outside the hull."""
    pos = points[:, None, :] + offsets[None, :, None] * normals[:, None, :]
    px, py = pos[..., 0].ravel(), pos[..., 1].ravel()
    val = _bilinear(filled, grid, px, py)
    return np.where(in_hull(hull, px, py), val, np.nan).reshape(len(points), len(offsets))


def _end_pins(points, refined, h):
    """[(index, (x, y))]: the ends of a path within ENDPOINT_SNAP h of a refined peak, held at it.

    When both ends are nearest to the same peak only the closer one is held (the path would
    collapse onto the peak).
    """
    if not refined:
        return []
    spots = np.array(refined)
    near = []
    for end in (0, -1):
        gap = np.hypot(*(spots - points[end]).T)
        near.append((end, int(gap.argmin()), float(gap.min())))
    near = [n for n in near if n[2] <= ENDPOINT_SNAP * h]
    if len(near) == 2 and near[0][1] == near[1][1]:
        near = [min(near, key=lambda n: n[2])]
    return [(end, tuple(refined[i])) for end, i, _ in near]


def _clip_to_hull(hull, points):
    """The points moved onto the hull where they are outside (projection on its half-planes)."""
    eq = hull["equations"]
    points = points.copy()
    for _ in range(10):
        over = np.maximum(points @ eq[:, :2].T + eq[:, 2], 0.0)
        if not over.any():
            break
        worst = over.argmax(axis=1)
        points -= over[np.arange(len(points)), worst][:, None] * eq[worst, :2]
    return points


def _sky_inside(points, grid, hull):
    """(ra, dec) of points (Mpc) with NaN where they are outside the survey hull."""
    ra, dec = _to_sky(points, grid)
    out = ~in_hull(hull, points[:, 0], points[:, 1])
    return np.where(out, np.nan, ra), np.where(out, np.nan, dec)


def _local_crest(val, offsets, near=None, window=None):
    """(d, no_crest): the offset of the local maximum of every profile (row of val) nearest to near.

    A local maximum is an interior sample not lower than its left neighbour and higher than
    its right one, refined with a 3-point parabola; near defaults to 0. With window, only
    the maxima within window of near count and a row without one keeps near. Without it, a
    row with no local maximum takes its highest sample nearest to near and is flagged in
    no_crest (a row with no valid sample keeps near and is flagged).
    """
    step = offsets[1] - offsets[0]
    near = np.zeros(len(val)) if near is None else np.asarray(near, dtype=float)
    d, no_crest = near.copy(), np.zeros(len(val), dtype=bool)
    for k, row in enumerate(val):
        top = (row[1:-1] >= row[:-2]) & (row[1:-1] > row[2:]) & np.isfinite(row[1:-1])
        found = np.flatnonzero(top) + 1
        if window is not None:
            found = found[np.abs(offsets[found] - near[k]) <= window]
        if len(found):
            i = found[np.argmin(np.abs(offsets[found] - near[k]))]
            den = row[i - 1] - 2.0 * row[i] + row[i + 1]
            ok = den < 0 and np.isfinite(row[i - 1]) and np.isfinite(row[i + 1])
            d[k] = offsets[i] + (np.clip(0.5 * (row[i - 1] - row[i + 1]) / den, -1.0, 1.0) * step
                                 if ok else 0.0)
        elif window is None:
            no_crest[k] = True
            if np.isfinite(row.max()):
                best = np.flatnonzero(row == row.max())
                d[k] = offsets[best[np.argmin(np.abs(offsets[best] - near[k]))]]
    return d, no_crest


def _snap_path(filled, hull, grid, points, h, pins):
    """(points, n_iter, snap): the polyline moved onto the local crest of the map, point by point.

    points: the drawn polyline inside the hull, pins: [(index, (x, y))], ends held at a
    peak. The path is smoothed and resampled at one cell (q_k, fixed unit normals n_k); the
    offset d_k of every point is the local maximum of the map along n_k within SNAP_MAX h
    nearest to the drawn line (the highest sample when there is none: no crest in reach),
    regularised along the path (running median over WINDOW h of arc length, mean of 5),
    searched again within one cell and smoothed; the fitted points are q_k + d_k n_k, held
    at the pins and inside the hull (n_iter = 2 passes). snap: shift_max_mpc (largest
    |q_k + d_k n_k - q_k| except the pinned ends), clamped_fraction (share of the points
    without a crest in reach), end_pin_mpc (largest distance of a pin from the drawn end).
    """
    pixel = grid["pixel"]
    reach = SNAP_MAX * h
    offsets = np.linspace(-reach, reach, 2 * int(np.ceil(reach / (pixel / 4.0))) + 1)
    q = _densify(_smooth(points), pixel)
    normals = _unit_normals(q)
    arc = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(q, axis=0).T))])
    val = _across(filled, hull, grid, q, normals, offsets)
    val = np.where(np.isfinite(val), val, -np.inf)
    d, no_crest = _local_crest(val, offsets)
    near = np.abs(arc[:, None] - arc[None, :]) <= WINDOW * h
    d = _smooth(np.array([np.median(d[row]) for row in near]))
    d = _smooth(_local_crest(val, offsets, d, pixel)[0])
    fitted = q + np.clip(d, -reach, reach)[:, None] * normals
    held = np.zeros(len(q), dtype=bool)
    for end, pos in pins:
        fitted[end] = pos
        held[end] = True
    fitted = _clip_to_hull(hull, fitted)
    move = np.hypot(*(fitted - q).T)
    pinned = max([float(np.hypot(*(np.array(pos) - q[end]))) for end, pos in pins], default=0.0)
    return fitted, 2, {"shift_max_mpc": float(move[~held].max(initial=0.0)),
                       "clamped_fraction": float(no_crest.mean()), "end_pin_mpc": pinned}


def _transverse_fit(filled, hull, grid, points, h):
    """(width, whole): Gaussian width across a path at each of its points, and the fit of all.

    The profile across the path (PROFILE_HALF h each side, half a cell apart, NaN outside the
    hull) is averaged over the points within WINDOW h of arc length and fitted; a fit that
    fails or hits a bound is filled by interpolation along the path (by the whole-path fit
    when every one fails, by h if that fails too), then the widths are smoothed. whole:
    (A, s, B, status) of the profile averaged over all the points.
    """
    pixel = grid["pixel"]
    half = PROFILE_HALF * h
    offsets = np.linspace(-half, half, 2 * int(np.ceil(half / (pixel / 2.0))) + 1)
    val = _across(filled, hull, grid, points, _unit_normals(points), offsets)
    arc = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(points, axis=0).T))])

    def fit(profile):
        finite = np.flatnonzero(np.isfinite(profile))
        base = float(profile[finite[[0, -1]]].mean()) if len(finite) else 0.0
        return _fit_gauss(offsets, profile, pixel / 2.0, half, h, base)

    width = np.full(len(points), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        whole = fit(np.nanmean(val, axis=0))
        for k in range(len(points)):
            a, s, b, status = fit(np.nanmean(val[np.abs(arc - arc[k]) <= WINDOW * h], axis=0))
            if status == "ok":
                width[k] = s
    bad = ~np.isfinite(width)
    if bad.all():
        width[:] = whole[1] if np.isfinite(whole[1]) else h
    elif bad.any():
        width[bad] = np.interp(arc[bad], arc[~bad], width[~bad])
    return _smooth(width), whole


def _check_spec(spec):
    """(peaks, paths): [(ra, dec)] and [(label, ra, dec arrays)] of a model spec.

    Raises ConcentrationError when the spec is not {"peaks": [[ra, dec], ...], "paths":
    [{"label", "ra": [...], "dec": [...]}, ...]} with finite peak positions.
    """
    try:
        peaks = [(float(p[0]), float(p[1])) for p in (spec.get("peaks") or [])]
        paths = []
        for k, p in enumerate(spec.get("paths") or [], start=1):
            ra, dec = np.asarray(p["ra"], dtype=float), np.asarray(p["dec"], dtype=float)
            if ra.ndim != 1 or ra.shape != dec.shape:
                raise ValueError("ra and dec of a path must be lists of the same length")
            paths.append((str(p.get("label") or f"path {k}"), ra, dec))
        if not np.isfinite(peaks).all():
            raise ValueError("a peak position is not finite")
    except (AttributeError, TypeError, ValueError, KeyError, IndexError) as err:
        raise ConcentrationError(f"malformed model spec ({err}): expected peaks [[ra, dec], ...] "
                                 "and paths [{'label', 'ra': [...], 'dec': [...]}, ...]") from None
    return peaks, paths


def compute_model(df, result, spec):
    """(columns, model): the model of the selected peaks and paths. See the module docstring.

    df, result: the table and the result of compute_concentration (every row of df gets the
    columns); spec: {"peaks": [[ra, dec], ...], "paths": [{"label", "ra": [...], "dec":
    [...]}, ...]}, degrees. Raises ConcentrationError for an empty or malformed spec.

    columns (len(df); NaN / False / '' for a row without a position or without a
    structure): model_dist_mpc (distance to the nearest structure), model_dist_sigma (the
    smallest distance / width over the structures), model_inside (model_dist_sigma <= 1),
    model_structure (label of the structure with that smallest value).
    model: peaks (label, ra, dec, x, y, height, n_sigma, A, B, width_mpc,
    width_intrinsic_mpc, fit_ok, shift_mpc, circle_ra, circle_dec), paths (label, ra, dec,
    x_mpc, y_mpc, input_ra, input_dec, left_ra, left_dec, right_ra, right_dec, width_mpc
    (list), width_median_mpc, width_intrinsic_median_mpc, fit {A, s, B, ok},
    crest_n_sigma_mean, crest_n_sigma_min, length_mpc, shift_max_mpc, clamped_fraction,
    end_pin_mpc, n_iter, open; the band edges and circles are NaN outside the survey hull),
    skipped ({label, reason}), info (JSON-serialisable: n_peaks, n_paths, n_skipped,
    bandwidth_mpc, counts, constants) and inside (the column model_inside).
    """
    peaks_in, paths_in = _check_spec(spec)
    if not peaks_in and not paths_in:
        raise ConcentrationError("select at least one peak or path")
    g, hull = result["grid"], result["hull"]
    dens, footprint, filled = result["dens"], result["footprint"], result["filled"]
    h, pixel = float(result["info"]["bandwidth"]["mpc"]), float(g["pixel"])
    mean, sigma = result["mean"], result["sigma"]
    angle = np.linspace(0.0, 2.0 * np.pi, CIRCLE_POINTS + 1)
    skipped, peaks_out, paths_out, structures, refined = [], [], [], [], []

    def project(ra, dec):
        xi, eta = tangent_plane(ra, dec, g["ra0"], g["dec0"])
        return np.column_stack([np.atleast_1d(xi) * g["d_a"], np.atleast_1d(eta) * g["d_a"]])

    for k, (ra, dec) in enumerate(peaks_in, start=1):
        label = f"peak {k}"
        gx, gy = project(ra, dec)[0]
        found = _refine_peak(dens, footprint, filled, g, gx, gy, h)
        if found is None:
            skipped.append({"label": label, "reason": "no map cell within the search radius"})
            continue
        x, y = found
        a, s, b, status = _radial_fit(dens, footprint, g, x, y, h)
        height = float(_bilinear(filled, g, np.array([x]), np.array([y]))[0])
        sky = _to_sky(np.array([[x, y]]), g)
        circle = _sky_inside(np.column_stack([x + s * np.cos(angle), y + s * np.sin(angle)]),
                             g, hull)
        peaks_out.append({
            "label": label, "ra": float(sky[0][0]), "dec": float(sky[1][0]), "x": x, "y": y,
            "height": height, "n_sigma": (height - mean) / sigma, "A": a, "B": b,
            "width_mpc": s, "width_intrinsic_mpc": float(np.sqrt(max(s ** 2 - h ** 2, 0.0))),
            "fit_ok": status != "fail", "shift_mpc": float(np.hypot(x - gx, y - gy)),
            "circle_ra": circle[0].tolist(), "circle_dec": circle[1].tolist()})
        structures.append({"label": label, "points": np.array([[x, y]]), "width": np.array([s])})
        refined.append([x, y])

    for label, ra, dec in paths_in:
        given = project(ra, dec)
        keep = np.isfinite(given).all(axis=1) & in_hull(hull, given[:, 0], given[:, 1])
        start = _densify(given[keep], pixel) if keep.sum() >= 2 else given[:0]
        if len(start) < 2:
            skipped.append({"label": label,
                            "reason": "fewer than 2 distinct points inside the survey area"})
            continue
        pts, n_iter, snap = _snap_path(filled, hull, g, start, h, _end_pins(start, refined, h))
        width, whole = _transverse_fit(filled, hull, g, pts, h)
        normals = _unit_normals(pts)
        crest = (_bilinear(filled, g, pts[:, 0], pts[:, 1]) - mean) / sigma
        arc = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))])
        fine = _densify(pts, pixel / 4.0)
        fine_arc = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(fine, axis=0).T))])
        median = float(np.median(width))
        centre = _to_sky(pts, g)
        left, right = (_sky_inside(pts + sign * width[:, None] * normals, g, hull)
                       for sign in (1.0, -1.0))
        gap = -(pts[[0, -1]] @ hull["equations"][:, :2].T + hull["equations"][:, 2]).max(axis=1)
        paths_out.append({
            "label": label, "ra": centre[0].tolist(), "dec": centre[1].tolist(),
            "x_mpc": pts[:, 0].tolist(), "y_mpc": pts[:, 1].tolist(),
            "input_ra": ra.tolist(), "input_dec": dec.tolist(),
            "left_ra": left[0].tolist(), "left_dec": left[1].tolist(),
            "right_ra": right[0].tolist(), "right_dec": right[1].tolist(),
            "width_mpc": width.tolist(), "width_median_mpc": median,
            "width_intrinsic_median_mpc": float(np.sqrt(max(median ** 2 - h ** 2, 0.0))),
            "fit": {"A": whole[0], "s": whole[1], "B": whole[2], "ok": whole[3] == "ok"},
            "crest_n_sigma_mean": float(np.nanmean(crest)),
            "crest_n_sigma_min": float(np.nanmin(crest)), "length_mpc": _length(pts),
            "shift_max_mpc": snap["shift_max_mpc"], "clamped_fraction": snap["clamped_fraction"],
            "end_pin_mpc": snap["end_pin_mpc"],
            "n_iter": int(n_iter), "open": bool(gap.min() <= pixel)})
        structures.append({"label": label, "points": fine,
                           "width": np.interp(fine_arc, arc, width)})

    n = len(df)
    ra_df, dec_df = df["ra"].to_numpy(float), df["dec"].to_numpy(float)
    finite = np.isfinite(ra_df) & np.isfinite(dec_df)
    best, near, who = np.full(n, np.nan), np.full(n, np.nan), np.full(n, "", dtype=object)
    if structures and finite.any():
        xy = project(ra_df[finite], dec_df[finite])
        ratio_min, dist_min = np.full(len(xy), np.inf), np.full(len(xy), np.inf)
        which = np.zeros(len(xy), dtype=int)
        for i, st in enumerate(structures):
            dist, j = cKDTree(st["points"]).query(xy)
            ratio = dist / st["width"][j]
            better = ratio < ratio_min
            ratio_min, which = np.where(better, ratio, ratio_min), np.where(better, i, which)
            dist_min = np.minimum(dist_min, dist)
        best[finite], near[finite] = ratio_min, dist_min
        who[finite] = np.array([s["label"] for s in structures], dtype=object)[which]
    with np.errstate(invalid="ignore"):
        inside = best <= 1.0
    columns = {"model_dist_mpc": near, "model_dist_sigma": best, "model_inside": inside,
               "model_structure": who}
    spec_z = df["z_type"].to_numpy(object) == "spec"
    info = {"n_peaks": len(peaks_out), "n_paths": len(paths_out), "n_skipped": len(skipped),
            "bandwidth_mpc": h,
            "counts": {"inside_spec": int((inside & spec_z).sum()),
                       "inside_phot": int((inside & ~spec_z).sum()),
                       "outside_spec": int((~inside & spec_z).sum()),
                       "outside_phot": int((~inside & ~spec_z).sum())},
            "constants": {"PEAK_SEARCH": PEAK_SEARCH, "PROFILE_RADIUS": PROFILE_RADIUS,
                          "SNAP_MAX": SNAP_MAX, "ENDPOINT_SNAP": ENDPOINT_SNAP,
                          "PROFILE_HALF": PROFILE_HALF, "WINDOW": WINDOW}}
    return columns, {"peaks": peaks_out, "paths": paths_out, "skipped": skipped, "info": info,
                     "inside": inside}


def default_spec(result):
    """Model spec of a result: every significant peak and every ridge segment ('ridge <id>')."""
    g = result["grid"]
    ra, dec = _to_sky(np.array([[p["x"], p["y"]] for p in result["peaks"]]), g)
    paths = []
    for r in result["ridges"]:
        r_ra, r_dec = _to_sky(r["points"], g)
        paths.append({"label": f"ridge {r['id']}", "ra": r_ra.tolist(), "dec": r_dec.tolist()})
    return {"peaks": [[float(a), float(d)] for a, d in zip(np.atleast_1d(ra), np.atleast_1d(dec))],
            "paths": paths}


def describe_model(model, mask=None, df=None):
    """Lines of text summarising a model; with mask and df the counts are those of the mask."""
    info, c = model["info"], model["info"]["counts"]
    skipped = "".join(f" {s['label']} skipped ({s['reason']})." for s in model["skipped"])
    lines = [f"Model: {info['n_peaks']} peaks and {info['n_paths']} paths, bandwidth "
             f"{info['bandwidth_mpc']:.2f} Mpc.{skipped}"]
    for p in model["peaks"]:
        lines.append(f"{p['label']}: {p['n_sigma']:.1f} sigma, 1 sigma radius {p['width_mpc']:.2f} "
                     f"Mpc (intrinsic {p['width_intrinsic_mpc']:.2f})"
                     + ("" if p["fit_ok"] else ", profile fit failed"))
    for p in model["paths"]:
        lines.append(f"{p['label']}: {p['length_mpc']:.1f} Mpc, crest "
                     f"{p['crest_n_sigma_mean']:.1f} sigma (lowest {p['crest_n_sigma_min']:.1f}), "
                     f"1 sigma half-width "
                     f"{p['width_median_mpc']:.2f} Mpc (intrinsic "
                     f"{p['width_intrinsic_median_mpc']:.2f}), moved up to "
                     f"{p['shift_max_mpc']:.2f} Mpc"
                     + (f", no crest within {SNAP_MAX:g} h of the drawn path along "
                        f"{100 * p['clamped_fraction']:.0f} % of it" if p["clamped_fraction"] > 0
                        else "")
                     + (", open to the border" if p["open"] else ""))
    if mask is not None and df is not None:
        use = np.asarray(mask, dtype=bool)
        spec_z, inside = df["z_type"].to_numpy(object) == "spec", model["inside"]
        c = {"inside_spec": int((inside & spec_z & use).sum()),
             "inside_phot": int((inside & ~spec_z & use).sum()),
             "outside_spec": int((~inside & spec_z & use).sum()),
             "outside_phot": int((~inside & ~spec_z & use).sum())}
    lines.append(f"Inside the 1 sigma band of the model: {c['inside_spec'] + c['inside_phot']} "
                 f"({c['inside_spec']} spec-z + {c['inside_phot']} photo-z); outside: "
                 f"{c['outside_spec'] + c['outside_phot']} ({c['outside_spec']} + "
                 f"{c['outside_phot']}).")
    return lines


def describe(info):
    """Lines of text summarising the info of compute_concentration."""
    s, b, c = info["settings"], info["bandwidth"], info["counts"]
    weight = s["weight"] + (f" (capped at the {s['weight_clip']:g}th percentile)"
                            if s["weight_clip"] else "")
    levels = info["contour_levels_n_sigma"]
    lines = [
        f"Map of {info['n_sample']} galaxies ({s['sample']}), weight {weight}, bandwidth "
        f"{b['mpc']:.2f} Mpc ({b['source']}), cell {info['pixel_mpc']:.2f} Mpc, survey area "
        f"{info['hull_area_mpc2']:.0f} Mpc^2 (hull of {info['n_rows']} galaxies).",
        f"Mean {info['mean']:.3g}, sigma {info['sigma']:.3g}, maximum {info['top_n_sigma']:.1f} "
        f"sigma; {len(levels)} contour levels from 1 sigma"
        + (f" to {levels[-1]:.1f} sigma." if levels else ".")]
    lines.append(f"Peaks: {info['n_peaks']} significant (prominence >= "
                 f"{s['min_prominence_sigma']:g} sigma), "
                 f"{sum(p['open'] for p in info['peaks'])} open; ridges: {info['n_ridges']} "
                 f"segments, {info['n_junctions']} branch points, {info['n_ridges_open']} open "
                 f"to the border, {info['ridges_total_length_mpc']:.1f} Mpc in total.")
    lines.append(f"Inside the 1 sigma contour: {c['inside_spec'] + c['inside_phot']} "
                 f"({c['inside_spec']} spec-z + {c['inside_phot']} photo-z); outside: "
                 f"{c['outside_spec'] + c['outside_phot']} ({c['outside_spec']} + "
                 f"{c['outside_phot']}).")
    return lines
