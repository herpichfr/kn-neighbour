# Density contours, peaks and ridges (MKW4)

Code: `src/knn/concentration.py`; config section `concentration` in `config/mkw4.yaml`; columns `conc_*`
in `data/derived/MKW4_enriched.parquet` (summary in `build_info.concentration` of the provenance file);
web panel "Mass concentration" and the sky map. This replaces the Hessian spine tracing with 1 sigma
widths (`filaments.py`, `fil_*` columns, spines sidecar), which found no usable spines on MKW4 and was
dropped. Slice 0.017 < z < 0.024, 1879 galaxies (1785 spec-z, 94 photo-z). The map is projected only (see
the line-of-sight argument: LOS 1 sigma 11.1 Mpc spec-z, 28.7 Mpc photo-z).

**Update 2026-10-08:** the map is now a k-th nearest-neighbour surface density and the automatic
ridges have a 5 sigma floor (last section). The method, results and verification sections below
describe the earlier weighted Gaussian KDE map (Scott bandwidth) and its ridge network without a
floor; their numbers do not apply to the current code.

## Method (Gaussian KDE map, replaced)

- Map: weighted 2D KDE on the tangent plane in Mpc at D_A(z = 0.02) = 83.6 Mpc. Weight `count`, `mass`
  (10^logM* x p_in_window, clipped at the 95th percentile) or any column. Bandwidth by Scott's rule
  (as `scipy.stats.gaussian_kde`) unless set; pixel = h/4.
- Footprint: convex hull of every catalogue row (all redshifts), the same for any slice. Cells inside the
  hull are never masked: empty cells have density ~0 and are drawn darkest. Cells outside are NaN
  (transparent). Edge correction: smoothed weights / smoothed hull indicator, divisor floored at 0.2
  (at most 5x at hull corners).
- Contours: mean + n sigma of the map over the hull cells, n = 1, 1 + step, ... (step 1 sigma; more than
  12 levels are thinned keeping 1 sigma and the top). Lines from contourpy on the masked grid, so a
  contour that reaches the survey edge ends there (open) instead of closing along it.
- Peaks: union-find merge tree over cells in decreasing density (8-connected). A peak is significant if
  its height >= the 1 sigma level and its prominence (height minus the highest saddle to a higher peak)
  >= 1 sigma, i.e. it has its own contour. The global maximum always counts. A peak is open when its own
  region reaches the survey edge.
- Ridge network (rules from the author: every peak connects to at least one other; a peak can
  connect through a branch point to several; a ridge can run off the map): the Steiner subtree of the
  maximum spanning tree of the cells (the maximin, highest-lying paths) spanning every significant
  peak plus border terminals (the highest cell of each run of border cells at or above 1 sigma, unless
  a significant peak is within 2 cells of it). Cut at peaks, branch points and border ends into
  segments; no level threshold, so a segment can cross a valley (its saddle is reported and the app
  draws segments with a saddle below 1 sigma thin and dotted).
- Model (Run model in the app; knn.concentration.compute_model): the selected peaks are refined to the
  highest cell within 1 h and get a radial Gaussian fit B + A exp(-r^2/2s^2) out to 4 h; each selected
  path (automatic segment or hand-drawn polyline) is resampled at one cell, every point moves only
  along the fixed normal of the drawn line to the nearest local maximum of the map within
  SNAP_MAX = 1 h (offsets median-filtered over 2 h of arc), an end within 1 h of a selected peak is
  pinned to it; a transverse Gaussian is fitted at every point over a 2 h window, out to 3 h. The 1
  sigma band is the distance s of that fit (s includes the smoothing; intrinsic = sqrt(s^2 - h^2)).
  A galaxy is inside the model when it is within s of any selected structure (model_inside,
  model_dist_sigma <= 1).
- Inside/outside 1 sigma: inside if the map density at the galaxy >= mean + 1 sigma. Columns
  `conc_density`, `conc_nsigma`, `conc_inside`, `conc_region`, `conc_ridge_dist_mpc`.

## MKW4 results

The ridge columns of these two tables used the earlier rule (a ridge only when the saddle is
>= 1 sigma); the network that replaces it is in the last section.

Mass weight (map units M_sun/Mpc^2):

| h [Mpc] | levels | max | significant peaks (RA, Dec, n sigma; o = open) | ridges (n, length, open) | inside spec/photo |
|---|---|---|---|---|---|
| 1.81 (Scott, default) | 5 | 5.3 sigma | (181.13, 1.84) 5.3; (186.39, 9.78) 3.3 o; (175.24, 9.78) 3.2 o; (189.43, -3.69) 1.0 o | 0 | 566 / 22 |
| 1.4 | 6 | 6.6 sigma | (181.17, 1.72) 6.6; (185.97, 9.76) 4.1 o; (175.42, 9.75) 3.5 o; (189.50, -3.75) 1.2 o; (178.06, -5.68) 1.1 | 0 | 559 / 18 |
| 1.0 | 9 | 9.2 sigma | (181.10, 1.79) 9.2; (186.07, 9.73) 4.5 o; (175.80, 8.89) 3.3 o; (179.90, -1.47) 2.7; (178.03, -5.72) 2.1; (189.43, -3.81) 1.6 o; (189.60, 0.24) 1.4; (185.72, 3.15) 1.3; (170.92, 2.60) 1.2 o | 2: 5.3 Mpc cluster to (179.90, -1.47), saddle 1.13 sigma; 2.0 Mpc open, (175.80, 8.89) to the edge | 617 / 15 |

Count weight (galaxies/Mpc^2):

| h [Mpc] | levels | max | significant peaks | ridges | inside spec/photo |
|---|---|---|---|---|---|
| 1.81 (Scott) | 4 | 4.6 sigma | (181.44, 1.53) | 0 | 540 / 29 |
| 1.4 | 5 | 5.5 sigma | (181.17, 1.72) | 0 | 525 / 27 |
| 1.0 | 7 | 7.4 sigma | (181.10, 1.79) 7.4; (179.73, -1.81) 4.1; (187.75, 0.59) 1.7; (189.43, -3.81) 1.3 o | 1: 7.0 Mpc cluster to (179.73, -1.81), saddle 1.77 sigma | 553 / 29 |

- At Scott and 1.4 Mpc no two significant peaks share a 1 sigma region, so no ridges are drawn. Ridges
  appear at 1.0 Mpc; the cluster to (179.8, -1.6) ridge is found with both weights.
- The two mass-weighted peaks at Dec ~ 9.8 lie on the straight northern edge of the survey and are absent
  from the count map at Scott: a few massive galaxies amplified by the ~2x edge correction. Not verified as
  real structure.

## Verification

- Independent recomputation (ndimage labelling of superlevel sets, binary search on the level) gives the
  same prominences for the significant peaks and the same number of 1 sigma regions as the code for
  mass/1.0, count/1.0 and mass/Scott; no NaN inside the hull, none outside; `conc_inside` equals
  `conc_nsigma >= 1` for every row.
- `pytest`: 173 passed.
- Browser (port 8051, mass weight): no white holes, empty areas black, contours open at the northern edge,
  peak stars on top of the galaxies, at h = 1.0 the cyan ridge and the dashed open ridge are drawn; no
  console errors.
- Test temporaries under /ssdsto1/data/mnt were removed.

## Ridge network and model on MKW4 (this round)

Network (slice 0.017-0.024, 1879 galaxies); every map is one connected network:

| map | peaks | segments | branch points | border ends | peak degrees | segments with saddle < 1 sigma |
|---|---|---|---|---|---|---|
| mass, Scott 1.81 Mpc | 4 | 4 | 1 | 0 | 2, 1, 1, 1 | 4 |
| mass, 1.0 Mpc | 9 | 9 | 0 | 1 | 3, 1, 3, 2, 1, 1, 2, 3, 1 | 7 |
| count, Scott | 1 | 0 | 0 | 0 | 0 | - |
| count, 1.0 Mpc | 4 | 4 | 1 | 0 | 1, 1, 2, 1 | 2 |

Model with every peak and segment selected (the default of Run model), galaxies of the slice:

| map | inside the model band | inside the 1 sigma contour | median width s, peaks / paths [Mpc] (intrinsic) |
|---|---|---|---|
| mass, Scott | 723 | 588 | 1.98 / 1.73 (0.32) |
| mass, 1.0 Mpc | 951 | 632 | 1.27 / 1.09 (0.43) |
| count, Scott | 259 | 569 | 2.38 / - |
| count, 1.0 Mpc | 602 | 582 | 1.25 / 1.09 (0.44) |

- Most automatic segments cross ground below 1 sigma (rule 1 joins distant peaks), so the
  all-selected band takes in field galaxies; switching low segments off in the app is the intended use.
- The widths are dominated by the smoothing: the intrinsic widths are ~0.3-0.4 Mpc.
- A path drawn by hand from the cluster down its south-west extension (181.02, 0.74) to (179.29, -2.60)
  is pinned to the cluster peak and fitted along a crest at 2.4 sigma (lowest 1.3), 7.0 Mpc long,
  moved at most 0.48 Mpc from the drawing, half-width 2.13 Mpc.

Verification of this round: pytest 203 passed; independent check that each segment's saddle equals
the map minimum along its cells and that the network is one component; browser (port 8051): star and
ridge clicks toggle once in Select peaks mode, a drag in Select maximums mode stores a path, Run model
draws the fitted paths, bands (cut at the survey border) and peak circles, the inside filter gives the
model counts, no console errors. Two snap defects found in the browser were fixed (drift onto another
ridge: capped at 1 h; retraction of a path up the cluster slope: fixed-normal correspondence). Plotly
sends no clicks while drawing, so Select maximums is for drawing only. Test temporaries removed.

## kNN map and 5σ ridge floor (2026-10-08)

Requested by the author: drop the 3D panel, make the sky map full width and square, replace the
density map by the k-th nearest-neighbour (kNN) relation, put a 5 sigma floor under the automatic
ridges, let the user add peaks and remove the automatic ridges. Code: `concentration.py`,
`config/mkw4.yaml`, `app/{layout,callbacks,figures}.py`, `app/assets/style.css`. This section
describes the method change; the sections above (KDE) are kept as history.

Method change:

- Map: at the centre p of every grid cell inside the survey hull, r_k(p) is the projected distance
  (Mpc, tangent plane at D_A(z = 0.02)) to the k-th nearest galaxy of the slice sample (cKDTree),
  and Sigma(p) = W_k / A with W_k the sum of the weights of the k nearest galaxies (k for the count
  weight) and A the area of the disc of radius r_k inside the hull, A = pi r_k^2 f, f floored at
  COVER_FLOOR = 0.2 (at most 5 times). f is computed numerically with 128 points of a fixed
  sunflower pattern in the unit disc, scaled by r_k and tested with `in_hull` (a disc farther than
  r_k from every edge is not sampled); it agrees with the circular-segment area to 0.01 in the
  tests. k is the setting `k` (default 10, integer, 2 <= k <= sample size - 1); it replaces
  `bandwidth_mpc` (the old key is now an unknown config key).
- Scale h = the median over the galaxies of the sample of the distance to their k-th neighbour
  (the galaxy excluded), reported as the kNN scale. It sets the cell (h / 4 in this first version, h / 2 since the subsection below, unless `pixel_mpc`) and
  every length of the model (PEAK_SEARCH, SNAP_MAX, WINDOW, PROFILE_*, in units of h). The cell
  depends on the slice through h, so two slices share the hull but not the grid unless
  `pixel_mpc` is set.
- The estimator k / (pi r_k^2) is high by k / (k - 1) on a uniform field (5 % at k = 20; checked in
  the tests, together with the flat map up to the hull border) and is noisier than a kernel map: the
  relative noise is about 1 / sqrt(k), the map has a heavy upper tail, and cells far from the sample
  are low but positive (never exactly 0). Mean and sigma are over the footprint as before.
- The Gaussian-kernel intrinsic width sqrt(s^2 - h^2) is no longer defined and was removed from
  the model outputs, the structure table and `describe_model`; s is the width of the Gaussian that
  fits the kNN map (it still contains the resolution of the map). `scott_bandwidth` and
  `density_map` were removed.
- Ridge floor (`ridge_floor_sigma`, default 5, at least 1): the maximin tree is restricted to the
  cells at or above mean + floor x sigma, which makes it a forest. In each connected component with
  at least two terminals (significant peaks at or above the floor, border ends: the highest cell of
  each run of border cells at or above the floor) the network is the Steiner subtree of that tree,
  cut into segments as before. A peak alone above the floor, or below it, has no ridge (degree 0),
  and a map can have no ridge. The old rule that every peak connects to another is dropped; the
  low-crest style (saddle < 1 sigma) cannot occur and was removed. Ridges have a `component`
  number; `info` has `ridge_floor_sigma`, `ridge_floor_level`, `n_peaks_above_floor` and
  `n_ridge_components`. Drawn paths of the model are not subject to the floor.
- App: no 3D panel (state key `coords` ignored when a snapshot is restored); the sky panel is the
  full width of the main column and square (CSS `aspect-ratio: 1 / 1` on `.sky-square`); Concentration
  section: "k (nearest neighbour)" (state `conck`, id `conc-k`) replaces the bandwidth (`conch`),
  "Ridge floor [σ]" (`concfloor`, `conc-floor`). Select peaks: a click on a map cell with no
  structure adds a peak (`user_peaks` in store-struct, ids from `next_id`, magenta ★ annotation named
  `user peak <id>`; the position is the centre of the clicked heatmap cell; reset when the map
  changes; "Clear drawn" removes them with the drawn paths; All on / All off leave them); a click on
  its star removes it; they enter the model after the automatic peaks that are on and are refined to
  the highest cell within 1 h. The heatmap answers "click: add peak" in that mode, with a
  transparent click-catcher heatmap when the density background is off. Select maximums: the ridges
  and drawn paths that are on are editable layout shapes named `ridge <id>` / `path <id>`; the eraser
  of the modebar sends the remaining shapes in relayoutData and `struct_update` switches the missing
  ones off (a ridge goes to `off_ridges`, a drawn path gets `on = False`: not deleted); unnamed shapes
  are new drawn paths; edits of named shapes are not read back (`rev` moves so that the figure
  restores them).

MKW4 with the first version of the map (cell h / 4, value at the cell centre; superseded by the
subsection below) (slice 0.017 < z < 0.024, sample all, 1879 galaxies, 4942 rows set the hull, weight clip 95,
min prominence 1 sigma, ridge floor 5 sigma, contour step 1 sigma), computed with
`Explorer.concentration` on `data/MKW4+v7.parquet` (built in memory from `config/mkw4.yaml`):

| weight | k | h [Mpc] (cell) | levels | max | significant peaks (above the floor) | five highest peaks (RA, Dec, n sigma; o = open) | ridges at 5 sigma: segments, components | ridges at a 1 sigma floor (reference) | inside 1 sigma spec / photo |
|---|---|---|---|---|---|---|---|---|---|
| mass | 10 | 0.82 (0.20) | 12 | 39.4 sigma | 33 (14) | (181.15, 1.76) 39.4; (186.04, 9.35) 27.8 o; (175.71, 8.80) 20.0; (179.62, -2.01) 19.8; (187.83, 0.50) 14.5 | 2, 2 | 8, 7 | 448 / 7 |
| count | 10 | 0.82 (0.20) | 12 | 36.0 sigma | 37 (16) | (181.15, 1.76) 36.0; (187.83, 0.50) 19.0; (182.69, 0.64) 15.9; (179.62, -2.15) 14.5; (182.69, 1.06) 11.9 | 2, 1 | 11, 3 | 631 / 12 |
| mass | 5 | 0.55 (0.14) | 12 | 52.9 sigma | 72 (37) | (176.70, -3.86) 52.9; (180.90, 2.02) 47.0; (181.18, 1.83) 45.8; (179.40, -1.26) 38.8; (175.57, 8.86) 28.5 | 2, 1 | 16, 7 | 460 / 4 |
| count | 5 | 0.55 (0.14) | 12 | 55.3 sigma | 90 (44) | (181.18, -2.38) 55.3; (179.40, -1.26) 54.1; (177.35, -1.07) 53.8; (176.70, -3.86) 38.0; (182.96, 1.36) 37.4 | 2, 1 | 30, 13 | 637 / 10 |

(Every map has 12 levels, the maximum, because the maxima are 36-55 sigma and more levels are
thinned; for k = 10 the highest peak is the cluster at (181.1, 1.8).)

- With the floor at its default the network is two segments in every map, each 0.3-0.9 Mpc long
  (k = 10, mass: saddles 5.5 and 5.2 sigma, 0.6 and 0.9 Mpc; count: 7.5 and 6.5 sigma, 0.6 and
  0.5 Mpc; k = 5, mass: 12.6 and 6.7 sigma, 0.5 and 0.7 Mpc; count: 10.4 and 7.8 sigma, 0.3 and
  0.8 Mpc). They join neighbouring bumps inside the dense cores (for k = 10, mass, one of them
  ends at the cluster peak (181.15, 1.76)); none is open and none is longer than 0.9 Mpc. Most
  peaks above the floor have no ridge (degree 0). The floor was not tuned.
- The kNN map is spiky: the maximum is 36-55 sigma and, in units of the global sigma, a bump of
  1 sigma prominence is common, so the default `min_prominence_sigma` = 1 marks 33-90 peaks
  (the KDE map had 1-9). A larger prominence or k is the way to thin them; the tests use 6 sigma.
- Galaxies inside 1 sigma differ from the KDE at Scott's bandwidth (mass 566 / 22, count 540 / 29):
  mass 448 / 7 and count 631 / 12 at k = 10.
- The older tables and the 1 sigma counts above refer to the KDE map; the `conc_*` columns of
  `data/derived/MKW4_enriched.parquet` are those of the KDE until `knn build` is run again (the
  config changed, so the app builds in memory).

Verification of this round: `pytest` 217 passed (old tests updated where the behaviour changed by
design; new ones for the kNN map against an independent cKDTree computation, the disc fraction
against the circular segment, the floor, the forest, user peaks, shape events, the layout and the
snapshot with the stale `coords` key). Not verified in a browser: whether plotly keeps `name` of
layout shapes in relayoutData and activates editable path shapes by a click in `drawopenpath` mode,
whether the eraser sends the remaining shapes, whether a ridge or path vertex click wins over the
heatmap in closest hover mode, and the square layout at several widths.

### Coarser, averaged cells (pixel h / 2, 3 x 3 sub-points)

Requested to smooth the map while keeping the 5 sigma floor. The default cell is now h / 2
(`PIXELS_PER_SCALE` = 2; 0.41 Mpc at k = 10, 4 times fewer cells), and the value of a cell is the
mean of the kNN density at the 3 x 3 points of a regular sub-grid of the cell that lie inside the
hull (`SUBSAMPLES` = 3; each point has its own edge correction; a cell whose centre is in the hull
but none of whose sub-points is takes its centre value; at most `MAX_EVALUATIONS` = 12 million
density evaluations, cells x 9). Levels, peaks, ridges, the model and the `conc_*` columns use this
map; `conc_density` of a galaxy is still the map interpolated at its position. New app input "Map
pixel [Mpc]" (state `concpix`, empty = h / 2, setting `pixel_mpc`); the map summary shows the cell and
the grid. In a random field at 0.4 Mpc the spread of the difference between neighbouring cells
falls from 5.0 to 4.0 (galaxies per Mpc^2, same mean). Floor unchanged (5 sigma), minimum
prominence 1 sigma, k = 10, slice 0.017-0.024, sample all (1879 galaxies), h = 0.82 Mpc:

| weight | pixel [Mpc] | grid (nx x ny) | max | significant peaks (above floor) | ridges: segments, components | inside 1 sigma spec / photo |
|---|---|---|---|---|---|---|
| mass | h / 2 = 0.41 (default) | 80 x 72 | 20.3 sigma | 22 (7) | 0, 0 | 501 / 7 |
| mass | h = 0.82 | 43 x 39 | 14.0 sigma | 14 (5) | 0, 0 | 540 / 9 |
| count | h / 2 = 0.41 (default) | 80 x 72 | 21.1 sigma | 23 (7) | 0, 0 | 674 / 13 |
| count | h = 0.82 | 43 x 39 | 14.4 sigma | 17 (4) | 0, 0 | 660 / 18 |

(Every map has 12 levels.) With the coarser, averaged map the maximum falls from 36-39 to 20-21 sigma
and the significant peaks from 33-37 to 22-23, but no ridge survives the 5 sigma floor at either
pixel: the two sub-Mpc segments of the first version (saddles 5.2-7.5 sigma) are gone, because
averaging lowers the saddles between neighbouring bumps below the floor. The floor was not tuned.
The first-version tables above (cell h / 4, centre value) are not reproduced by the current code.
