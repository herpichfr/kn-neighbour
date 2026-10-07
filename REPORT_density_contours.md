# Density contours, peaks and ridges (MKW4)

Code: `src/knn/concentration.py`; config section `concentration` in `config/mkw4.yaml`; columns `conc_*`
in `data/derived/MKW4_enriched.parquet` (summary in `build_info.concentration` of the provenance file);
web panel "Mass concentration" and the sky map. This replaces the Hessian spine tracing with 1 sigma
widths (`filaments.py`, `fil_*` columns, spines sidecar), which found no usable spines on MKW4 and was
dropped. Slice 0.017 < z < 0.024, 1879 galaxies (1785 spec-z, 94 photo-z). The map is projected only (see
the line-of-sight argument: LOS 1 sigma 11.1 Mpc spec-z, 28.7 Mpc photo-z).

## Method

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
