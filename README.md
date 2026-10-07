#calculate_kn_neighbours

Calculates the 3D separation to the 10th neighbour for each galaxy with a measured redshift.

## Install

```
pip install -e .[app,test]
```

## Build the enriched catalogue

```
python -m knn.cli build config/mkw4.yaml
```

This reads the catalogue, assigns redshifts, computes comoving coordinates and
the k-th nearest neighbour environment measure, and writes
`data/derived/MKW4_enriched.parquet` plus a provenance file
(`MKW4_enriched.parquet.provenance.json`). Paths in the config are resolved
relative to the repository root. After `pip install`, `knn build config/mkw4.yaml`
does the same. Use `--output PATH` to write somewhere else.

## Masses, indicators, joins, phase space

With a `photometry` section in the config the build also writes stellar
masses (`logmstar`, `logmstar_taylor`, `logmstar_w1`, `M_r`, `M_i`, each with
`_lo`/`_hi`; photo-z rows carry the distance uncertainty from the same
redshift draws as the environment), mass-weighted environment entries
(`weight: mass`, columns `..._k<k>_mass`), colours, a red-sequence offset and
`quenched` flag, positional cross-matches (`joins:`) and cluster phase space
(`cluster:`). Every coefficient and its source is cited in the docstrings of
`knn/photometry.py`, `knn/mass.py` and `knn/indicators.py`. `extinction.ebv:
sfd` needs the `sfdmap` package and the SFD map files (`pip install -e .[dust]`).

## Mass concentration

With a `concentration` section in the config (needs `cluster.z`, and `photometry` for the
default stellar-mass weight) the build also maps the projected concentration of the
galaxies of a redshift slice: a weighted Gaussian-kernel map (Scott's rule bandwidth as
in the legacy `gaussian_kde` plot), read like a mountain range. The survey area is the
convex hull of all the galaxies of the catalogue; an empty cell inside it has density 0,
a cell outside it is not drawn, and the map is corrected for the dilution at the border.
Contours are drawn at mean + n sigma of the map (n = 1, 2, 3, ...); a contour that
reaches the border of the area stays open, as it would continue beyond the data. Peaks
are the local maxima that reach the 1 sigma level and rise at least one sigma above the
saddle that joins them to a higher peak; ridges follow the higher ground between
significant peaks that share a 1 sigma region, and from a peak to the border when its 1
sigma region reaches it. A galaxy is inside when the map at its position is at or above
the 1 sigma level. Columns `conc_density` (map value at the galaxy), `conc_nsigma`,
`conc_inside`, `conc_region` (id of the connected 1 sigma region, -1 outside) and
`conc_ridge_dist_mpc`; a summary (bandwidth, mean, sigma, levels, peaks, ridges, counts)
goes to `build_info.concentration` of the provenance file. The method and its parameters
are documented in `knn/concentration.py` and `config/mkw4.yaml`.

## Web app

```
pip install -e .[app]
python -m knn.cli serve config/mkw4.yaml          # or: knn serve config/mkw4.yaml
```

Options: `--host 127.0.0.1`, `--port 8050`, `--debug`. Open
http://127.0.0.1:8050/. The app reads `data/derived/MKW4_enriched.parquet` and
`MKW4_enriched_tracers.parquet` (both written by `knn build`) when the
provenance sidecar's config sha256 equals the current config; otherwise it
builds in memory at start-up (a few seconds for MKW4; run `knn build` to cache).
Editing the config, including the optional `app: {min_per_bin: 10}` section,
makes the parquet stale.

Tour:

- **Sample** (sidebar): redshift range, spec-z / photo-z, minimum p(in window)
  and photo-z odds, log M* range, M_r maximum, membership, substructure and group
  multi-selects, quenched yes / no, morphology available. The counts of the
  selected spec-z and photo-z galaxies are shown live.
- **Environment**: estimator (`knn3d`, `sigma2d`), k, count or mass weight,
  `dv` (sigma2d spec-z slice, km/s), `n_sigma` (sigma2d photo-z slice) and the
  photo-z mode (gauss, point, pdf). Every change recomputes the environment
  from the tracers and the redshift draws (same seed as the build; the defaults
  reproduce the enriched columns) and is cached. `knn3d` is not defined for
  photo-z galaxies (REPORT_photoz_validation.md): they have no value and are
  grey in the maps. The selected estimator's log(1+delta) is the column `env`
  used in every view.
- **3D map** and **sky map**: colour by any numeric column (default `env`),
  optional size column, percentile clip of the colour range, RA/Dec/z or
  comoving x, y, z. Spec-z are circles, photo-z faded diamonds. Box or lasso
  select on the sky map: the galaxies are circled in the 3D map and restrict the
  analysis panel until "Clear selection". With a `concentration` section the sky map
  shows the concentration map of the current sample (Inferno, linear from 0; the
  area outside the survey hull is blank) with its contours (the 1 sigma one heavier),
  the border of the survey, the ridges (dashed where they continue to the border) and
  the peaks (open stars where the region reaches the border); legend entries hide
  them. The sample has an all / inside / outside 1 sigma region filter, and the
  Concentration section sets the weight, bandwidth, contour step and minimum peak
  prominence. The app starts on the configured redshift slice.
  The sky map is interactive when the config has a `concentration` section (controls
  above the map): the checkbox **Select peaks** is the click mode: a click on a star, a
  ridge or a drawn path switches it on / off (grey = off); **Select maximums** is the
  drawing mode: the mouse drag becomes a freehand path (stored point by point, so it can
  follow a curved crest). Plotly sends no click while it is drawing, so uncheck
  **Select maximums** to click (the drawn paths stay). **All on**, **All off** and **Clear drawn paths** act on
  everything at once; the text next to them counts what is on. **Run model** fits the
  peaks, ridge segments and drawn paths that are on (see Mass concentration) and draws
  the fitted paths, their 1 sigma bands and the peak circles in green (legend entries
  "model: fitted paths" and "model: 1σ band"); **Clear model** removes it. While a pick
  mode is on the map and the galaxies do not answer the hover, so that a click lands on a
  structure; box and lasso selection of galaxies is the mode when both boxes are
  unchecked. Ridge segments whose saddle is below 1 sigma (they cross a valley) are thin
  and dotted. The selection of peaks and ridges belongs to the map: it is reset when the
  sample, the weight or the bandwidth change, while the drawn paths stay.
- **Analysis**: y (a numeric column or a boolean indicator such as `quenched`,
  `sf_w2w3`) against x (default `env`) in quantile bins, one line per log M* bin
  (edges editable, empty = no mass binning), spec-z solid and photo-z dashed,
  bands = 16-84 percentiles (Wilson 68% interval for fractions); bins with fewer
  than 10 galaxies are left empty. Second plot: histogram of `env` for the
  selection against the full sample; the text box gives N per bin.
- **Export**: the selection as CSV, the UI state as a YAML snapshot (with the
  config sha256 and git sha; drop a snapshot on the upload box to restore it), PNG
  or SVG from the plot modebar (client side; WebGL traces are rasterised in SVG),
  and "Publication figure (PDF)" of the analysis plot, rendered with matplotlib.

- **Mass concentration**: summary of the map (bandwidth, mean, sigma, peaks, ridges, counts)
  and a table of statistics inside and outside the 1 sigma contour for spec-z, photo-z and
  all galaxies (N, log M*, total M*, map value, distance to the nearest ridge, quenched
  fraction with its Wilson interval, environment, g-r, R/R200, Sersic n, concentration,
  fraction in a group). The map recomputes from the sample filters (cached); the table also
  follows the sky selection. After **Run model** inside / outside is the 1 sigma band of the
  model (the galaxies within one fitted width of a fitted path or of a peak) instead of the
  1 sigma contour of the map: the statistics table, the summary lines of the model, the
  all / inside / outside sample filter and the CSV columns `model_inside`,
  `model_dist_sigma` and `model_structure` follow it (`conc_inside` is always written).
  A second table lists the structures of the model: width s (1 sigma distance), intrinsic
  width (the bandwidth removed in quadrature), crest height in sigma (mean / lowest for
  a path), length, shift from the input and the number of selected galaxies inside its
  band. The model is part of the YAML snapshot; it is fitted again on the map of the
  sample whenever the sample changes.

The session layer is `knn.explorer.Explorer` (no Dash imports); `knn.app.create_app(explorer)`
builds the Dash app without starting a server.

## Tests

```
python -m pytest
```

## Legacy script

`src/calculate_kn_neighbour.py` is the original implementation. It is kept for
regression: `src/knn/legacy.py` reproduces its selection and its
angle x 3D-separation metric, and `tests/test_legacy_regression.py` checks
that against `data/MKW4+v7_k10_neighbours.csv` (skipped if the data files are
missing).
