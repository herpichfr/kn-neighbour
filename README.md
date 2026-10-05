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
  analysis panel until "Clear selection".
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
