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
