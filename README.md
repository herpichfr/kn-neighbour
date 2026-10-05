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
