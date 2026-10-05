from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from knn.legacy import legacy_select, legacy_separation

DATA = Path(__file__).resolve().parents[1] / "data"
PARQUET = DATA / "MKW4+v7.parquet"
CSV = DATA / "MKW4+v7_k10_neighbours.csv"

pytestmark = pytest.mark.skipif(
    not (PARQUET.is_file() and CSV.is_file()),
    reason="MKW4 data files are not available")


def test_legacy_reproduces_csv():
    raw = pd.read_parquet(PARQUET)
    sel = legacy_select(raw)
    ref = pd.read_csv(CSV, usecols=["ra", "dec", "separation_k10"])

    assert len(sel) == 4328
    assert len(ref) == len(sel)
    np.testing.assert_allclose(sel["ra"], ref["ra"], rtol=1e-12, atol=0)
    np.testing.assert_allclose(sel["dec"], ref["dec"], rtol=1e-12, atol=0)

    sep = legacy_separation(sel["ra"], sel["dec"], sel["redshift"], index=11)
    rel = np.abs(sep - ref["separation_k10"]) / np.abs(ref["separation_k10"])
    print(f"max relative difference: {rel.max():.3e}")
    np.testing.assert_allclose(sep, ref["separation_k10"], rtol=1e-6)


def test_chunking_does_not_change_result():
    rng = np.random.default_rng(3)
    ra = rng.uniform(180, 181, 300)
    dec = rng.uniform(0, 1, 300)
    z = rng.uniform(0.01, 0.03, 300)
    a = legacy_separation(ra, dec, z, index=11, chunk=300)
    b = legacy_separation(ra, dec, z, index=11, chunk=37)
    np.testing.assert_array_equal(a, b)
