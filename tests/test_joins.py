import numpy as np
import pandas as pd
import pytest

from knn.joins import apply_join, apply_joins, match_positions

ARCSEC = 1.0 / 3600.0


def _table():
    # table rows: 0 near cat 0; 1 and 2 both near cat 1 (double match);
    # 3 near cat 2 and cat 3 (one table row claimed by two catalogue rows); 4 far from all
    ra = np.array([10.0, 20.0, 20.0 + 1.0 * ARCSEC, 30.0, 50.0])
    dec = np.zeros(5)
    return ra, dec


def test_match_positions_matches_and_ambiguities():
    t_ra, t_dec = _table()
    cat_ra = np.array([10.0 + 0.5 * ARCSEC, 20.0, 30.0 + 0.3 * ARCSEC, 30.0 - 0.4 * ARCSEC,
                       40.0, np.nan])
    cat_dec = np.zeros(6)
    idx, sep, matched, amb = match_positions(cat_ra, cat_dec, t_ra, t_dec, 1.5)
    assert list(matched) == [True, True, True, True, False, False]
    assert idx[0] == 0 and idx[1] == 1 and idx[2] == 3 and idx[3] == 3
    np.testing.assert_allclose(sep[:4], [0.5, 0.0, 0.3, 0.4], atol=1e-6)
    assert np.isnan(sep[5])
    # cat 0 is clean; cat 1 has two table rows within 1.5 arcsec; cats 2 and 3 share a table row
    assert list(amb) == [False, True, True, True, False, False]
    # a smaller radius removes the second table row of cat 1 from consideration
    _, _, matched, amb = match_positions(cat_ra, cat_dec, t_ra, t_dec, 0.9)
    assert list(matched[:4]) == [True, True, True, True]
    assert list(amb[:2]) == [False, False]
    # an empty table matches nothing
    _, _, matched, _ = match_positions(cat_ra, cat_dec, [], [], 1.5)
    assert not matched.any()


def test_apply_join(tmp_path):
    t_ra, t_dec = _table()
    table = pd.DataFrame({"RA": t_ra, "Dec": t_dec, "label": ["A\\_1", "A\\_2", "A\\_3", "B\\_1", "C"],
                          "gid": [-1.0, 3.0, 4.0, 2.0, 1.0], "val": [1.5, 2.5, 3.5, 4.5, 5.5]})
    table.to_csv(tmp_path / "t.csv", index=False)
    cat = pd.DataFrame({"ra": [10.0 + 0.5 * ARCSEC, 20.0, 40.0], "dec": 0.0, "x": [1, 2, 3]})
    entry = {"name": "toy", "path": "t.csv", "ra": "RA", "dec": "Dec", "prefix": "tt",
             "columns": ["val", "gid"], "int_columns": ["gid"]}
    out, rep = apply_join(cat, entry, root=tmp_path)
    assert list(out.columns) == ["ra", "dec", "x", "tt_val", "tt_gid",
                                 "join_toy_sep_arcsec", "join_toy_ambiguous"]
    assert out["tt_val"][0] == 1.5 and out["tt_val"][1] == 2.5 and np.isnan(out["tt_val"][2])
    assert str(out["tt_gid"].dtype) == "Int64"
    assert out["tt_gid"][0] == -1 and out["tt_gid"][1] == 3 and out["tt_gid"][2] is pd.NA
    assert list(out["join_toy_ambiguous"]) == [False, True, False]
    assert np.isnan(out["join_toy_sep_arcsec"][2])
    assert rep["n_table"] == 5 and rep["n_rows"] == 3 and rep["n_rows_matched"] == 2
    assert rep["n_ambiguous"] == 1
    assert rep["n_table_rows_without_catalogue_match"] == 2     # rows 3 and 4 have no catalogue row
    assert rep["separation_arcsec"]["max"] <= 1.5
    assert cat.columns.tolist() == ["ra", "dec", "x"]           # the input is not modified

    # explicit output names and LaTeX underscores
    entry2 = {"name": "lab", "path": "t.csv", "ra": "RA", "dec": "Dec", "unescape_latex": True,
              "columns": {"label": "substructure"}}
    out2, _ = apply_join(cat, entry2, root=tmp_path)
    assert out2["substructure"][0] == "A_1" and out2["substructure"][2] is np.nan or \
        pd.isna(out2["substructure"][2])
    # more than one entry; an existing column is never overwritten
    out3, reports = apply_joins(cat, [entry, entry2], root=tmp_path)
    assert len(reports) == 2 and "substructure" in out3.columns
    with pytest.raises(ValueError, match="already exists"):
        apply_join(out, entry, root=tmp_path)
    with pytest.raises(KeyError, match="nope"):
        apply_join(cat, {**entry, "columns": ["nope"]}, root=tmp_path)
    with pytest.raises(FileNotFoundError):
        apply_join(cat, {**entry, "path": "missing.csv"}, root=tmp_path)
