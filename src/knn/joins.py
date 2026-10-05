"""Positional cross-match of the catalogue with auxiliary tables."""

from pathlib import Path

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord

JOIN_DEFAULTS = {"ra": "ra", "dec": "dec", "radius_arcsec": 1.5, "unescape_latex": False,
                 "int_columns": []}
SEPARATION_PERCENTILES = (16.0, 50.0, 84.0)


def join_settings(entry):
    """A 'joins' entry with defaults filled in; prefix defaults to the name."""
    out = {**JOIN_DEFAULTS, **entry}
    out.setdefault("prefix", entry["name"])
    return out


def match_positions(ra, dec, table_ra, table_dec, radius_arcsec):
    """Nearest-neighbour match of catalogue positions to a table.

    Every catalogue row is matched to the nearest table row; the match
    counts when it is within radius_arcsec. Returns (idx, sep_arcsec,
    matched, ambiguous), one entry per catalogue row:

    idx : index of the nearest table row (meaningful where matched)
    sep_arcsec : separation to it
    matched : within the radius (False for a catalogue row without finite
        coordinates)
    ambiguous : matched, and either a second table row is also within the
        radius, or the same table row is the match of more than one
        catalogue row. The nearest match is kept and flagged.
    """
    ra, dec = np.asarray(ra, float), np.asarray(dec, float)
    n = len(ra)
    idx = np.zeros(n, dtype=np.intp)
    sep = np.full(n, np.nan)
    matched = np.zeros(n, dtype=bool)
    ambiguous = np.zeros(n, dtype=bool)
    good = np.isfinite(ra) & np.isfinite(dec)
    if not good.any() or len(table_ra) == 0:
        return idx, sep, matched, ambiguous
    cat = SkyCoord(ra[good] * u.deg, dec[good] * u.deg)
    tab = SkyCoord(np.asarray(table_ra, float) * u.deg, np.asarray(table_dec, float) * u.deg)
    i1, s1, _ = cat.match_to_catalog_sky(tab)
    s1 = s1.arcsec
    m = s1 <= radius_arcsec
    amb = np.zeros(len(s1), dtype=bool)
    if len(tab) >= 2:
        _, s2, _ = cat.match_to_catalog_sky(tab, nthneighbor=2)
        amb |= m & (s2.arcsec <= radius_arcsec)
    claimed = np.bincount(i1[m], minlength=len(tab))
    amb |= m & (claimed[i1] > 1)
    idx[good], sep[good], matched[good], ambiguous[good] = i1, s1, m, amb
    return idx, sep, matched, ambiguous


def _read_table(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"join table not found: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def output_names(entry):
    """Mapping source column -> output column of a join entry.

    'columns' is either a list (output name = '<prefix>_<column>') or a
    mapping source -> output name.
    """
    cols = entry["columns"]
    if isinstance(cols, dict):
        return dict(cols)
    return {c: f"{entry['prefix']}_{c}" for c in cols}


def apply_join(df, entry, root=".", reference=None):
    """Add the columns of one join entry to df (returns a new DataFrame).

    entry: name, path, columns (see output_names), and optionally prefix, ra,
    dec (column names in the table), radius_arcsec, unescape_latex (replace
    '\\_' by '_' in text columns), int_columns (source columns converted to
    nullable integers). Besides the table columns, join_<name>_sep_arcsec
    (NaN where unmatched) and join_<name>_ambiguous (bool) are added.

    reference, optional (ra, dec) arrays of the whole catalogue, is used only
    to count the table rows that match no catalogue row at all (df may be
    restricted to a redshift window).

    Returns (df, report); report has the match counts and the separation
    percentiles of the matched rows.
    """
    e = join_settings(entry)
    path = Path(e["path"])
    if not path.is_absolute():
        path = Path(root) / path
    table = _read_table(path)
    names = output_names(e)
    for col in [e["ra"], e["dec"], *names]:
        if col not in table.columns:
            raise KeyError(f"join '{e['name']}': column '{col}' not found in {path}")
    for out_col in names.values():
        if out_col in df.columns:
            raise ValueError(f"join '{e['name']}': output column '{out_col}' already exists")

    radius = float(e["radius_arcsec"])
    idx, sep, matched, ambiguous = match_positions(
        df["ra"].to_numpy(float), df["dec"].to_numpy(float),
        table[e["ra"]].to_numpy(float), table[e["dec"]].to_numpy(float), radius)

    out = df.copy()
    for src, out_col in names.items():
        values = table[src].reset_index(drop=True)
        if e["unescape_latex"] and values.dtype == object:
            values = values.str.replace("\\_", "_", regex=False)
        picked = values.iloc[idx].reset_index(drop=True)
        if src in e["int_columns"]:
            picked = pd.to_numeric(picked, errors="coerce").astype("Float64").round().astype("Int64")
        picked = picked.where(pd.Series(matched, index=picked.index))
        picked.index = out.index
        out[out_col] = picked
    out[f"join_{e['name']}_sep_arcsec"] = np.where(matched, sep, np.nan)
    out[f"join_{e['name']}_ambiguous"] = ambiguous

    ref_ra, ref_dec = reference if reference is not None else (df["ra"], df["dec"])
    _, _, back, _ = match_positions(table[e["ra"]].to_numpy(float), table[e["dec"]].to_numpy(float),
                                    np.asarray(ref_ra, float), np.asarray(ref_dec, float), radius)
    n_unmatched_table = int((~back).sum())
    sep_matched = sep[matched]
    report = {
        "name": e["name"], "path": str(path), "radius_arcsec": radius,
        "n_table": int(len(table)),
        "n_rows": int(len(df)),
        "n_rows_matched": int(matched.sum()),
        "n_table_rows_used": int(len(np.unique(idx[matched]))),
        "n_ambiguous": int(ambiguous.sum()),
        "n_table_rows_without_catalogue_match": n_unmatched_table,
        "separation_arcsec": (
            {"p16": float(np.percentile(sep_matched, 16)),
             "median": float(np.median(sep_matched)),
             "p84": float(np.percentile(sep_matched, 84)),
             "max": float(sep_matched.max())} if sep_matched.size else None),
        "columns": list(names.values()),
    }
    return out, report


def apply_joins(df, entries, root=".", reference=None):
    """Apply the config's 'joins' entries in order. Returns (df, [reports])."""
    reports = []
    for entry in entries:
        df, report = apply_join(df, entry, root=root, reference=reference)
        reports.append(report)
    return df, reports
