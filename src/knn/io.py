"""Catalogue input."""

from pathlib import Path

import pandas as pd


def load_catalogue(path, columns):
    """Read a parquet or csv catalogue and rename columns to canonical names.

    columns maps canonical name -> name in the file, e.g. {'z_spec': 'z'}.
    All columns not in the map are kept unchanged.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"catalogue not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(path)
    elif suffix == ".csv":
        df = pd.read_csv(path, low_memory=False)
    else:
        raise ValueError(f"unsupported catalogue format '{suffix}' (use .parquet or .csv)")

    rename = {}
    for canonical, source in columns.items():
        if source not in df.columns:
            raise KeyError(
                f"column '{source}' (mapped to '{canonical}') not found in {path}")
        if source != canonical:
            if canonical in df.columns:
                raise ValueError(
                    f"cannot rename '{source}' to '{canonical}': "
                    f"the catalogue already has a column '{canonical}'")
            rename[source] = canonical
    return df.rename(columns=rename)
