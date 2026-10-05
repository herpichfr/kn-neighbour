"""Redshift assignment."""

import numpy as np
import pandas as pd


def assign_redshifts(df, window):
    """Add z_best, z_type and in_window to a catalogue.

    z_best is z_spec where it exists, otherwise z_phot. z_type is 'spec',
    'phot' or NaN. Spec-z takes precedence even when it lies outside the
    window: the galaxy is then out of the window, whatever its z_phot.
    in_window is True when z_lo < z_best < z_hi (open interval).

    Photo-z probability distributions are not handled here (Phase 2).
    """
    z_lo, z_hi = window
    out = df.copy()
    has_spec = out["z_spec"].notna()
    has_phot = ~has_spec & out["z_phot"].notna()

    z_best = out["z_spec"].where(has_spec, out["z_phot"])
    z_type = pd.Series(np.nan, index=out.index, dtype=object)
    z_type[has_spec] = "spec"
    z_type[has_phot] = "phot"

    out["z_best"] = z_best
    out["z_type"] = z_type
    out["in_window"] = ((z_best > z_lo) & (z_best < z_hi)).to_numpy()
    return out
