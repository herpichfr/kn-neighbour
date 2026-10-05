import numpy as np
import pandas as pd

from knn.redshift import assign_redshifts


def test_assign_redshifts():
    df = pd.DataFrame({
        "z_spec": [0.02, 0.05, np.nan, np.nan, 0.02],
        "z_phot": [0.03, 0.02, 0.02, np.nan, np.nan],
    })
    out = assign_redshifts(df, [0.007, 0.037])
    np.testing.assert_allclose(out["z_best"], [0.02, 0.05, 0.02, np.nan, 0.02])
    assert list(out["z_type"][[0, 1, 2, 4]]) == ["spec", "spec", "phot", "spec"]
    assert pd.isna(out["z_type"][3])
    # spec-z outside the window excludes the galaxy even if z_phot is inside
    assert list(out["in_window"]) == [True, False, True, False, True]
    assert out["in_window"].dtype == bool
