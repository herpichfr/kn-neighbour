"""Reproduction of the legacy src/calculate_kn_neighbour.py metric.

Kept only for the regression test. The legacy 'separation_k10' is the
sorted position-11 value (index 0 is the galaxy itself) of
angular_separation[deg] * separation_3d, where the 3D separation is the
Euclidean distance between points at distance z (read as Mpc) along each
line of sight. The product has no physical meaning; the new pipeline uses
comoving 3D distance.
"""

import numpy as np

Z_LO, Z_HI = 0.007, 0.037


def legacy_select(df):
    """Reproduce the selection of the legacy script on a raw catalogue.

    df must have the raw columns ra, dec, z and zml. Returns the selected
    rows in catalogue order with a reset index, plus a 'redshift' column
    holding the redshift the legacy script used as distance.
    """
    mask_specz = (df["z"] > Z_LO) & (df["z"] < Z_HI)
    mask_photz = (df["zml"] > Z_LO) & (df["zml"] < Z_HI)

    redshift = df["zml"].copy()
    redshift[mask_photz] = df["z"][mask_photz]
    redshift[mask_specz] = df["z"][mask_specz]

    mask = ~np.isnan(redshift)
    mask &= (redshift > Z_LO) & (redshift < Z_HI)

    out = df[mask].reset_index(drop=True)
    out["redshift"] = redshift[mask].to_numpy()
    return out


def _unit_vectors(ra, dec):
    ra = np.radians(np.asarray(ra, dtype=float))
    dec = np.radians(np.asarray(dec, dtype=float))
    return np.column_stack([np.cos(dec) * np.cos(ra),
                            np.cos(dec) * np.sin(ra),
                            np.sin(dec)])


def legacy_separation(ra, dec, z, index=11, chunk=256):
    """Vectorised version of the legacy metric.

    For each galaxy i, the value at sorted position 'index' of
    angle_deg(i, j) * |z_i u_i - z_j u_j| over all j (including i itself,
    which sits at position 0). ra, dec in degrees; z used as distance.
    The angle is arctan2(|u_i x u_j|, u_i . u_j). Rows are processed in
    blocks of 'chunk' galaxies so memory stays at O(chunk * N).
    """
    z = np.asarray(z, dtype=float)
    u = _unit_vectors(ra, dec)
    n = len(u)
    if not 0 <= index < n:
        raise ValueError(f"index {index} out of range for {n} galaxies")
    points = z[:, None] * u

    result = np.empty(n)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        ui = u[start:stop]
        cross = np.cross(ui[:, None, :], u[None, :, :])
        angle = np.degrees(np.arctan2(np.linalg.norm(cross, axis=-1), ui @ u.T))
        sep3d = np.linalg.norm(points[start:stop, None, :] - points[None, :, :], axis=-1)
        prod = angle * sep3d
        result[start:stop] = np.partition(prod, index, axis=1)[:, index]
    return result
