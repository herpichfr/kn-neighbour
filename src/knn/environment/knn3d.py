"""k-th nearest neighbour density in 3D comoving space."""

import numpy as np
from scipy.spatial import cKDTree

from . import register


def knn3d(tracer_xyz, target_xyz, k, self_index=None):
    """Distance to the k-th nearest tracer and the implied density.

    tracer_xyz : (N, 3) comoving Mpc
    target_xyz : (M, 3) comoving Mpc
    k          : neighbour number, k >= 1
    self_index : optional (M,) integer array. self_index[i] is the index in
        tracer_xyz of target i, or -1 if target i is not a tracer. A tracer
        is excluded from its own neighbours by index (k+1 neighbours are
        queried and the self hit is dropped), not by zero distance, so
        coincident galaxies still count as neighbours of each other.

    Returns (r_k, rho_k): r_k in comoving Mpc and
    rho_k = k / (4/3 pi r_k^3) in Mpc^-3. r_k = 0 gives rho_k = inf.
    """
    tracer_xyz = np.asarray(tracer_xyz, dtype=float)
    target_xyz = np.asarray(target_xyz, dtype=float)
    if k < 1:
        raise ValueError("k must be >= 1")
    n_target = len(target_xyz)
    tree = cKDTree(tracer_xyz)

    if self_index is None:
        if len(tracer_xyz) < k:
            raise ValueError(f"need at least k={k} tracers, got {len(tracer_xyz)}")
        dist, _ = tree.query(target_xyz, k=k, workers=-1)
        r_k = np.asarray(dist, dtype=float).reshape(n_target, k)[:, -1]
    else:
        self_index = np.asarray(self_index, dtype=np.intp)
        if self_index.shape != (n_target,):
            raise ValueError("self_index must have one entry per target")
        if len(tracer_xyz) < k + 1:
            raise ValueError(
                f"need at least k+1={k + 1} tracers to exclude self, got {len(tracer_xyz)}")
        dist, idx = tree.query(target_xyz, k=k + 1, workers=-1)
        dist = np.asarray(dist, dtype=float).reshape(n_target, k + 1)
        idx = np.asarray(idx).reshape(n_target, k + 1)
        is_self = (idx == self_index[:, None]) & (self_index[:, None] >= 0)
        # Drop the self hit; if self is not among the k+1 hits (more than
        # k+1 coincident tracers), drop the last hit instead.
        drop = np.where(is_self.any(axis=1), is_self.argmax(axis=1), k)
        keep = np.ones(dist.shape, dtype=bool)
        keep[np.arange(n_target), drop] = False
        r_k = dist[keep].reshape(n_target, k)[:, -1]

    with np.errstate(divide="ignore"):
        rho_k = k / (4.0 / 3.0 * np.pi * r_k ** 3)
    return r_k, rho_k


@register("knn3d")
def knn3d_columns(tracers, targets, k, self_index=None, *, sigma=None):
    """knn3d as a registered estimator.

    tracers, targets are Points; sigma is accepted for the common estimator
    interface and not used. Returns {'r': r_k, 'rho': rho_k,
    'log1p_delta': ...} with
    log1p_delta = log10(rho_k / median(rho_k over the tracers)).

    NOTE: the median normalisation is provisional. The median density of the
    tracers is a stand-in for the mean density of the survey volume, which
    needs a selection function / completeness model (later phase).
    """
    r_k, rho_k = knn3d(tracers.xyz, targets.xyz, k, self_index=self_index)
    n_tracer = len(tracers)
    _, rho_tracers = knn3d(tracers.xyz, tracers.xyz, k, self_index=np.arange(n_tracer))
    with np.errstate(divide="ignore"):
        log1p_delta = np.log10(rho_k / np.median(rho_tracers))
    return {"r": r_k, "rho": rho_k, "log1p_delta": log1p_delta}
