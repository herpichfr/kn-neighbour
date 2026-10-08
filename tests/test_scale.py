"""Responsiveness at 10^5 galaxies (synthetic; every step well under the limit on a laptop)."""

import time

import numpy as np
import pytest

from knn.app import callbacks as cb
from synthetic import synthetic_explorer

N = 100_000
LIMIT = 4.0          # seconds; the target is about 2 s, the margin keeps loaded machines green


@pytest.fixture(scope="module")
def big():
    return synthetic_explorer(n=N, phot_fraction=0.15, n_draws=20)


def _timed(func, *args, **kwargs):
    t0 = time.perf_counter()
    result = func(*args, **kwargs)
    return result, time.perf_counter() - t0


def test_environment_recompute_at_1e5(big):
    assert len(big.df) == N
    timings = {}
    for name, weight in (("knn3d", "count"), ("sigma2d", "count"), ("sigma2d", "mass")):
        env, timings[(name, weight)] = _timed(big.environment, name, 10, weight)
        assert np.isfinite(env[f"log1p_delta_{name}_k10" + ("_mass" if weight == "mass" else "")]
                           ).sum() > 0.8 * N * 0.85
    print("environment timings [s]:", {f"{k[0]}/{k[1]}": round(v, 2) for k, v in timings.items()})
    assert max(timings.values()) < LIMIT
    again, t = _timed(big.environment, "sigma2d", 10)
    assert t < 0.01                                          # cache hit


def test_views_at_1e5(big):
    state = cb.make_state(big, estimator="sigma2d", k=10, mass_bins="8, 9, 10, 11")
    sample, t_sample = _timed(cb.sample_info, big, state)
    env, t_env = _timed(cb.env_info, big, state)
    sky, t_sky = _timed(cb.sky_callback, big, state, env, sample)
    out, t_an = _timed(cb.analysis_callback, big, state, env, sample)
    csv, t_csv = _timed(cb.export_csv_frame, big, state, env, sample)
    pdf, t_pdf = _timed(cb.publication_pdf, big, state, env, sample)
    print("1e5 timings [s]: sample %.2f env %.2f sky %.2f analysis %.2f csv %.2f pdf %.2f"
          % (t_sample, t_env, t_sky, t_an, t_csv, t_pdf))
    assert sum(len(t.x) for t in sky.data[1:]) == N
    assert len(out[0].data) > 0 and pdf.startswith(b"%PDF") and len(csv) == N
    for t in (t_sample, t_sky, t_an, t_csv, t_pdf):
        assert t < LIMIT
    assert t_env < LIMIT
