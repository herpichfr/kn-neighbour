"""Build the enriched catalogue."""

import datetime
import json
import subprocess
from pathlib import Path

import numpy as np

from . import __version__
from .config import resolve_path
from .environment import get_estimator
from .geometry import comoving_distance, comoving_xyz, make_cosmology
from .io import load_catalogue
from .redshift import assign_redshifts


def _git_head(root):
    """sha of HEAD in the repository at root, or None."""
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                             capture_output=True, text=True, check=True)
        return res.stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def write_provenance(cfg, output):
    """Write <output>.provenance.json next to the output file."""
    meta = cfg.get("_meta", {})
    config = {k: v for k, v in cfg.items() if k != "_meta"}
    record = {
        "config": config,
        "config_sha256": meta.get("sha256"),
        "git_head": _git_head(meta.get("root", ".")),
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "knn_version": __version__,
    }
    path = Path(str(output) + ".provenance.json")
    path.write_text(json.dumps(record, indent=2, default=str) + "\n")
    return path


def build(cfg, output=None):
    """Run the pipeline described by cfg and write the enriched parquet.

    Steps: load catalogue, assign redshifts, keep galaxies in the redshift
    window, compute comoving coordinates, compute each configured
    environment estimator. Returns the enriched DataFrame; also writes it to
    'output' (default: cfg['output']) with a provenance sidecar.
    """
    df = load_catalogue(resolve_path(cfg, cfg["catalogue"]), cfg["columns"])
    df = assign_redshifts(df, cfg["redshift_window"])
    df = df[df["in_window"]].reset_index(drop=True)

    cosmo = make_cosmology(cfg)
    xyz = comoving_xyz(df["ra"], df["dec"], df["z_best"], cosmo)
    df["cx"], df["cy"], df["cz"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    df["d_com"] = comoving_distance(df["z_best"], cosmo)

    is_spec = (df["z_type"] == "spec").to_numpy()
    if cfg["tracers"] == "spec":
        is_tracer = is_spec
    else:
        is_tracer = np.ones(len(df), dtype=bool)

    # Targets are spec-z galaxies only for now; photo-z galaxies get NaN
    # environment columns. Phase 2 replaces this with marginalisation over
    # each galaxy's redshift PDF.
    target_rows = np.flatnonzero(is_spec)
    tracer_pos = np.full(len(df), -1, dtype=np.intp)
    tracer_pos[is_tracer] = np.arange(is_tracer.sum())
    self_index = tracer_pos[target_rows]

    tracer_xyz = xyz[is_tracer]
    target_xyz = xyz[target_rows]
    for entry in cfg["environment"]:
        name, k = entry["name"], entry["k"]
        result = get_estimator(name)(tracer_xyz, target_xyz, k, self_index=self_index)
        for quantity, values in result.items():
            col = np.full(len(df), np.nan)
            col[target_rows] = values
            df[f"{quantity}_{name}_k{k}"] = col

    out_path = Path(output) if output is not None else resolve_path(cfg, cfg["output"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    write_provenance(cfg, out_path)
    return df
