import copy

import pytest
import yaml

from knn.build import build
from knn.concentration import inverse_tangent_plane
from knn.config import load_config
from knn.explorer import Explorer
from synthetic import (BUILD_CONFIG, COSMO, CONC_TRUTH, build_synthetic_config, conc_positions,
                       write_build_catalogue)

# k = 20 (a kNN scale of about 1 Mpc), peaks that rise 6 sigma above their saddle (a kNN map has
# noise bumps inside a clump) and the lowest ridge floor allowed, the 1 sigma level
CONC_SECTION = {"z_range": [0.014, 0.028], "sample": "all", "weight": "mass", "k": 20,
                "min_prominence_sigma": 6.0, "ridge_floor_sigma": 1.0}


@pytest.fixture(scope="session")
def built(tmp_path_factory):
    """(cfg, enriched DataFrame) of the synthetic catalogue, built once, files on disk."""
    root = tmp_path_factory.mktemp("synth")
    cfg = build_synthetic_config(root)
    return cfg, build(cfg)


@pytest.fixture(scope="session")
def explorer(built):
    """Explorer of the built synthetic catalogue (read back from the parquet files)."""
    ex = Explorer.from_config(built[0])
    assert ex.source == "parquet"
    return ex


@pytest.fixture(scope="module")
def conc_built(tmp_path_factory):
    """(cfg, enriched table) of a catalogue with planted clumps and a 'concentration' section."""
    root = tmp_path_factory.mktemp("conc")
    pos = conc_positions(blobs=[(-4.0, 0.0, 1.0, 450), (5.0, 2.0, 1.0, 350)],
                         bridges=[(-4.0, 0.0, 5.0, 2.0, 1.5, 0.5, 300)], n_field=1100, seed=2)
    n_spec = int(0.82 * len(pos))
    df = write_build_catalogue(root, n_spec=n_spec, n_phot=len(pos) - n_spec)
    d_a = float(COSMO.angular_diameter_distance(CONC_TRUTH["z"]).value)
    ra, dec = inverse_tangent_plane(pos[:, 0] / d_a, pos[:, 1] / d_a, CONC_TRUTH["ra"],
                                    CONC_TRUTH["dec"])
    df["RA"], df["DEC"] = ra, dec
    df.to_parquet(root / "data" / "cat.parquet", index=False)
    raw = copy.deepcopy(BUILD_CONFIG)
    raw["concentration"] = CONC_SECTION
    path = root / "config" / "t.yaml"
    path.parent.mkdir(exist_ok=True)
    path.write_text(yaml.safe_dump(raw))
    cfg = load_config(path)
    return cfg, build(cfg)


@pytest.fixture(scope="module")
def conc_explorer(conc_built):
    """Explorer of conc_built (read back from the parquet files)."""
    ex = Explorer.from_config(conc_built[0])
    assert ex.source == "parquet"
    return ex
