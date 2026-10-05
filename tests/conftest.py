import pytest

from knn.build import build
from knn.explorer import Explorer
from synthetic import build_synthetic_config


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
