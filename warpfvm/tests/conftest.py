import importlib.util

import numpy as np
import pytest
import warp as wp

wp.config.log_level = wp.LOG_WARNING

# Collected from the repository root too; skip quietly where the subproject is not installed.
if importlib.util.find_spec("warpfvm") is None:
    collect_ignore_glob = ["test_*.py"]


def pytest_collection_modifyitems(config, items):
    wp.init()
    if wp.is_cuda_available():
        return
    skip = pytest.mark.skip(reason="no CUDA device available")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def cuda():
    wp.init()
    if not wp.is_cuda_available():
        pytest.skip("no CUDA device available")
    return wp.get_device("cuda:0")


@pytest.fixture
def rng():
    return np.random.default_rng(12345)
