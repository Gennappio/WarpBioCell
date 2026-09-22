import pytest
import warp as wp

from warpbiocell.device import cuda_available


def pytest_collection_modifyitems(config, items):
    if cuda_available():
        return
    skip = pytest.mark.skip(reason="no CUDA device available")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def cpu():
    wp.init()
    return wp.get_device("cpu")


@pytest.fixture(scope="session")
def cuda():
    wp.init()
    if not wp.is_cuda_available():
        pytest.skip("no CUDA device available")
    return wp.get_device("cuda:0")
