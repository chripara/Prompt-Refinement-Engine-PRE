import pytest

from pre.core.llm import loader


@pytest.fixture(autouse=True)
def _reset_loader_state():
    loader.reset()
    yield
    loader.reset()
