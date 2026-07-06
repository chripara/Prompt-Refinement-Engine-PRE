import pytest

from pre.core.llm import loader
from pre.core.logging import decision_log


@pytest.fixture(autouse=True)
def _reset_loader_state():
    loader.reset()
    yield
    loader.reset()


@pytest.fixture(autouse=True)
def _isolate_decision_log(tmp_path, monkeypatch):
    # Keep test runs from writing to the real logs/decisions.jsonl in the repo.
    monkeypatch.setenv(decision_log.LOG_PATH_ENV, str(tmp_path / "decisions.jsonl"))
