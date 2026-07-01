"""US-PRE-E00-S01 — Local LLM model loading."""
import os

import pytest

from pre.core.errors import PREError
from pre.core.llm import loader


def test_missing_env_raises_model_not_found(monkeypatch):
    monkeypatch.delenv(loader.MODEL_PATH_ENV, raising=False)
    with pytest.raises(PREError) as exc:
        loader.load_model()
    assert exc.value.code == "MODEL_NOT_FOUND"


def test_missing_file_raises_model_not_found(monkeypatch, tmp_path):
    monkeypatch.setenv(loader.MODEL_PATH_ENV, str(tmp_path / "nope.gguf"))
    with pytest.raises(PREError) as exc:
        loader.load_model()
    assert exc.value.code == "MODEL_NOT_FOUND"


def test_get_model_info_before_load_raises():
    with pytest.raises(PREError) as exc:
        loader.get_model_info()
    assert exc.value.code == "MODEL_NOT_FOUND"


def test_get_llm_before_load_raises():
    with pytest.raises(PREError):
        loader.get_llm()


@pytest.mark.skipif(
    not os.environ.get("PRE_TEST_MODEL_PATH"),
    reason="Set PRE_TEST_MODEL_PATH to a local GGUF file to run the real-model integration test.",
)
def test_real_model_loads_once_and_is_reused(monkeypatch):
    monkeypatch.setenv(loader.MODEL_PATH_ENV, os.environ["PRE_TEST_MODEL_PATH"])

    llm_first = loader.load_model()
    llm_second = loader.load_model()

    assert llm_first is llm_second, "model must be loaded once and reused, not reloaded per call"

    info = loader.get_model_info()
    assert info.load_time_s > 0
    assert info.quantization != "unknown"
    assert info.path == os.environ["PRE_TEST_MODEL_PATH"]
