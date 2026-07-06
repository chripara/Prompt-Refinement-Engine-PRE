"""US-PRE-E00-S03 — end-to-end refine() against the real local LLM.

Skipped unless PRE_TEST_MODEL_PATH points at a real GGUF file.
"""
import os

import pytest

from pre.core.llm import loader
from pre.core.pipeline import refine

pytestmark = pytest.mark.skipif(
    not os.environ.get("PRE_TEST_MODEL_PATH"),
    reason="Set PRE_TEST_MODEL_PATH to a local GGUF file to run the real-model integration test.",
)


@pytest.fixture(autouse=True)
def _model_path(monkeypatch):
    monkeypatch.setenv(loader.MODEL_PATH_ENV, os.environ["PRE_TEST_MODEL_PATH"])


def test_refine_produces_valid_output():
    result = refine({
        "text": "A lone warrior stands in a ruined castle at dusk",
        "checkpoint": "sdxl-base-1.0",
    })
    assert result.output.positive_prompt
    assert result.output.negative_prompt
    assert result.output.checkpoint == "sdxl-base-1.0"
    assert result.output.seed == result.seed


def test_same_input_and_explicit_seed_is_deterministic():
    payload = {
        "text": "A lone warrior stands in a ruined castle at dusk",
        "checkpoint": "sdxl-base-1.0",
        "seed": 123456,
    }
    first = refine(payload)
    second = refine(payload)

    assert first.output.model_dump() == second.output.model_dump()
    assert first.seed == second.seed == 123456
