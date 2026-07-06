"""US-PRE-E00-S03/S05 — end-to-end refine() against the real local LLM.

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


def test_same_seed_repeated_five_times_is_always_identical():
    # Exact user-specified acceptance case: seed=10, run 5 times in a row ->
    # always the same output.
    payload = {
        "text": "A lone warrior stands in a ruined castle at dusk",
        "checkpoint": "sdxl-base-1.0",
        "seed": 10,
    }
    results = [refine(payload) for _ in range(5)]

    assert all(r.seed == 10 for r in results)
    first_dump = results[0].output.model_dump()
    assert all(r.output.model_dump() == first_dump for r in results[1:])


def test_different_seed_produces_different_variation():
    base = {
        "text": "A lone warrior stands in a ruined castle at dusk",
        "checkpoint": "sdxl-base-1.0",
    }
    first = refine({**base, "seed": 111})
    second = refine({**base, "seed": 222})

    assert first.seed == 111
    assert second.seed == 222
    assert first.output.positive_prompt != second.output.positive_prompt


def test_seed_10_vs_seed_11_matches_user_acceptance_case():
    # Exact user-specified acceptance case: seed=10 repeated is stable;
    # seed=11 on the same input must differ from it.
    base = {
        "text": "A lone warrior stands in a ruined castle at dusk",
        "checkpoint": "sdxl-base-1.0",
    }
    seed_10_runs = [refine({**base, "seed": 10}) for _ in range(3)]
    seed_11 = refine({**base, "seed": 11})

    first_dump = seed_10_runs[0].output.model_dump()
    assert all(r.output.model_dump() == first_dump for r in seed_10_runs[1:])
    assert seed_11.output.positive_prompt != seed_10_runs[0].output.positive_prompt


def test_checkpoint_specific_system_prompt_reaches_the_real_llm():
    # US-PRE-E01-S01 AC: "LLM system prompt includes checkpoint name and its
    # known token conventions." Empirically, asserting the *output* differs
    # across checkpoints is unreliable: sdxl-base-1.0 and juggernaut-xl
    # produced byte-identical positive_prompt for this seed/input even across
    # separate fresh processes (no caching involved) — a general-purpose LLM
    # at temperature=0.7 isn't guaranteed to meaningfully vary its wording
    # based on stylistic system-prompt guidance alone. The fair, non-flaky
    # signal is that the checkpoint-specific system prompt text is genuinely
    # sent to the real model — spying on the live call rather than asserting
    # on its (model-dependent) response content.
    loader.load_model()
    llm = loader.get_llm()
    original_create_chat_completion = llm.create_chat_completion
    captured: dict = {}

    def _spy(**kwargs):
        captured["messages"] = kwargs["messages"]
        return original_create_chat_completion(**kwargs)

    llm.create_chat_completion = _spy
    try:
        refine({
            "text": "A lone warrior stands in a ruined castle at dusk",
            "checkpoint": "juggernaut-xl",
            "seed": 1,
        })
    finally:
        llm.create_chat_completion = original_create_chat_completion

    system_message = captured["messages"][0]["content"]
    assert "juggernaut-xl" in system_message
    assert "hotorealistic" in system_message  # from checkpoints.json style_notes
    assert "cartoon" in system_message  # from checkpoints.json negative_defaults
