"""US-PRE-E00-S03/S05 — LLM inference contract, exercised against a fake LLM
(no GPU/model required): configured temperature + seed on every call,
JSON-only system prompt, one repair attempt, LLM_OUTPUT_INVALID if still
broken.
"""
import json

import pytest

from pre.core.errors import PREError
from pre.core.llm import inference, loader
from pre.core.models.input import RefineInput


class _FakeLLM:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []
        self.reset_count = 0

    def reset(self):
        self.reset_count += 1

    def create_chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        content = self._responses.pop(0)
        return {"choices": [{"message": {"content": content}}]}


def _valid_json() -> str:
    return json.dumps({
        "positive_prompt": "a warrior in dramatic lighting",
        "negative_prompt": "blurry, extra limbs",
        "aspect_ratio": "16:9",
        "seed_strategy": "random",
    })


def _install_fake_llm(monkeypatch, responses) -> _FakeLLM:
    fake = _FakeLLM(responses)
    monkeypatch.setattr(loader, "_llm", fake)
    monkeypatch.setattr(
        loader,
        "_model_info",
        loader.ModelInfo(
            name="fake-model", path="fake.gguf", quantization="Q4_K_M",
            load_time_s=0.0, n_gpu_layers=0, backend="cpu",
        ),
    )
    return fake


def test_valid_first_response_returns_output(monkeypatch):
    fake = _install_fake_llm(monkeypatch, [_valid_json()])

    inp = RefineInput(text="A warrior at night", checkpoint="sdxl-base-1.0")
    output = inference.run_inference(inp, seed=42)

    assert output.positive_prompt == "a warrior in dramatic lighting"
    assert output.seed == 42
    assert output.checkpoint == "sdxl-base-1.0"
    assert len(fake.calls) == 1
    assert fake.calls[0]["temperature"] == inference._DEFAULT_TEMPERATURE
    assert fake.calls[0]["seed"] == 42
    assert fake.calls[0]["response_format"]["type"] == "json_object"
    assert "schema" in fake.calls[0]["response_format"]
    assert fake.reset_count == 1  # context reset before the generation (determinism)


def test_temperature_env_override(monkeypatch):
    monkeypatch.setenv(inference.TEMPERATURE_ENV, "1.2")
    fake = _install_fake_llm(monkeypatch, [_valid_json()])

    inp = RefineInput(text="A warrior at night", checkpoint="sdxl-base-1.0")
    inference.run_inference(inp, seed=42)

    assert fake.calls[0]["temperature"] == 1.2


def test_invalid_then_valid_repairs_on_second_attempt(monkeypatch):
    fake = _install_fake_llm(monkeypatch, ["not json", _valid_json()])

    inp = RefineInput(text="A mage at dawn", checkpoint="sdxl-base-1.0")
    output = inference.run_inference(inp, seed=7)

    assert output.positive_prompt == "a warrior in dramatic lighting"
    assert len(fake.calls) == 2
    # repair attempt must still use the same temperature and seed
    assert fake.calls[1]["temperature"] == inference._DEFAULT_TEMPERATURE
    assert fake.calls[1]["seed"] == 7
    assert fake.reset_count == 2  # reset before both the initial and repair attempt


def test_invalid_twice_raises_llm_output_invalid(monkeypatch):
    fake = _install_fake_llm(monkeypatch, ["not json", "still not json"])

    inp = RefineInput(text="A rogue at midnight", checkpoint="sdxl-base-1.0")
    with pytest.raises(PREError) as exc:
        inference.run_inference(inp, seed=1)

    assert exc.value.code == "LLM_OUTPUT_INVALID"
    assert len(fake.calls) == 2


def test_literal_newline_in_string_value_is_repaired(monkeypatch):
    # Local LLMs writing long, multi-paragraph strings sometimes emit a raw
    # newline instead of an escaped "\n" — invalid JSON per spec, but an
    # unambiguous, auto-repairable formatting slip, not a real schema error.
    raw_with_literal_newline = (
        '{"positive_prompt": "Line one.\n\nLine two.", '
        '"negative_prompt": "blurry", "aspect_ratio": "16:9", "seed_strategy": "random"}'
    )
    fake = _install_fake_llm(monkeypatch, [raw_with_literal_newline])

    inp = RefineInput(text="A rogue at midnight", checkpoint="sdxl-base-1.0")
    output = inference.run_inference(inp, seed=1)

    assert output.positive_prompt == "Line one.\n\nLine two."
    assert len(fake.calls) == 1  # repaired without needing the retry attempt


def test_system_prompt_is_checkpoint_specific():
    sdxl_prompt = inference._system_prompt("sdxl-base-1.0")
    juggernaut_prompt = inference._system_prompt("juggernaut-xl")

    assert "sdxl-base-1.0" in sdxl_prompt
    assert "juggernaut-xl" in juggernaut_prompt
    assert sdxl_prompt != juggernaut_prompt


def test_system_prompt_falls_back_for_unknown_checkpoint():
    prompt = inference._system_prompt("some-checkpoint-nobody-registered")

    assert "some-checkpoint-nobody-registered" in prompt
