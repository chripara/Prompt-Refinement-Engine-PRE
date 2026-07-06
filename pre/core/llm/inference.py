"""LLM inference contract (US-PRE-E00-S03, FR-PRE-001/002/003).

Every call uses temperature=0 and the resolved canonical/explicit seed, and
is constrained to return a single JSON object matching the PRE output
schema. If the model's first response is not schema-valid JSON, one repair
attempt is made with the parse/validation error fed back to the model;
if that also fails, LLM_OUTPUT_INVALID is raised.
"""
from __future__ import annotations

import json
import os

from pydantic import ValidationError

from pre.core.errors import PREError
from pre.core.llm import loader
from pre.core.logging import decision_log
from pre.core.models.input import RefineInput
from pre.core.models.output import SCHEMA_VERSION, LLMGeneratedFields, RefineOutput

MAX_TOKENS_ENV = "PRE_MAX_TOKENS"
_DEFAULT_MAX_TOKENS = 512
_MAX_ATTEMPTS = 2  # first attempt + one repair attempt

_OUTPUT_JSON_SCHEMA = LLMGeneratedFields.model_json_schema()

_SYSTEM_PROMPT_TEMPLATE = (
    "You are the Prompt Refinement Engine for an image-generation pipeline. "
    "Expand the user's scene description into a hyper-detailed prompt for the "
    "'{checkpoint}' checkpoint: a rich, model-specific positive_prompt and a "
    "matching negative_prompt, plus camera/composition/lighting/color tags. "
    "Never contradict the user's stated scene; add only visually necessary "
    "detail that the user did not explicitly rule out."
)


def _system_prompt(checkpoint: str) -> str:
    return _SYSTEM_PROMPT_TEMPLATE.format(checkpoint=checkpoint)


def _user_message(inp: RefineInput) -> str:
    lines = [f"Scene description: {inp.text}"]
    if inp.ssa_metadata.characters:
        lines.append(f"Characters: {', '.join(inp.ssa_metadata.characters)}")
    if inp.ssa_metadata.environment:
        lines.append(f"Environment: {inp.ssa_metadata.environment}")
    if inp.ssa_metadata.mood:
        lines.append(f"Mood: {inp.ssa_metadata.mood}")
    if inp.controls.style_presets:
        lines.append(f"Style presets: {', '.join(inp.controls.style_presets)}")
    return "\n".join(lines)


def _max_tokens() -> int:
    value = os.environ.get(MAX_TOKENS_ENV)
    return int(value) if value is not None else _DEFAULT_MAX_TOKENS


def run_inference(inp: RefineInput, seed: int) -> RefineOutput:
    """Call the local LLM under the deterministic contract and return validated output.

    Raises:
        PREError(LLM_OUTPUT_INVALID): the model did not return schema-valid JSON
            even after one repair attempt.
    """
    llm = loader.get_llm()

    messages: list[dict] = [
        {"role": "system", "content": _system_prompt(inp.checkpoint)},
        {"role": "user", "content": _user_message(inp)},
    ]

    last_error: Exception | None = None
    last_raw_text = ""

    for attempt in range(1, _MAX_ATTEMPTS + 1):
        # Reset the shared model's context before every generation. Without
        # this, reusing the same Llama instance across calls lets KV-cache/
        # prefix-reuse state leak between generations and breaks determinism
        # (FR-PRE-012) even at temperature=0 with a fixed seed — confirmed by
        # reproducing byte-identical output across repeated calls only once
        # reset() was added, on both CPU and GPU (CUDA) backends.
        llm.reset()
        completion = llm.create_chat_completion(
            messages=messages,
            temperature=0,
            seed=seed,
            max_tokens=_max_tokens(),
            response_format={"type": "json_object", "schema": _OUTPUT_JSON_SCHEMA},
        )
        raw_text = completion["choices"][0]["message"]["content"]
        last_raw_text = raw_text

        try:
            generated = LLMGeneratedFields.model_validate_json(raw_text)
            output = RefineOutput(
                schema_version=SCHEMA_VERSION,
                checkpoint=inp.checkpoint,
                seed=seed,
                **generated.model_dump(),
            )
        except (json.JSONDecodeError, ValidationError) as exc:
            last_error = exc
            decision_log.record(
                stage="inference",
                seed=seed,
                checkpoint=inp.checkpoint,
                attempt=attempt,
                status="invalid_json",
                note=str(exc),
            )
            messages.append({"role": "assistant", "content": raw_text})
            messages.append({
                "role": "user",
                "content": (
                    "That response was not valid JSON matching the required schema "
                    f"({exc}). Respond again with ONLY the corrected JSON object."
                ),
            })
            continue

        decision_log.record(
            stage="inference",
            seed=seed,
            checkpoint=inp.checkpoint,
            attempt=attempt,
            status="ok",
        )
        return output

    raise PREError(
        "LLM_OUTPUT_INVALID",
        "LLM did not return schema-valid JSON after one repair attempt.",
        details={"last_error": str(last_error), "last_raw_text": last_raw_text[:500]},
    )
