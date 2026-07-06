"""LLM inference contract (US-PRE-E00-S03, FR-PRE-001/002/003).

Every call uses a configured sampling temperature (US-PRE-E00-S05;
originally fixed at temperature=0 by US-PRE-E00-S03) and the resolved
canonical/explicit seed, and is constrained to return a single JSON object
matching the PRE output schema. At a fixed temperature, the same seed
reproduces the same output and a different seed produces a different, valid
variation of the same prompt. If the model's response is not schema-valid
JSON, one repair attempt is made with the parse/validation error fed back to
the model; if that also fails, LLM_OUTPUT_INVALID is raised.
"""
from __future__ import annotations

import json
import os
import time

from pydantic import ValidationError

from pre.core.errors import PREError
from pre.core.llm import loader
from pre.core.logging import decision_log
from pre.core.models.input import RefineInput
from pre.core.models.output import SCHEMA_VERSION, LLMGeneratedFields, RefineOutput
from pre.core.resource_loader import get_checkpoint_conventions

MAX_TOKENS_ENV = "PRE_MAX_TOKENS"
_DEFAULT_MAX_TOKENS = 512
TEMPERATURE_ENV = "PRE_TEMPERATURE"
_DEFAULT_TEMPERATURE = 0.7
_MAX_ATTEMPTS = 2  # first attempt + one repair attempt

_OUTPUT_JSON_SCHEMA = LLMGeneratedFields.model_json_schema()

_SYSTEM_PROMPT_TEMPLATE = (
    "You are the Prompt Refinement Engine for an image-generation pipeline. "
    "Expand the user's scene description into a hyper-detailed prompt for the "
    "'{checkpoint}' checkpoint: a rich, model-specific positive_prompt and a "
    "matching negative_prompt, plus camera/composition/lighting/color tags. "
    "Never contradict the user's stated scene; add only visually necessary "
    "detail that the user did not explicitly rule out.\n\n"
    "Checkpoint-specific conventions for '{checkpoint}': {style_notes}\n"
    "Default negative-prompt terms to include unless the scene contradicts "
    "them: {negative_defaults}."
)


def _system_prompt(checkpoint: str) -> str:
    conventions = get_checkpoint_conventions(checkpoint)
    return _SYSTEM_PROMPT_TEMPLATE.format(
        checkpoint=checkpoint,
        style_notes=conventions["style_notes"],
        negative_defaults=", ".join(conventions["negative_defaults"]),
    )


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


def _temperature() -> float:
    value = os.environ.get(TEMPERATURE_ENV)
    return float(value) if value is not None else _DEFAULT_TEMPERATURE


_JSON_ESCAPES = {"\n": "\\n", "\r": "\\r", "\t": "\\t"}


def _escape_control_chars_in_json_strings(text: str) -> str:
    """Escape raw control characters (e.g. literal newlines) found inside
    JSON string literals. Local LLMs generating long, multi-paragraph
    string values frequently emit a literal newline instead of an escaped
    "\\n" — invalid per the JSON spec (and rejected by pydantic's strict
    parser) even though the intent is unambiguous. Structural whitespace
    outside of strings is left untouched.
    """
    out: list[str] = []
    in_string = False
    escaped = False
    for ch in text:
        if in_string:
            if escaped:
                out.append(ch)
                escaped = False
            elif ch == "\\":
                out.append(ch)
                escaped = True
            elif ch == '"':
                in_string = False
                out.append(ch)
            elif ord(ch) < 0x20:
                out.append(_JSON_ESCAPES.get(ch, f"\\u{ord(ch):04x}"))
            else:
                out.append(ch)
        else:
            if ch == '"':
                in_string = True
            out.append(ch)
    return "".join(out)


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

    temperature = _temperature()

    for attempt in range(1, _MAX_ATTEMPTS + 1):
        # Reset the shared model's context before every generation. Without
        # this, reusing the same Llama instance across calls lets KV-cache/
        # prefix-reuse state leak between generations and breaks determinism
        # (FR-PRE-012) for a given (seed, temperature) pair — confirmed by
        # reproducing byte-identical output across repeated calls only once
        # reset() was added, on both CPU and GPU (CUDA) backends.
        llm.reset()
        backend = loader.get_model_info().backend
        start = time.perf_counter()
        completion = llm.create_chat_completion(
            messages=messages,
            temperature=temperature,
            seed=seed,
            max_tokens=_max_tokens(),
            response_format={"type": "json_object", "schema": _OUTPUT_JSON_SCHEMA},
        )
        latency_s = round(time.perf_counter() - start, 3)
        raw_text = completion["choices"][0]["message"]["content"]
        last_raw_text = raw_text

        try:
            sanitized_text = _escape_control_chars_in_json_strings(raw_text)
            generated = LLMGeneratedFields.model_validate_json(sanitized_text)
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
                temperature=temperature,
                checkpoint=inp.checkpoint,
                attempt=attempt,
                status="invalid_json",
                note=str(exc),
                backend=backend,
                latency_s=latency_s,
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
            temperature=temperature,
            checkpoint=inp.checkpoint,
            attempt=attempt,
            status="ok",
            backend=backend,
            latency_s=latency_s,
        )
        return output

    raise PREError(
        "LLM_OUTPUT_INVALID",
        "LLM did not return schema-valid JSON after one repair attempt.",
        details={"last_error": str(last_error), "last_raw_text": last_raw_text[:500]},
    )
