"""PRE input contract (SRS_PRE_v2.md §2, §4 UC-01/UC-02).

Field descriptions surface directly in the OpenAPI schema (Swagger /docs,
Insomnia's imported collection, etc.) so the request shape is self-explanatory
without needing this file open.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class SSAMetadata(BaseModel):
    """Optional scene context from an upstream Scene Analyzer (SSA) stage.

    Purely additive: these fields only fill gaps the LLM couldn't infer from
    `text` — they never override or contradict the user's own wording.
    """

    characters: list[str] = Field(
        default_factory=list,
        description=(
            "Known character names/ids present in the scene, if already "
            "identified by an upstream Scene Analyzer stage. Optional."
        ),
    )
    environment: str | None = Field(
        default=None,
        description=(
            "Scene environment/setting hint (e.g. 'castle', 'forest'). "
            "Optional context only — the user's text always wins on conflict."
        ),
    )
    mood: str | None = Field(
        default=None,
        description="Scene mood/emotional tone hint (e.g. 'epic', 'somber'). Optional.",
    )


class Controls(BaseModel):
    """User-declared creative controls layered on top of the free-text scene."""

    style_presets: list[str] = Field(
        default_factory=list,
        description=(
            "Named style presets to apply (e.g. 'cinematic'). Passed through "
            "as guidance in the LLM system prompt."
        ),
    )


class RefineInput(BaseModel):
    text: str = Field(
        description=(
            "The scene description to expand into a hyper-detailed prompt. "
            "Required — every job needs a narration/scene anchor."
        ),
    )
    checkpoint: str = Field(
        description=(
            "Target image-generation checkpoint id (e.g. 'sdxl-base-1.0', "
            "'juggernaut-xl'). Selects checkpoint-specific token conventions "
            "injected into the LLM system prompt (see pre/resources/checkpoints.json); "
            "an unregistered id falls back to generic conventions rather than failing."
        ),
    )
    ssa_metadata: SSAMetadata = Field(
        default_factory=SSAMetadata,
        description="Optional scene context from an upstream Scene Analyzer stage.",
    )
    controls: Controls = Field(
        default_factory=Controls,
        description="Optional user-declared creative controls.",
    )
    seed: int | None = Field(
        default=None,
        description=(
            "Explicit seed for reproducible output. If omitted, a canonical seed "
            "is derived deterministically from the SHA-256 hash of the normalised "
            "request. At a fixed sampling temperature, the same seed always "
            "reproduces the same output, and a different seed produces a "
            "different, valid variation of the same prompt."
        ),
    )
