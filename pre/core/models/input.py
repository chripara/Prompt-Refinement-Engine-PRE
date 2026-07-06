"""PRE input contract (SRS_PRE_v2.md §2, §4 UC-01/UC-02)."""
from __future__ import annotations

from pydantic import BaseModel, Field


class SSAMetadata(BaseModel):
    characters: list[str] = Field(default_factory=list)
    environment: str | None = None
    mood: str | None = None


class Controls(BaseModel):
    style_presets: list[str] = Field(default_factory=list)


class RefineInput(BaseModel):
    text: str
    checkpoint: str
    ssa_metadata: SSAMetadata = Field(default_factory=SSAMetadata)
    controls: Controls = Field(default_factory=Controls)
    seed: int | None = None
