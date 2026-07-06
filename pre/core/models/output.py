"""PRE output contract — strict JSON (SRS_PRE_v2.md §7)."""
from __future__ import annotations

from pydantic import BaseModel, Field

SCHEMA_VERSION = "1.0.0"


class CameraTags(BaseModel):
    lens_mm: int | None = None
    shot: str | None = None
    angle: str | None = None
    dof: str | None = None


class LoraTag(BaseModel):
    id: str
    strength: float


class ControlNetTag(BaseModel):
    type: str
    strength: float


class LLMGeneratedFields(BaseModel):
    """The subset of the output contract the LLM itself must produce.

    schema_version/checkpoint/seed are engine-injected (US-PRE-E00-S02/S03),
    not left to the model — this is also the schema used to grammar-constrain
    LLM decoding so it can't drift onto unrelated keys.
    """

    positive_prompt: str
    negative_prompt: str
    camera: CameraTags = Field(default_factory=CameraTags)
    composition: list[str] = Field(default_factory=list)
    lighting: list[str] = Field(default_factory=list)
    color: list[str] = Field(default_factory=list)
    lora: list[LoraTag] = Field(default_factory=list)
    controlnet: list[ControlNetTag] = Field(default_factory=list)
    aspect_ratio: str = "16:9"
    seed_strategy: str = "random"
    style_presets: list[str] = Field(default_factory=list)


class RefineOutput(LLMGeneratedFields):
    schema_version: str = SCHEMA_VERSION
    checkpoint: str
    seed: int | None = None
