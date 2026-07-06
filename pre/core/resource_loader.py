"""Static resource loading — checkpoint token conventions (US-PRE-E01-S01)."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

_RESOURCES_DIR = Path(__file__).resolve().parent.parent / "resources"

# Used for any checkpoint id not present in checkpoints.json, so an unknown
# checkpoint degrades to sensible generic behavior instead of failing.
_DEFAULT_CHECKPOINT_CONVENTIONS: dict[str, Any] = {
    "style_notes": (
        "No specific token conventions are known for this checkpoint — use "
        "standard descriptive, comma-separated tags and common quality boosters."
    ),
    "negative_defaults": ["low quality", "blurry", "watermark", "extra limbs", "deformed"],
}


@lru_cache(maxsize=1)
def _load_checkpoints() -> dict[str, Any]:
    path = _RESOURCES_DIR / "checkpoints.json"
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def get_checkpoint_conventions(checkpoint: str) -> dict[str, Any]:
    """Known token conventions for a checkpoint id, or a generic default."""
    return _load_checkpoints().get(checkpoint, _DEFAULT_CHECKPOINT_CONVENTIONS)
