"""Structured decision log (FR-PRE-014) — one JSON object per line.

Every entry that affects the output (seed choice, conflict resolution,
LLM retries, ...) is recorded here so runs are auditable and reproducible.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

LOG_PATH_ENV = "PRE_LOG_PATH"
_DEFAULT_LOG_PATH = Path("logs") / "decisions.jsonl"


def _log_path() -> Path:
    return Path(os.environ.get(LOG_PATH_ENV, _DEFAULT_LOG_PATH))


def record(stage: str, **fields) -> None:
    path = _log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {"ts": time.time(), "stage": stage, **fields}
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, sort_keys=True) + "\n")
