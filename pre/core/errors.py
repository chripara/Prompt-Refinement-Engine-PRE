"""PRE error catalog (SRS_PRE_v2.md §9)."""
from __future__ import annotations

ERROR_CODES = frozenset({
    "INVALID_INPUT",
    "SCHEMA_VALIDATION_FAILED",
    "RESOURCE_NOT_FOUND",
    "MODEL_INCOMPATIBLE",
    "MODEL_NOT_FOUND",
    "LLM_OUTPUT_INVALID",
    "INTERNAL_ERROR",
})


class PREError(Exception):
    """Raised for any structured, catalog error the API/CLI must surface to callers."""

    def __init__(self, code: str, message: str, details: dict | None = None):
        if code not in ERROR_CODES:
            raise ValueError(f"Unknown PRE error code: {code!r}")
        self.code = code
        self.message = message
        self.details = details or {}
        super().__init__(f"{code}: {message}")

    def to_dict(self) -> dict:
        return {"error": self.code, "message": self.message, "details": self.details}
