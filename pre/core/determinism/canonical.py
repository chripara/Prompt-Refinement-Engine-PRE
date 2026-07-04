"""Canonical seed derivation (US-PRE-E00-S02, FR-PRE-017).

canonical_seed = int(sha256(json.dumps(normalised_input, sort_keys=True)).hexdigest(), 16) % 2**32

An explicit user-supplied seed always overrides the hash-derived value.
"""
from __future__ import annotations

import hashlib
import json

_SEED_MODULUS = 2 ** 32


def canonical_json(normalised_input: dict) -> str:
    return json.dumps(normalised_input, sort_keys=True, ensure_ascii=True)


def canonical_sha256_hex(normalised_input: dict) -> str:
    return hashlib.sha256(canonical_json(normalised_input).encode()).hexdigest()


def derive_seed(normalised_input: dict) -> int:
    return int(canonical_sha256_hex(normalised_input), 16) % _SEED_MODULUS


def resolve_seed(normalised_input: dict, explicit_seed: int | None) -> int:
    """Explicit seed wins; otherwise derive deterministically from the input."""
    if explicit_seed is not None:
        return explicit_seed
    return derive_seed(normalised_input)
