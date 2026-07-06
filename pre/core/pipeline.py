"""refine() — the single entry point tying seed derivation to LLM inference."""
from __future__ import annotations

from dataclasses import dataclass

from pre.core.determinism.canonical import canonical_sha256_hex, resolve_seed
from pre.core.llm import loader
from pre.core.llm.inference import run_inference
from pre.core.logging import decision_log
from pre.core.models.input import RefineInput
from pre.core.models.output import RefineOutput


@dataclass(frozen=True)
class RefineResult:
    output: RefineOutput
    seed: int
    canonical_hash: str


def refine(raw_input: dict) -> RefineResult:
    inp = RefineInput.model_validate(raw_input)
    loader.load_model()  # no-op if already loaded (US-PRE-E00-S01)

    normalised = inp.model_dump(mode="json", exclude={"seed"})
    canonical_hash = canonical_sha256_hex(normalised)
    seed = resolve_seed(normalised, inp.seed)

    decision_log.record(
        stage="seed",
        seed=seed,
        explicit=inp.seed is not None,
        checkpoint=inp.checkpoint,
        canonical_hash=canonical_hash,
    )

    output = run_inference(inp, seed)
    return RefineResult(output=output, seed=seed, canonical_hash=canonical_hash)
