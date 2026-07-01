# Software Requirements Specification — Prompt Refinement Engine v2

**Component:** Prompt Refinement Engine (`PRE`)
**Alignment:** ISO/IEC/IEEE 29148 practice, adapted.
**Status:** Draft for review — v2 (generative local LLM + seed).
**Runtime baseline:** Python 3.11, Windows-first, fully offline, CLI + Python API.
**Backlog:** [`EPICS_AND_STORIES_v2.md`](EPICS_AND_STORIES_v2.md)

---

## 1. Introduction

### 1.1 Purpose
Define what the Prompt Refinement Engine SHALL do, measurably enough to build and verify without external context.

### 1.2 Scope
Expand a short or long English user description into a **strictly-structured JSON** of model-specific prompts and parameters (positive/negative prompts, camera/composition/lighting/color tags, LoRA, ControlNet, aspect ratio, seed strategy) for the Image engine. Operates standalone (CLI/API) or as the stage between SSA and IMG. Out of scope: scene segmentation (SSA), image generation (IMG), multi-language.

**v2 Architecture:** The expansion is performed by a **local LLM** running entirely on-device (no internet), guided by structured prompt templates and constrained to `temperature=0` with a **fixed seed** derived from the canonical input hash. This replaces the rule-based lexicon pipeline of v1 while preserving the determinism and offline guarantees.

### 1.3 Definitions
**Refinement** — the deterministic transform from user text (+ optional SSA metadata) to the PRE output JSON, performed by a local LLM at temperature=0 with fixed seed. **Checkpoint** — the target image model the prompt is tuned for. **Preset** — a predefined style bundle. **Conflict resolution** — internal, silent selection of the higher-priority option when inputs disagree. **Canonical seed** — a 32-bit integer derived from the SHA-256 of the normalised input JSON; used as the LLM inference seed unless the user provides one explicitly.

---

## 2. Overall description

### 2.1 Product perspective
A replaceable, loosely-coupled engine. Single prompt per run (no batch). Contract-based JSON I/O. Fidelity to user intent is prioritized over stylistic uniformity when the two conflict. The local LLM acts as an intelligent expander guided by structured templates; it does not receive open-ended creative freedom — the system prompt constrains it to produce strictly valid JSON conforming to the output contract.

### 2.2 Operational environment
Windows-first, Python 3.11, **fully offline**. Local LLM runs on-device (CPU or GPU). CLI accepts a JSON input; a Python API is also provided.

### 2.3 LLM model constraints
- Model runs **locally** — no API calls, no internet access.
- Recommended: quantized 7B-class model (GGUF format via `llama-cpp-python`, e.g., Mistral-7B-Instruct or LLaMA-3-8B-Instruct, Q4_K_M quantization).
- Minimum VRAM (GPU): 4 GB for Q4 quantized; CPU inference also supported (slower).
- Minimum RAM (CPU-only): 8 GB.
- Model file is configured via `PRE_MODEL_PATH` environment variable; startup fails with `MODEL_NOT_FOUND` if absent.

---

## 3. Stakeholder needs (informative)
From the questionnaire: re-express user text into hyper-detailed, model-specific prompts + parameters; English only; single prompt per run; optional SSA metadata; accept user style/LoRA/ControlNet/constraints; output strict JSON with positive+negative prompts and camera/composition/lighting/color/LoRA/ControlNet/aspect/seed tags; predefined presets only; silent best-fidelity conflict resolution (no warnings); LoRA fusion up to ~3 with conflict detection; ControlNet added only when useful and removed when redundant; per-scene aspect and seed reproducibility for recurring characters; deterministic (via seed + temperature=0); offline via local LLM; model-specific (not model-agnostic); CLI + API; decision logging.

---

## 4. Use cases
- **UC-01 Standalone refine:** user text (+ checkpoint) → PRE JSON via CLI/API.
- **UC-02 Pipeline refine:** SSA scene object → PRE JSON for IMG.
- **UC-03 Reproduce:** same input + same seed → identical JSON.
- **UC-04 Reject/repair:** malformed input → repaired output or structured error.
- **UC-05 Explicit seed:** user supplies seed → LLM uses that seed exactly; output records it.

---

## 5. Functional requirements (testable)

| ID | Requirement | Acceptance (summary) | Phase |
|----|-------------|----------------------|-------|
| **FR-PRE-001** | Expand short/long English text into hyper-detailed, **model-specific** prompts using a local LLM. | Output adheres to intent and to the declared checkpoint's conventions. | mvp |
| **FR-PRE-002** | Emit **strict JSON** with positive + negative prompts and parameter tags (camera, composition, lighting, color, LoRA, ControlNet, aspect, seed). | Golden JSON fixtures pass schema validation. | mvp |
| **FR-PRE-003** | Run as **CLI (JSON in)** and as a **Python API**, sharing one core library. | Both entry points produce identical output for identical input. | mvp |
| **FR-PRE-004** | Accept optional **SSA metadata** (characters, environment, mood) and merge deterministically. | Pipeline mode enriches output; absent fields fall back cleanly. | mvp |
| **FR-PRE-005** | Accept user **style preset / LoRA / ControlNet / constraints** (predefined options). | Declarative controls honored; predefined-only. | mvp |
| **FR-PRE-006** | Resolve conflicts **silently** preferring fidelity to user text; record decisions in logs. | No warning UX; logs contain each decision. | mvp |
| **FR-PRE-007** | Add **hidden detail** required for rendering even when not user-specified. | LLM enriches output beyond literal user text; spot checks show completeness without contradicting intent. | rich |
| **FR-PRE-008** | Produce layered **negative prompts** (scene + style + artifact bans), de-duplicated. | Negative-merge tests pass. | mvp |
| **FR-PRE-009** | Support **LoRA fusion ≤ N (default 3)** with strengths and conflict detection. | Two-character-LoRA conflict detected; N configurable. | p1 |
| **FR-PRE-010** | Manage **ControlNet lifecycle**: add when useful, strip when redundant, multi-CN, style-compatible. | Compatibility-matrix tests pass. | p1 |
| **FR-PRE-011** | Propose **aspect ratio per scene** and **seed strategy** (reproducible for recurring characters). | Recurring character → same seed across scenes when SSA character id is stable. | mvp |
| **FR-PRE-012** | Be **deterministic**: same input + same seed → same JSON. | Hash test on canonical JSON with fixed seed. | mvp |
| **FR-PRE-013** | Run **100% offline** using a local LLM. | Network-off test passes; LLM inference uses no external calls. | mvp |
| **FR-PRE-014** | Emit **structured decision logs** including the seed used. | Documented log schema; decisions and seed recorded. | mvp |
| **FR-PRE-015** | **Validate/repair** malformed prompts; check LoRA/ControlNet availability; validate model compatibility. | Error catalog; repair-or-reject behavior. | mvp |
| **FR-PRE-016** *(SHOULD)* | **Latency**: single-prompt refine within acceptable bounds on reference hardware. | P95 ≤ 3 s on GPU (RTX 3060 or better); P95 ≤ 15 s on CPU-only. | rich |
| **FR-PRE-017** | **Seed management**: derive canonical seed from SHA-256 of normalised input JSON (mod 2³²); accept user-provided seed override. | Same canonical input → same seed; user seed override is recorded and used. | mvp |
| **FR-PRE-018** | **LLM model management**: load model at startup from `PRE_MODEL_PATH`; fail fast with `MODEL_NOT_FOUND` if absent; expose model info in API health endpoint. | Startup test: missing model → structured error; health endpoint returns model name + quantization. | mvp |

---

## 6. Non-functional requirements (measurable)

| ID | Attribute | Statement | Verification |
|----|-----------|-----------|--------------|
| **NFR-001** | Determinism | Identical input + identical seed → identical JSON. | Hash diff with fixed seed |
| **NFR-002** | Performance (GPU) | Single-prompt refine ≤ 3 s P95 on RTX 3060 or equivalent. | Timed runs |
| **NFR-003** | Performance (CPU) | Single-prompt refine ≤ 15 s P95 on 8-core modern CPU. | Timed runs |
| **NFR-004** | Offline | Zero runtime network dependency; LLM runs entirely on-device. | Network-blocked test |
| **NFR-005** | Platform | Windows-first; CI on Windows. | CI |
| **NFR-006** | Logging | Structured decision logs with stable schema; seed recorded per run. | Log review |
| **NFR-007** | Contract stability | Output JSON schema versioned. | Schema review |
| **NFR-008** | VRAM | Runs with ≤ 4 GB VRAM (Q4_K_M quantized model). | Profiler |
| **NFR-009** | RAM (CPU) | Runs with ≤ 8 GB RAM on CPU-only inference. | Profiler |
| **NFR-010** | Model format | Model is GGUF, loaded via `llama-cpp-python`; no mandatory GPU required. | Integration test |

---

## 7. Output contract (strict JSON)

The PRE output is the IMG-consumable request (positive/negative prompts + parameters). It SHALL be a versioned, schema-validated object. Logical shape:

```json
{
  "schema_version": "1.0.0",
  "checkpoint": "<target model id>",
  "positive_prompt": "<string>",
  "negative_prompt": "<string>",
  "camera": { "lens_mm": null, "shot": null, "angle": null, "dof": null },
  "composition": ["<rule>"],
  "lighting": ["<tag>"],
  "color": ["<tag>"],
  "lora": [{ "id": "<string>", "strength": 0.0 }],
  "controlnet": [{ "type": "<string>", "strength": 0.0 }],
  "aspect_ratio": "<preset>",
  "seed_strategy": "consistent_character|consistent_scene|random",
  "seed": "<integer|null>",
  "style_presets": ["<preset>"]
}
```

**New in v2:** `"seed"` field — the actual integer seed used for LLM inference. If the user provided an explicit seed it is echoed here; otherwise the hash-derived canonical seed is recorded so any run is fully reproducible.

This maps directly onto the IMG canonical request (IMG SRS §11.2).

---

## 8. Determinism

**Seed derivation (automatic):**
```python
import hashlib, json

canonical_input = json.dumps(normalised_input, sort_keys=True, ensure_ascii=True)
canonical_seed  = int(hashlib.sha256(canonical_input.encode()).hexdigest(), 16) % (2**32)
```

If the user supplies an explicit `seed` in the input, that value is used and the hash-derived seed is discarded.

LLM inference is called with `temperature=0` and `seed=canonical_seed`. Same input + same model version + same seed → identical token sequence → identical JSON output. (FR-PRE-012 / NFR-001.)

---

## 9. Error handling
Codes: `INVALID_INPUT`, `SCHEMA_VALIDATION_FAILED`, `RESOURCE_NOT_FOUND` (missing LoRA/ControlNet), `MODEL_INCOMPATIBLE`, `MODEL_NOT_FOUND` (LLM model file absent or corrupt), `LLM_OUTPUT_INVALID` (LLM returned non-parseable JSON; retry once with re-prompt then reject), `INTERNAL_ERROR`. Malformed prompts are repaired where possible, else rejected with `INVALID_INPUT`.

---

## 10. Acceptance criteria (selected)

| FR | Test | Expected |
|----|------|----------|
| FR-PRE-002 | Validate output vs schema | Strict pass; positive+negative present |
| FR-PRE-006 | Conflicting style + text | Fidelity wins; decision logged; no warning |
| FR-PRE-009 | Two character LoRAs | Conflict detected; ≤ N enforced |
| FR-PRE-010 | Redundant ControlNet | Stripped |
| FR-PRE-012 | Same input twice (fixed seed) | Identical JSON (hash match) |
| FR-PRE-013 | Network blocked | Succeeds (local LLM, no external calls) |
| FR-PRE-015 | Malformed input | Repaired or `INVALID_INPUT` |
| FR-PRE-017 | Same canonical input | Same seed derived; user seed override echoed in output |
| FR-PRE-018 | Missing model file | `MODEL_NOT_FOUND` at startup |

---

## 11. Traceability & change policy
Each FR maps to a story in [`EPICS_AND_STORIES_v2.md`](EPICS_AND_STORIES_v2.md). **Change policy:** SOLID / Open–Closed — extend with new stories; modify existing requirements only when necessary. Update this file and the backlog together on scope change.
