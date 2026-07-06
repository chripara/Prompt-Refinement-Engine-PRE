# Epics & User Stories — Prompt Refinement Engine v2

English-only backlog for the **Prompt Refinement Engine** (`PRE`).
Source of truth parsed by [`../tools/jira_importer.py`](../tools/jira_importer.py) (run with `--component PRE`).

**v2 Architecture:** Local LLM (GGUF / llama-cpp-python) with temperature=0 and canonical seed replaces the rule-based lexicon pipeline. Determinism is guaranteed via seed, not lookup tables.

**Format contract (do not break — the importer parses it):**
- Epic header: `## Epic <ID> — <Title>`
- Story header: `### <STORY_ID>: <Title>`
- Meta line: `- priority_weight: <int> · estimate: <range> · phase: <thin|rich|mvp|p1|p2|p3>`
- Statement line: `- As a ... I want ... so that ...`
- Acceptance list: `- Acceptance criteria:` followed by `  - ` bullets.

**Change policy (SOLID / Open–Closed):** open for extension, closed for modification. A new need becomes a **new story** (next/intermediate step); refine an existing story **only if absolutely necessary** (correct an error/regression). Don't break the existing structure or re-open completed stories.

---

## Epic E00 — LLM Runtime

Bootstrap the local LLM, seed management, and inference contract that all other epics depend on.

### US-PRE-E00-S01: Local LLM model loading
- priority_weight: 100 · estimate: 2–4 d · phase: mvp
- As a platform engineer, I want the PRE to load a GGUF model from `PRE_MODEL_PATH` at startup and fail fast with `MODEL_NOT_FOUND` if absent, so that misconfigured environments surface immediately.
- Acceptance criteria:
  - Startup test: model present → engine ready; model absent → `MODEL_NOT_FOUND` error, no crash.
  - Health endpoint (`GET /health`) returns model name, quantization type, and load time.
  - Model is loaded once and reused across all requests (no re-load per call).

### US-PRE-E00-S02: Canonical seed derivation
- priority_weight: 99 · estimate: 1–2 d · phase: mvp
- As an SRE, I want the engine to derive a deterministic 32-bit seed from the SHA-256 of the normalised input JSON, so that the same input always produces the same seed without user intervention.
- Acceptance criteria:
  - `canonical_seed = int(sha256(json.dumps(normalised_input, sort_keys=True)).hexdigest(), 16) % 2**32`
  - If user supplies an explicit `seed` field in input, that value is used instead and echoed in output.
  - The seed used is recorded in the output JSON (`"seed": <int>`) and in the decision log.
  - Hash test: two runs of identical input → identical seed → identical LLM output.

### US-PRE-E00-S03: LLM inference contract
- priority_weight: 98 · estimate: 2–3 d · phase: mvp
- As a platform engineer, I want the LLM to be called with `temperature=0`, the canonical seed, and a strict JSON-output system prompt, so that output is deterministic and schema-compliant.
- Acceptance criteria:
  - LLM is called with `temperature=0` and `seed=canonical_seed` on every run.
  - System prompt instructs model to return **only** valid JSON conforming to the PRE output schema (§7 SRS v2).
  - If LLM returns non-parseable JSON, engine retries once with an explicit repair prompt; if still invalid, raises `LLM_OUTPUT_INVALID`.
  - Pydantic schema validation runs on LLM output before it is returned to the caller.

### US-PRE-E00-S04: GPU / CPU runtime support
- priority_weight: 97 · estimate: 1–2 d · phase: mvp
- As a user, I want the engine to auto-detect GPU availability and use it when present, falling back to CPU, so that it runs on any Windows machine.
- Acceptance criteria:
  - `n_gpu_layers=-1` when CUDA is available; `n_gpu_layers=0` on CPU-only.
  - Environment variable `PRE_GPU_LAYERS` overrides auto-detection.
  - Latency logged per run (GPU path vs CPU path distinguishable in logs).

### US-PRE-E00-S05: Configurable temperature for seed-driven variation
- priority_weight: 96 · estimate: 1–2 d · phase: mvp
- As an operator, I want a different `seed` at a fixed sampling temperature to produce a genuinely different variation of the same prompt (not the identical text every time), so that seed is a meaningful creative control rather than inert bookkeeping.
- Acceptance criteria:
  - Sampling temperature is configurable via `PRE_TEMPERATURE` (default > 0, not the `temperature=0` fixed by US-PRE-E00-S03); recorded per run in the decision log.
  - At a fixed temperature, two different seeds over the same input produce different `positive_prompt`/`negative_prompt` text.
  - Determinism (FR-PRE-012) is redefined as: same input + same seed + same temperature configuration → identical JSON output — a stronger/updated statement than the temperature=0-only guarantee from US-PRE-E00-S03, superseding it going forward without editing that story's historical record.
  - `US-PRE-E00-S03` is left unmodified as a historical record of the initial temperature=0 decision; this story is the documented policy change (SRS_PRE_v2.md updated alongside).

---

## Epic E01 — Role in the pipeline

Expand user text into model-specific, hyper-detailed prompts and parameters for the Image engine — standalone CLI/API or sandwiched between SSA and Image.

### US-PRE-E01-S01: Hyper-detailed SDXL-oriented prompts
- priority_weight: 97 · estimate: 3–6 d · phase: mvp
- As an image pipeline engineer, I want short or long user descriptions expanded into production-grade prompts with fidelity to intent via a local LLM, so that generated images match controllable detail.
- Acceptance criteria:
  - Output JSON validates against the PRE schema and includes positive and negative prompts.
  - Expansion is targeted to the declared checkpoint via a checkpoint-specific system prompt suffix (model-specific, not generic).
  - LLM system prompt includes checkpoint name and its known token conventions.

### US-PRE-E01-S02: Standalone and pipeline modes
- priority_weight: 96 · estimate: 3–6 d · phase: mvp
- As an integrator, I want to run as a CLI tool or as a pipeline stage after SSA, so that flexible deployment.
- Acceptance criteria:
  - The same core library backs both modes.
  - Pipeline mode consumes SSA scene metadata when present; metadata is injected into the LLM context.

### US-PRE-E01-S03: Automatic full parameterization
- priority_weight: 95 · estimate: 2–4 d · phase: rich
- As an operator, I want one input to yield camera/lighting/color/composition/aspect suggestions without manual per-field entry, so that fast authoring.
- Acceptance criteria:
  - LLM system prompt instructs the model to always fill camera, lighting, color, composition fields even when not specified by the user.
  - Defaults cover ≥ 80 % of fields on the internal fixture set.
  - Fields not inferable from text get sensible scene-appropriate defaults (not null).

### US-PRE-E01-S04: Semantic fidelity with hidden detail
- priority_weight: 94 · estimate: 1–2 d · phase: rich
- As a user, I want details I did not explicitly list but that are needed for rendering to still appear when required, so that images look complete, not sparse.
- Acceptance criteria:
  - Spot checks on sample prompts show added detail without contradicting the user intent.
  - System prompt instructs LLM: "Infer and add visually necessary detail not stated by the user, but never contradict the user's stated scene."

### US-PRE-E01-S05: Fidelity over stylistic consistency on conflict
- priority_weight: 93 · estimate: 1–2 d · phase: mvp
- As an art director, I want the engine to prefer literal adherence to the user text when conflicts appear, so that story accuracy wins.
- Acceptance criteria:
  - Conflict-resolution policy stated explicitly in the LLM system prompt: "When user text conflicts with a style preset, the user text is authoritative."
  - Covered by unit tests with contradictory input fixtures.

---

## Epic E02 — Inputs

English-only; single prompt per run; optional SSA metadata; user style/LoRA/ControlNet/constraints.

### US-PRE-E02-S01: From one sentence to rich paragraphs
- priority_weight: 92 · estimate: 2–4 d · phase: mvp
- As an author, I want minimal text to expand into scene-ready descriptions (subject, environment, time of day, emotion, motivation), so that writers are not forced to write SD keywords.
- Acceptance criteria:
  - Single-sentence input produces full camera/lighting/composition fields in output.
  - Expansion quality evaluated on curated fixtures.

### US-PRE-E02-S02: Optional Scene Analyzer metadata
- priority_weight: 91 · estimate: 2–4 d · phase: mvp
- As an integrator, I want characters/environment/mood fields from SSA when present to be injected into the LLM context, so that pipeline mode is richer than standalone.
- Acceptance criteria:
  - SSA fields merged into LLM user message as structured context block.
  - Absent fields fall back cleanly (LLM infers from text alone).
  - SSA fields cannot override explicit user text (fidelity rule enforced in system prompt).

### US-PRE-E02-S03: User style, LoRA, ControlNet selections
- priority_weight: 90 · estimate: 2–4 d · phase: mvp
- As an artist, I want declarative controls on top of text, so that house styles apply.
- Acceptance criteria:
  - Predefined presets only; preset tokens appended to LLM context.
  - Invalid combinations resolved per the documented priority policy.

---

## Epic E03 — Structured JSON output

Positive/negative prompts; camera/composition/lighting/color tags; LoRA strengths; ControlNet tags; aspect ratio; seed strategy; seed value.

### US-PRE-E03-S01: Full tag bundles
- priority_weight: 95 · estimate: 2–5 d · phase: mvp
- As a TD, I want all tag families emitted in one JSON for the image generator, so that there is a single consumer contract.
- Acceptance criteria:
  - Golden JSON fixtures; strict schema validation (Pydantic).
  - `seed` field present in all outputs (hash-derived or user-provided).

### US-PRE-E03-S02: Silent best-fidelity resolution (no warning UX)
- priority_weight: 94 · estimate: 2–5 d · phase: mvp
- As a PM, I want conflicts resolved internally without nagging popups, so that batch automation stays quiet.
- Acceptance criteria:
  - Logs capture every resolution decision for audit; no warnings emitted to stdout/stderr.

---

## Epic E04 — Style & aesthetics

Predefined presets only; palettes, lighting, materials, character identity.

### US-PRE-E04-S01: Preset-driven style system
- priority_weight: 83 · estimate: 2–3 d · phase: mvp
- As a brand owner, I want cinematic/anime/dark-fantasy presets with enforcement options, so that consistent series look.
- Acceptance criteria:
  - Preset catalog versioned; predefined-only (no custom presets in v2).
  - Preset tokens injected into LLM system prompt when active.

### US-PRE-E04-S02: Conflict resolution for incompatible styles
- priority_weight: 82 · estimate: 2–3 d · phase: mvp
- As an engineer, I want anime-vs-photoreal conflicts resolved with explicit priority rules, so that no garbage outputs.
- Acceptance criteria:
  - Priority rules encoded in system prompt: user text > user controls > SSA > preset > engine default.
  - Unit tests for the conflict resolver with contradictory preset + text fixtures.

---

## Epic E05 — Camera & composition

Lens, shot type, angle, DoF, composition rules, cinematic/portrait framing.

### US-PRE-E05-S01: Complete camera grammar
- priority_weight: 80 · estimate: 1–3 d · phase: mvp
- As a DP, I want lens mm, shot scale, angle, DoF, and composition rules when applicable, so that shots are intentional.
- Acceptance criteria:
  - LLM system prompt instructs model to always fill `camera.shot`, `camera.angle`, `composition`; `camera.lens_mm` and `camera.dof` when scene type suggests them.
  - Rendered prompts include camera tokens when relevant; camera object never all-null for a valid scene.

---

## Epic E06 — Negative prompt strategy

Scene/style negatives; artifact bans; deduplication.

### US-PRE-E06-S01: Layered negatives
- priority_weight: 79 · estimate: 1–2 d · phase: mvp
- As QA, I want scene + style + artifact negatives merged without contradiction, so that clean renders.
- Acceptance criteria:
  - LLM generates scene-specific and style-specific negatives.
  - Post-LLM deduplication step removes duplicates before output.
  - Artifact ban list (extra limbs, blurry, watermark…) always appended regardless of LLM output.

---

## Epic E07 — LoRA & ControlNet

Auto + manual selection; fusion limits; compatibility checks.

### US-PRE-E07-S01: LoRA fusion up to N adapters
- priority_weight: 85 · estimate: 2–4 d · phase: p1
- As an artist, I want multiple LoRAs with strengths and conflict detection, so that complex styles possible.
- Acceptance criteria:
  - N ≤ 3 default, configurable; conflict detection between two character LoRAs.
  - LLM may suggest LoRA from predefined catalog; user override takes priority.

### US-PRE-E07-S02: ControlNet lifecycle
- priority_weight: 84 · estimate: 2–4 d · phase: p1
- As a TD, I want ControlNet auto when helpful, stripped when redundant, multi-CN supported, and style-compatibility checked, so that no broken CN stacks.
- Acceptance criteria:
  - Compatibility-matrix tests; redundant ControlNet removed in post-LLM validation step.
  - LLM instructed: "Only include ControlNet when the scene clearly benefits; omit when uncertain."

---

## Epic E08 — Seeds & aspect

Canonical seed derivation; per-scene variation; aspect from prompt; reproducibility for recurring characters.

### US-PRE-E08-S01: Seed and aspect policies
- priority_weight: 82 · estimate: 1–2 d · phase: mvp
- As a pipeline engineer, I want automatic seed derivation from input and per-scene aspect suggestions, so that continuity and variety controls without manual seed entry.
- Acceptance criteria:
  - Canonical seed derived per FR-PRE-017; echoed in output `"seed"` field.
  - Aspect ratio inferred from scene type by LLM (landscape → 16:9, portrait → 2:3, etc.).
  - `seed_strategy` set to `consistent_character` when SSA provides a stable character id, else `random`.

### US-PRE-E08-S02: Consume character/identity registry
- priority_weight: 81 · estimate: 1–3 d · phase: p1
- As a pipeline engineer, I want PRE to read the job's character registry (character → stable seed/LoRA), so that recurring characters stay consistent across scenes and into IMG.
- Acceptance criteria:
  - When a registry is provided, recurring characters resolve to their registered seed/LoRA; clean fallback when absent.
  - Character seed used as LLM seed override for that scene.

---

## Epic E09 — Performance & operations

Deterministic via seed; offline via local LLM; Windows-first; model-specific; CLI + API; logging.

### US-PRE-E09-S01: Deterministic via seed
- priority_weight: 76 · estimate: 1–3 d · phase: mvp
- As an SRE, I want the same JSON in + same seed to produce the same JSON out (temperature=0), so that tests are reliable and runs are reproducible.
- Acceptance criteria:
  - Hash tests on canonical JSON with fixed seed across two runs.
  - Test explicitly sets seed to a known constant to isolate from hash derivation.

### US-PRE-E09-S02: Offline via local LLM
- priority_weight: 75 · estimate: 1–3 d · phase: mvp
- As an SRE, I want no cloud calls — LLM runs on-device, so that air-gapped operation.
- Acceptance criteria:
  - Network-off tests pass (firewall all external IPs).
  - `llama-cpp-python` backend only; no OpenAI/Anthropic/HuggingFace inference API calls.

### US-PRE-E09-S03: Model-specific outputs
- priority_weight: 74 · estimate: 1–3 d · phase: mvp
- As an SRE, I want prompts tuned to the declared checkpoint via checkpoint-specific system prompt sections, so that quality.
- Acceptance criteria:
  - Checkpoint name required in input; checkpoint's token conventions injected into LLM system prompt.
  - `checkpoints.json` maps checkpoint id → prompt style notes for the LLM.

### US-PRE-E09-S04: CLI (JSON) + Python API
- priority_weight: 73 · estimate: 1–3 d · phase: mvp
- As an SRE, I want automation and embedding, so that flexible integration.
- Acceptance criteria:
  - CLI accepts a JSON input file or inline JSON string.
  - Python API documented with examples; both share one core library.

### US-PRE-E09-S05: Decision logging (seed included)
- priority_weight: 72 · estimate: 1–3 d · phase: mvp
- As an SRE, I want structured logs for refinement decisions including the seed used, so that debuggability and reproducibility.
- Acceptance criteria:
  - Log schema documented; seed recorded in every log entry.
  - Log includes: input hash, derived seed, model name, inference time, schema validation result.

### US-PRE-E09-S06: Latency targets
- priority_weight: 71 · estimate: 1–3 d · phase: rich
- As an SRE, I want acceptable latency for interactive use, so that rapid iteration.
- Acceptance criteria:
  - P95 ≤ 3 s on GPU (RTX 3060 or equivalent, Q4_K_M model).
  - P95 ≤ 15 s on CPU-only (8-core modern CPU, Q4_K_M model).
  - Latency measured end-to-end (input parse → validated JSON out).

### US-PRE-E09-S07: Windows-first
- priority_weight: 70 · estimate: 1–3 d · phase: mvp
- As an SRE, I want primary Windows support including GPU acceleration via CUDA, so that dev-team focus on target platform.
- Acceptance criteria:
  - CI runs on Windows; `llama-cpp-python` built with CUDA support on CI.
  - CPU fallback tested on CI without GPU.

---

## Epic E10 — Validation & errors

Malformed prompts; LLM output validation; availability checks; model compatibility.

### US-PRE-E10-S01: Robust validation layer
- priority_weight: 78 · estimate: 1–2 d · phase: mvp
- As QA, I want to repair or reject bad inputs, validate LLM JSON output, verify LoRA/ControlNet availability, and validate model compatibility, so that safe failures.
- Acceptance criteria:
  - Error catalog aligned with SRS v2: `INVALID_INPUT`, `SCHEMA_VALIDATION_FAILED`, `RESOURCE_NOT_FOUND`, `MODEL_INCOMPATIBLE`, `MODEL_NOT_FOUND`, `LLM_OUTPUT_INVALID`.
  - Malformed user input: repaired where possible, else `INVALID_INPUT`.
  - LLM returns invalid JSON: retry once with explicit repair prompt; if still invalid → `LLM_OUTPUT_INVALID`.
  - Pydantic validation runs on every LLM output before it reaches the caller.
