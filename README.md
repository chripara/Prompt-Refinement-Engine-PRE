# Prompt Refinement Engine (PRE)

**Local-LLM prompt expansion for image-generation pipelines** — turns a short scene
description into a hyper-detailed, model-specific, deterministic Stable Diffusion / SDXL
prompt (positive + negative prompt, camera/composition/lighting/color tags, LoRA,
ControlNet, aspect ratio, seed) as strict JSON, entirely offline via a local GGUF model
(Mistral/Llama-class, via [`llama-cpp-python`](https://github.com/abetlen/llama-cpp-python)) —
no OpenAI/cloud calls. Python 3.11, FastAPI REST API, Windows-first with GPU (CUDA) or
CPU-only auto-detection.

🚧 **Actively in development** — built story-by-story against
[`EPICS_AND_STORIES_v2.md`](EPICS_AND_STORIES_v2.md) with one PR per story (see the commit
history / PR descriptions for the engineering decisions behind each one, not just the diff).

See [`SRS_PRE_v2.md`](SRS_PRE_v2.md) for the full spec.

**Deterministic · Offline · Local LLM (GGUF) · No cloud calls · Prompt engineering for Stable Diffusion / SDXL**

---

## Status

| Epic | Stories done | Status |
|------|---------------|--------|
| E00 — LLM Runtime (model loading, seed determinism, inference contract, GPU/CPU auto-detect, temperature) | 5 / 5 | ✅ complete |
| E01 — Role in the pipeline (checkpoint-specific expansion, standalone/pipeline modes, fidelity) | 1 / 5 | 🚧 in progress |
| E02 – E10 (inputs, JSON output, style, camera, negatives, LoRA/ControlNet, seeds, performance, validation) | 0 / N | ⏳ not started |

Full story-level detail (including in-review PRs) in
[`EPICS_AND_STORIES_v2.md`](EPICS_AND_STORIES_v2.md).

---

## Development methodology

This project is built by **product-owner-directed AI-agent orchestration**, not
freehand vibe-coding — that's a deliberate part of what it demonstrates, alongside the
code itself:

1. **I** write the spec: [`SRS_PRE_v2.md`](SRS_PRE_v2.md) (FR/NFR, use cases) and
   [`EPICS_AND_STORIES_v2.md`](EPICS_AND_STORIES_v2.md) (epics, user stories, acceptance
   criteria, priority/estimate per story).
2. Claude Code implements **one story per branch, one PR per story** — never more than
   one story's worth of scope in a single PR.
3. **I** review every PR, test the running API manually (Swagger/Insomnia, real local
   LLM), and decide product/architecture tradeoffs when they come up mid-implementation —
   e.g. choosing CPU-only vs. GPU when they first appeared to trade off speed against
   bit-exact determinism (later resolved in the same PR once the actual root cause —
   a KV-cache reset bug, not GPU nondeterminism — was found and fixed), or deciding that
   seed should drive creative variation via a configurable sampling temperature rather
   than staying inert at `temperature=0`.
4. Commit messages carry a `Summary` / `Changes` / `Tests` body so every PR is reviewable
   against what it actually changed and how it was verified — not just a diff.

Only I merge PRs into `main`; the agent proposes, I decide.

---

## Quick start

```cmd
cd "C:\Projects\Prompt Renifinement Engine (PRE) with Sonet"
python run.py
```

First run auto-creates `.venv` and installs the base dependencies, then detects whether
this machine has a *usable* CUDA runtime (an NVIDIA GPU present **and** the CUDA 12.x
runtime DLLs locatable — e.g. via a local Ollama install) and installs the matching
`llama-cpp-python` wheel: CUDA-enabled if so, plain CPU-only PyPI wheel otherwise
(US-PRE-E00-S04). This matters because a CUDA-linked `llama.dll` fails to even *import*
without those runtime DLLs on `PATH`, regardless of `n_gpu_layers` — picking the wrong
wheel breaks the app rather than just running slower, so `run.py` picks correctly up
front instead of assuming a GPU is present.

It also auto-adds Ollama's bundled CUDA runtime DLLs to `PATH` at startup and
auto-resolves `PRE_MODEL_PATH` from a local Ollama model (default `mistral:latest`)
if you haven't set it yourself. Then starts the API at `http://localhost:8000`.

Override before running if needed: `PRE_MODEL_PATH` (explicit GGUF path), `PRE_OLLAMA_MODEL`
(which Ollama model to resolve), `PRE_GPU_LAYERS` (`-1` = all layers on GPU, `0` = force
CPU-only; auto-detected the same way as the install-time wheel choice if unset).

## Manual setup

### Model

Point `PRE_MODEL_PATH` at a local GGUF file (a quantized 7B-class instruct model,
e.g. Mistral-7B-Instruct, Q4_K_M). If you already pulled a model through **Ollama**,
you can point directly at its blob — no need to duplicate the 4+ GB file:

```cmd
ollama show mistral:latest --modelfile
:: prints "FROM C:\Users\<you>\.ollama\models\blobs\sha256-<hash>" — that file *is* the GGUF.

set PRE_MODEL_PATH=C:\Users\<you>\.ollama\models\blobs\sha256-<hash>
```

Startup fails fast with `MODEL_NOT_FOUND` if the variable is unset or the file is missing.

### GPU (CUDA) without installing the full CUDA Toolkit

`llama-cpp-python` is intentionally left out of `requirements.txt` — `run.py` installs it
separately after checking, at install time, whether this machine has a usable CUDA
runtime (`nvidia-smi` succeeds **and** `cudart64_12.dll` is locatable via `CUDA_PATH`, a
local Ollama install, or `PATH`). If so it installs the CUDA-enabled wheel from abetlen's
index (`--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124`), which
needs the CUDA 12.x **runtime** DLLs (`cudart64_12.dll`, `cublas64_12.dll`, ...) on `PATH`
at import time — it does **not** need `nvcc`/MSVC. If you have Ollama installed, it
already ships these, and `run.py` adds that directory to `PATH` automatically:

```cmd
set PATH=%LOCALAPPDATA%\Programs\Ollama\lib\ollama\cuda_v12;%PATH%
```

If no usable CUDA runtime is detected, `run.py` installs the plain CPU-only wheel instead
— the app still runs, just slower, rather than failing to import `llama_cpp` entirely.
Installing manually? Pick one:

```cmd
:: CPU-only
.venv\Scripts\pip install llama-cpp-python

:: CUDA
.venv\Scripts\pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
```

Set `PRE_GPU_LAYERS=0` to force CPU-only inference even with a CUDA-enabled wheel installed.

### Run the API

```cmd
.venv\Scripts\uvicorn pre.api.service:create_app --factory --port 8000
```

`GET /health` returns `{"status", "version", "model_name", "quantization", "load_time_s",
"backend"}` ("gpu" or "cpu") once the model has loaded; the model loads once at process
startup and is reused for every request. `POST /refine` runs the full contract (US-PRE-E00-S03):
deterministic seed, grammar-constrained JSON output, one repair attempt on invalid JSON.

---

## Tests

```cmd
.venv\Scripts\python -m pytest pre/tests/ -v
```

Tests that need a real model file are skipped unless `PRE_TEST_MODEL_PATH` is set:

```cmd
set PRE_TEST_MODEL_PATH=C:\Users\<you>\.ollama\models\blobs\sha256-<hash>
```

---

## Project structure

```
pre/
  core/
    llm/
      loader.py        # US-PRE-E00-S01/S04 — GGUF loading, GPU/CPU auto-detect, MODEL_NOT_FOUND
      inference.py      # US-PRE-E00-S03 — temperature=0 + seed contract, grammar-constrained JSON
    determinism/
      canonical.py      # US-PRE-E00-S02 — canonical seed derivation
    models/
      input.py           # RefineInput (SRS §2)
      output.py          # RefineOutput / LLMGeneratedFields (SRS §7)
    logging/
      decision_log.py   # structured JSONL decision log (FR-PRE-014)
    pipeline.py          # refine() — ties seed derivation to inference
    errors.py            # PREError + error code catalog (SRS §9)
    version.py
  api/
    service.py           # FastAPI app factory, GET /health, POST /refine, dark /docs
  tests/
```
