# Prompt Refinement Engine (PRE) — v2

Local-LLM prompt expansion for image-generation pipelines. See [`SRS_PRE_v2.md`](SRS_PRE_v2.md)
for the full spec and [`EPICS_AND_STORIES_v2.md`](EPICS_AND_STORIES_v2.md) for the backlog.

**Deterministic · Offline · Local LLM (GGUF via `llama-cpp-python`) · No cloud calls**

---

## Status

Rebuilt from scratch against the v2 (local-LLM) architecture. Only the LLM runtime
foundation is implemented so far:

| Story | Title | Status |
|-------|-------|--------|
| US-PRE-E00-S01 | Local LLM model loading | done |
| US-PRE-E00-S02 | Canonical seed derivation | pending |
| US-PRE-E00-S03 | LLM inference contract | pending |
| US-PRE-E00-S04 | GPU / CPU runtime support | pending |
| US-PRE-E01-S01 | Hyper-detailed SDXL-oriented prompts | pending |

---

## Setup

```cmd
cd "C:\Projects\Prompt Renifinement Engine (PRE) with Sonet"
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

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

`requirements.txt` installs the CUDA-enabled wheel from abetlen's index
(`--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124`), which needs
the CUDA 12.x **runtime** DLLs (`cudart64_12.dll`, `cublas64_12.dll`, ...) on `PATH` —
it does **not** need `nvcc`/MSVC. If you have Ollama installed, it already ships these:

```cmd
set PATH=%LOCALAPPDATA%\Programs\Ollama\lib\ollama\cuda_v12;%PATH%
```

Without a compatible GPU/runtime, set `PRE_GPU_LAYERS=0` to force CPU-only inference.

### Run the API

```cmd
.venv\Scripts\uvicorn pre.api.service:create_app --factory --port 8000
```

`GET /health` returns `{"status", "version", "model_name", "quantization", "load_time_s"}`
once the model has loaded; the model loads once at process startup and is reused for
every request.

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
      loader.py       # US-PRE-E00-S01 — GGUF loading, singleton reuse, MODEL_NOT_FOUND
    errors.py         # PREError + error code catalog (SRS §9)
    version.py
  api/
    service.py        # FastAPI app factory, GET /health
  tests/
    test_llm_loader.py
    test_health.py
```
