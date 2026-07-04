"""run.py — bootstrap + launch the Prompt Refinement Engine API.

Usage (from CMD, no activation needed):
    cd "C:\\Projects\\Prompt Renifinement Engine (PRE) with Sonet"
    python run.py

First run: creates .venv and installs all dependencies (including the
CUDA-enabled llama-cpp-python wheel) automatically.
Subsequent runs: boots straight into the venv.

Before starting the server this also:
  - adds Ollama's bundled CUDA 12 runtime DLLs to PATH, if Ollama is
    installed, so the CUDA wheel can load without a full CUDA Toolkit;
  - auto-resolves PRE_MODEL_PATH from a local Ollama model (default
    "mistral:latest") if PRE_MODEL_PATH is not already set.

Override via environment variables before running, if needed:
    PRE_MODEL_PATH      explicit path to a GGUF file (skips Ollama lookup)
    PRE_OLLAMA_MODEL    which Ollama model to resolve (default: mistral:latest)
    PRE_GPU_LAYERS      forwarded to llama-cpp-python (-1 = all layers on GPU)

Starts:
  FastAPI REST API  ->  http://localhost:8000
                        http://localhost:8000/docs  (OpenAPI docs)
                        http://localhost:8000/health

Stop with Ctrl+C.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap: ensure we are running inside the project venv.
# If not, create it (once), install deps, and re-exec via venv Python.
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent.resolve()
VENV = ROOT / ".venv"
REQUIREMENTS = ROOT / "requirements.txt"

VENV_PYTHON = (
    VENV / "Scripts" / "python.exe"
    if sys.platform == "win32"
    else VENV / "bin" / "python"
)

LLAMA_CPP_CUDA_INDEX = "https://abetlen.github.io/llama-cpp-python/whl/cu124"


def _in_venv() -> bool:
    """True if we're already running inside any virtual environment."""
    return (
        sys.prefix != sys.base_prefix
        or os.environ.get("VIRTUAL_ENV") is not None
    )


def _bootstrap() -> None:
    """Create venv, install deps, re-exec this script inside the venv."""
    if not VENV_PYTHON.exists():
        print("[PRE] Creating virtual environment (.venv) ...")
        subprocess.check_call([sys.executable, "-m", "venv", str(VENV)])
        print("[PRE] Virtual environment created.")

    print("[PRE] Installing dependencies (this takes a moment on first run) ...")
    pip = str(VENV_PYTHON)
    subprocess.check_call(
        [pip, "-m", "pip", "install", "--quiet", "--upgrade", "pip"]
    )
    if REQUIREMENTS.exists():
        subprocess.check_call([
            pip, "-m", "pip", "install", "--quiet",
            "-r", str(REQUIREMENTS),
            "--extra-index-url", LLAMA_CPP_CUDA_INDEX,
        ])
    else:
        subprocess.check_call([
            pip, "-m", "pip", "install", "--quiet",
            "pydantic>=2,<3", "fastapi>=0.110", "uvicorn[standard]>=0.29",
            "llama-cpp-python", "--extra-index-url", LLAMA_CPP_CUDA_INDEX,
        ])
    print("[PRE] Dependencies ready.")

    # subprocess + sys.exit instead of os.execv — execv splits on spaces on Windows.
    print("[PRE] Restarting inside virtual environment ...\n")
    result = subprocess.run(
        [str(VENV_PYTHON), str(Path(__file__).resolve())] + sys.argv[1:]
    )
    sys.exit(result.returncode)


if not _in_venv():
    _bootstrap()

# ---------------------------------------------------------------------------
# Main — only reached when running inside the venv
# ---------------------------------------------------------------------------

import uvicorn  # noqa: E402


def _find_ollama_cuda_dir() -> Path | None:
    candidates = [
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "lib" / "ollama" / "cuda_v12",
        Path("C:/Program Files/Ollama/lib/ollama/cuda_v12"),
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


def _resolve_model_path_from_ollama(model_name: str) -> str | None:
    try:
        result = subprocess.run(
            ["ollama", "show", model_name, "--modelfile"],
            capture_output=True, text=True, timeout=15,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        if line.strip().startswith("FROM "):
            return line.strip()[len("FROM "):].strip()
    return None


def _configure_environment() -> None:
    cuda_dir = _find_ollama_cuda_dir()
    if cuda_dir:
        os.environ["PATH"] = f"{cuda_dir}{os.pathsep}{os.environ.get('PATH', '')}"
        print(f"[PRE] CUDA runtime DLLs found via Ollama: {cuda_dir}")
    else:
        print("[PRE] No bundled CUDA runtime found (Ollama not detected on the "
              "default install path) — GPU load may fail unless the CUDA Toolkit "
              "is on PATH, or set PRE_GPU_LAYERS=0 to force CPU-only.")

    if not os.environ.get("PRE_MODEL_PATH"):
        model_name = os.environ.get("PRE_OLLAMA_MODEL", "mistral:latest")
        resolved = _resolve_model_path_from_ollama(model_name)
        if resolved:
            os.environ["PRE_MODEL_PATH"] = resolved
            print(f"[PRE] PRE_MODEL_PATH auto-resolved from Ollama model '{model_name}':")
            print(f"      {resolved}")
        else:
            print(
                "[PRE] PRE_MODEL_PATH is not set and could not be auto-resolved "
                f"from Ollama model '{model_name}'.\n"
                "      Set it explicitly, e.g.:\n"
                "      set PRE_MODEL_PATH=C:\\path\\to\\model.gguf"
            )


API_HOST = "0.0.0.0"
API_PORT = 8000


if __name__ == "__main__":
    print("=" * 60)
    print("  Prompt Refinement Engine")
    print("=" * 60)

    _configure_environment()

    from pre.api.service import create_app

    print(f"  API  ->  http://localhost:{API_PORT}")
    print(f"  Docs ->  http://localhost:{API_PORT}/docs")
    print(f"  Health -> http://localhost:{API_PORT}/health")
    print("  Stop with Ctrl+C")
    print("=" * 60)

    uvicorn.run(create_app(), host=API_HOST, port=API_PORT, log_level="warning")
