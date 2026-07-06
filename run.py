"""run.py — bootstrap + launch the Prompt Refinement Engine API.

Usage (from CMD, no activation needed):
    cd "C:\\Projects\\Prompt Renifinement Engine (PRE) with Sonet"
    python run.py

First run: creates .venv, installs the base dependencies, then detects
whether this machine has a usable CUDA runtime (an NVIDIA GPU present AND
the CUDA 12.x runtime DLLs locatable — e.g. via a local Ollama install) and
installs the matching llama-cpp-python wheel: CUDA-enabled if so, plain
CPU-only PyPI wheel otherwise. This matters because a CUDA-linked llama.dll
fails to even *import* without those runtime DLLs on PATH, regardless of
the n_gpu_layers setting — so picking the wrong wheel breaks the app
entirely rather than just running slower.
Subsequent runs: boots straight into the venv.

Before starting the server this also:
  - adds Ollama's bundled CUDA 12 runtime DLLs to PATH, if present, so a
    CUDA-enabled install can load without a full CUDA Toolkit;
  - auto-resolves PRE_MODEL_PATH from a local Ollama model (default
    "mistral:latest") if PRE_MODEL_PATH is not already set.

Override via environment variables before running, if needed:
    PRE_MODEL_PATH      explicit path to a GGUF file (skips Ollama lookup)
    PRE_OLLAMA_MODEL    which Ollama model to resolve (default: mistral:latest)
    PRE_GPU_LAYERS      forwarded to llama-cpp-python (-1 = all layers on GPU,
                        0 = CPU-only); auto-detected if unset (US-PRE-E00-S04)

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

ROOT = Path(__file__).parent.resolve()
VENV = ROOT / ".venv"
REQUIREMENTS = ROOT / "requirements.txt"

VENV_PYTHON = (
    VENV / "Scripts" / "python.exe"
    if sys.platform == "win32"
    else VENV / "bin" / "python"
)

LLAMA_CPP_CUDA_INDEX = "https://abetlen.github.io/llama-cpp-python/whl/cu124"
_CUDART_DLL_NAME = "cudart64_12.dll"


def _in_venv() -> bool:
    """True if we're already running inside any virtual environment."""
    return (
        sys.prefix != sys.base_prefix
        or os.environ.get("VIRTUAL_ENV") is not None
    )


def _find_ollama_cuda_dir() -> Path | None:
    candidates = [
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "lib" / "ollama" / "cuda_v12",
        Path("C:/Program Files/Ollama/lib/ollama/cuda_v12"),
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


def _cudart_locatable() -> Path | None:
    """Find a directory containing the CUDA 12 runtime DLL, checking
    CUDA_PATH, a local Ollama install, and the current PATH — without
    requiring the full CUDA Toolkit to be installed.
    """
    candidates: list[Path] = []
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        candidates.append(Path(cuda_path) / "bin")
    ollama_dir = _find_ollama_cuda_dir()
    if ollama_dir:
        candidates.append(ollama_dir)
    candidates += [Path(p) for p in os.environ.get("PATH", "").split(os.pathsep) if p]

    for directory in candidates:
        if (directory / _CUDART_DLL_NAME).is_file():
            return directory
    return None


def _detect_cuda_runtime_available() -> bool:
    """True only if an NVIDIA GPU is present AND the CUDA runtime DLLs are
    locatable somewhere — both are required for a CUDA-linked llama.dll to
    even load, let alone benefit from GPU offload.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True, timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    if result.returncode != 0 or not result.stdout.strip():
        return False
    return _cudart_locatable() is not None


def _install_llama_cpp_python(pip: str) -> None:
    if _detect_cuda_runtime_available():
        print("[PRE] CUDA runtime detected — installing GPU-enabled llama-cpp-python ...")
        subprocess.check_call([
            pip, "-m", "pip", "install", "--quiet",
            "llama-cpp-python", "--extra-index-url", LLAMA_CPP_CUDA_INDEX,
        ])
    else:
        print("[PRE] No usable CUDA runtime detected — installing CPU-only llama-cpp-python ...")
        subprocess.check_call([pip, "-m", "pip", "install", "--quiet", "llama-cpp-python"])


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
        subprocess.check_call([pip, "-m", "pip", "install", "--quiet", "-r", str(REQUIREMENTS)])
    else:
        subprocess.check_call([
            pip, "-m", "pip", "install", "--quiet",
            "pydantic>=2,<3", "fastapi>=0.110", "uvicorn[standard]>=0.29",
        ])
    _install_llama_cpp_python(pip)
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
              "default install path). If llama-cpp-python was installed CPU-only, "
              "this is expected and fine.")

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
