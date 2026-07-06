"""Local GGUF model loading (US-PRE-E00-S01, FR-PRE-018).

Loads exactly once per process and is reused across all requests. The model
path comes from the `PRE_MODEL_PATH` environment variable; startup fails fast
with `MODEL_NOT_FOUND` if the variable is unset or the file does not exist.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pre.core.errors import PREError

MODEL_PATH_ENV = "PRE_MODEL_PATH"
GPU_LAYERS_ENV = "PRE_GPU_LAYERS"
N_CTX_ENV = "PRE_N_CTX"
_DEFAULT_N_CTX = 4096

# ggml_ftype enum -> human-readable quantization label (llama.cpp gguf.py).
_FTYPE_LABELS = {
    "0": "F32",
    "1": "F16",
    "2": "Q4_0",
    "3": "Q4_1",
    "7": "Q8_0",
    "8": "Q5_0",
    "9": "Q5_1",
    "10": "Q2_K",
    "11": "Q3_K_S",
    "12": "Q3_K_M",
    "13": "Q3_K_L",
    "14": "Q4_K_S",
    "15": "Q4_K_M",
    "16": "Q5_K_S",
    "17": "Q5_K_M",
    "18": "Q6_K",
}


@dataclass(frozen=True)
class ModelInfo:
    name: str
    path: str
    quantization: str
    load_time_s: float
    n_gpu_layers: int


_llm: Any = None
_model_info: ModelInfo | None = None


def is_loaded() -> bool:
    return _llm is not None


def get_model_info() -> ModelInfo:
    if _model_info is None:
        raise PREError("MODEL_NOT_FOUND", "LLM model has not been loaded yet.")
    return _model_info


def get_llm() -> Any:
    if _llm is None:
        raise PREError("MODEL_NOT_FOUND", "LLM model has not been loaded yet.")
    return _llm


def _resolve_quantization(metadata: dict[str, str], filename: str) -> str:
    ftype = metadata.get("general.file_type")
    if ftype in _FTYPE_LABELS:
        return _FTYPE_LABELS[ftype]
    stem = filename.upper()
    for label in _FTYPE_LABELS.values():
        if label in stem:
            return label
    return "unknown"


def load_model(model_path: str | None = None, n_gpu_layers: int | None = None) -> Any:
    """Load the GGUF model once; subsequent calls return the cached instance.

    Raises:
        PREError(MODEL_NOT_FOUND): env var unset, or file missing.
    """
    global _llm, _model_info

    if _llm is not None:
        return _llm

    path = model_path or os.environ.get(MODEL_PATH_ENV)
    if not path:
        raise PREError(
            "MODEL_NOT_FOUND",
            f"Environment variable {MODEL_PATH_ENV} is not set.",
        )

    model_file = Path(path)
    if not model_file.is_file():
        raise PREError(
            "MODEL_NOT_FOUND",
            f"Model file not found at '{path}'.",
            details={"path": str(model_file)},
        )

    if n_gpu_layers is None:
        env_value = os.environ.get(GPU_LAYERS_ENV)
        n_gpu_layers = int(env_value) if env_value is not None else -1

    n_ctx_env = os.environ.get(N_CTX_ENV)
    n_ctx = int(n_ctx_env) if n_ctx_env is not None else _DEFAULT_N_CTX

    from llama_cpp import Llama  # imported lazily: keeps import-time light for tests

    start = time.perf_counter()
    llm = Llama(
        model_path=str(model_file),
        n_gpu_layers=n_gpu_layers,
        n_ctx=n_ctx,
        seed=0,
        verbose=False,
    )
    load_time_s = round(time.perf_counter() - start, 3)

    metadata = getattr(llm, "metadata", {}) or {}

    _llm = llm
    _model_info = ModelInfo(
        name=metadata.get("general.name", model_file.stem),
        path=str(model_file),
        quantization=_resolve_quantization(metadata, model_file.name),
        load_time_s=load_time_s,
        n_gpu_layers=n_gpu_layers,
    )
    return _llm


def reset() -> None:
    """Test-only: drop the cached model so `load_model` can run again."""
    global _llm, _model_info
    _llm = None
    _model_info = None
