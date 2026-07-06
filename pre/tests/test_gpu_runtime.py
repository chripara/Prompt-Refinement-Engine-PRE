"""US-PRE-E00-S04 — GPU/CPU runtime support."""
import json

import llama_cpp as _llama_cpp
import pytest

from pre.core.llm import loader


# ── n_gpu_layers resolution ────────────────────────────────────────────────

def test_resolve_n_gpu_layers_env_override_wins(monkeypatch):
    monkeypatch.setenv(loader.GPU_LAYERS_ENV, "7")
    monkeypatch.setattr(loader, "_gpu_available", lambda: False)
    assert loader._resolve_n_gpu_layers(None) == 7


def test_resolve_n_gpu_layers_auto_detects_gpu(monkeypatch):
    monkeypatch.delenv(loader.GPU_LAYERS_ENV, raising=False)
    monkeypatch.setattr(loader, "_gpu_available", lambda: True)
    assert loader._resolve_n_gpu_layers(None) == -1


def test_resolve_n_gpu_layers_auto_detects_cpu_only(monkeypatch):
    monkeypatch.delenv(loader.GPU_LAYERS_ENV, raising=False)
    monkeypatch.setattr(loader, "_gpu_available", lambda: False)
    assert loader._resolve_n_gpu_layers(None) == 0


def test_resolve_n_gpu_layers_explicit_arg_wins_over_env(monkeypatch):
    monkeypatch.setenv(loader.GPU_LAYERS_ENV, "7")
    assert loader._resolve_n_gpu_layers(3) == 3


# ── backend labeling ────────────────────────────────────────────────────────

def test_resolve_backend_labels():
    assert loader._resolve_backend(0) == "cpu"
    assert loader._resolve_backend(-1) == "gpu"
    assert loader._resolve_backend(20) == "gpu"


# ── GPU detection: build support + actual device presence ─────────────────

def test_gpu_available_false_when_build_lacks_support(monkeypatch):
    monkeypatch.setattr(_llama_cpp, "llama_supports_gpu_offload", lambda: False)
    assert loader._gpu_available() is False


def test_gpu_available_false_when_nvidia_smi_missing(monkeypatch):
    monkeypatch.setattr(_llama_cpp, "llama_supports_gpu_offload", lambda: True)

    def _raise(*args, **kwargs):
        raise FileNotFoundError()

    monkeypatch.setattr(loader.subprocess, "run", _raise)
    assert loader._gpu_available() is False


def test_gpu_available_true_when_supported_and_device_present(monkeypatch):
    monkeypatch.setattr(_llama_cpp, "llama_supports_gpu_offload", lambda: True)

    class _Result:
        returncode = 0
        stdout = "GPU 0: NVIDIA RTX 5070 Ti"

    monkeypatch.setattr(loader.subprocess, "run", lambda *a, **kw: _Result())
    assert loader._gpu_available() is True


# ── end-to-end: load_model() wires backend + logs the decision ────────────

def test_load_model_records_backend_and_decision_log(monkeypatch, tmp_path):
    model_file = tmp_path / "model.gguf"
    model_file.write_bytes(b"fake gguf bytes")
    monkeypatch.setenv(loader.MODEL_PATH_ENV, str(model_file))
    monkeypatch.setenv(loader.GPU_LAYERS_ENV, "0")  # force CPU path; no detection needed

    class _FakeLlama:
        def __init__(self, **kwargs):
            self.metadata = {"general.name": "fake-model", "general.file_type": "15"}

    monkeypatch.setattr(_llama_cpp, "Llama", _FakeLlama)

    loader.load_model()

    info = loader.get_model_info()
    assert info.backend == "cpu"
    assert info.n_gpu_layers == 0
    assert info.quantization == "Q4_K_M"

    from pre.core.logging import decision_log
    import os as _os

    log_lines = [
        json.loads(line)
        for line in open(_os.environ[decision_log.LOG_PATH_ENV], encoding="utf-8")
    ]
    model_load_entries = [e for e in log_lines if e["stage"] == "model_load"]
    assert len(model_load_entries) == 1
    assert model_load_entries[0]["backend"] == "cpu"
    assert model_load_entries[0]["n_gpu_layers"] == 0
