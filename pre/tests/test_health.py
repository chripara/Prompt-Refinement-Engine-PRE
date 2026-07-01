"""US-PRE-E00-S01 — GET /health exposes model name, quantization, load time."""
import os

import pytest
from fastapi.testclient import TestClient

from pre.core.llm import loader


@pytest.mark.skipif(
    not os.environ.get("PRE_TEST_MODEL_PATH"),
    reason="Set PRE_TEST_MODEL_PATH to a local GGUF file to run the real-model integration test.",
)
def test_health_reports_loaded_model(monkeypatch):
    monkeypatch.setenv(loader.MODEL_PATH_ENV, os.environ["PRE_TEST_MODEL_PATH"])

    from pre.api.service import create_app

    with TestClient(create_app()) as client:
        resp = client.get("/health")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_name"]
    assert body["quantization"] != "unknown"
    assert body["load_time_s"] > 0
