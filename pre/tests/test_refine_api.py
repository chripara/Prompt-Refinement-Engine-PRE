"""US-PRE-E00-S03 — POST /refine end-to-end via the real local LLM.

Skipped unless PRE_TEST_MODEL_PATH points at a real GGUF file.
"""
import os

import pytest
from fastapi.testclient import TestClient

from pre.core.llm import loader


@pytest.mark.skipif(
    not os.environ.get("PRE_TEST_MODEL_PATH"),
    reason="Set PRE_TEST_MODEL_PATH to a local GGUF file to run the real-model integration test.",
)
def test_refine_endpoint_returns_valid_output(monkeypatch):
    monkeypatch.setenv(loader.MODEL_PATH_ENV, os.environ["PRE_TEST_MODEL_PATH"])

    from pre.api.service import create_app

    with TestClient(create_app()) as client:
        resp = client.post(
            "/refine",
            json={"text": "A dragon soaring over mountains at dawn", "checkpoint": "sdxl-base-1.0"},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["positive_prompt"]
    assert body["checkpoint"] == "sdxl-base-1.0"
    assert body["seed"] is not None
