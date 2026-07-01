"""FastAPI service — wraps the PRE LLM runtime as a REST API."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from pre.core.errors import PREError
from pre.core.llm import loader
from pre.core.version import VERSION


@asynccontextmanager
async def _lifespan(app: FastAPI):
    # Fail fast on startup: MODEL_NOT_FOUND surfaces immediately, not on first request.
    loader.load_model()
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="Prompt Refinement Engine", version=VERSION, lifespan=_lifespan)

    @app.get("/health")
    def health():
        try:
            info = loader.get_model_info()
        except PREError as exc:
            raise HTTPException(status_code=503, detail=exc.to_dict())
        return {
            "status": "ok",
            "version": VERSION,
            "model_name": info.name,
            "quantization": info.quantization,
            "load_time_s": info.load_time_s,
        }

    return app
