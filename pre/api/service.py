"""FastAPI service — wraps the PRE LLM runtime as a REST API."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse

from pre.core.errors import PREError
from pre.core.llm import loader
from pre.core.models.input import RefineInput
from pre.core.pipeline import refine
from pre.core.version import VERSION

# Swagger UI ships no dark theme out of the box; this overrides just the
# color scheme so /docs isn't a stark white page.
_SWAGGER_DARK_CSS = """
<style>
  body { background-color: #1e1e1e; }
  .swagger-ui, .swagger-ui .info .title, .swagger-ui .opblock-tag,
  .swagger-ui .opblock .opblock-summary-operation-id,
  .swagger-ui .opblock .opblock-summary-path,
  .swagger-ui .opblock .opblock-summary-description,
  .swagger-ui table thead tr th, .swagger-ui .parameter__name,
  .swagger-ui .response-col_status, .swagger-ui .response-col_description,
  .swagger-ui .model-title, .swagger-ui .model, .swagger-ui label,
  .swagger-ui .tab li, .swagger-ui .opblock-description-wrapper p {
    color: #e0e0e0 !important;
  }
  .swagger-ui .topbar { background-color: #111; }
  .swagger-ui .scheme-container { background-color: #252525; }
  .swagger-ui .opblock { background: #262626; border-color: #444; }
  .swagger-ui .opblock .opblock-section-header { background: #2d2d2d; }
  .swagger-ui .opblock-body pre, .swagger-ui .highlight-code {
    background: #111 !important; color: #ddd !important;
  }
  .swagger-ui select, .swagger-ui input[type=text] {
    background: #2d2d2d; color: #e0e0e0; border-color: #555;
  }
  .swagger-ui .btn { color: #e0e0e0; }
</style>
"""


@asynccontextmanager
async def _lifespan(app: FastAPI):
    # Fail fast on startup: MODEL_NOT_FOUND surfaces immediately, not on first request.
    loader.load_model()
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Prompt Refinement Engine",
        version=VERSION,
        lifespan=_lifespan,
        docs_url=None,  # replaced below with a dark-themed variant
    )

    @app.get("/docs", include_in_schema=False)
    async def custom_swagger_ui() -> HTMLResponse:
        html = get_swagger_ui_html(
            openapi_url=app.openapi_url,
            title=f"{app.title} - Docs",
        ).body.decode()
        return HTMLResponse(html.replace("</head>", _SWAGGER_DARK_CSS + "</head>"))

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
            "backend": info.backend,
        }

    @app.post("/refine")
    def refine_endpoint(req: RefineInput):
        try:
            result = refine(req.model_dump())
        except PREError as exc:
            raise HTTPException(status_code=422, detail=exc.to_dict())
        return result.output.model_dump()

    return app
