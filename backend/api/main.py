"""
ArchitectAI FastAPI application entry point.

Registers all route modules and configures CORS so the React frontend
(running on a different port during development) can communicate freely.

Run with::

    uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes.generate_diagram import router as generate_diagram_router
from backend.api.routes.full_pipeline import router as full_pipeline_router
from backend.api.routes.parse_prompt import router as parse_prompt_router

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App instance
# ---------------------------------------------------------------------------

app = FastAPI(
    title="ArchitectAI",
    description=(
        "Generates software architecture diagrams from natural-language prompts "
        "and explains them using deep learning models."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---------------------------------------------------------------------------
# CORS – allow all origins in dev; lock down via env in production.
# ---------------------------------------------------------------------------

_ALLOWED_ORIGINS: list[str] = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:5173,http://localhost:3000,http://localhost:8080",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(parse_prompt_router)
app.include_router(generate_diagram_router)
app.include_router(full_pipeline_router)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health", tags=["System"], summary="Health check")
async def health() -> dict[str, str]:
    """Return service status. Used by Docker health checks and load balancers."""
    return {"status": "ok", "service": "ArchitectAI"}


logger.info("ArchitectAI API started — docs at http://localhost:8000/docs")
