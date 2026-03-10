"""
POST /generate-diagram

Accepts a validated Architecture JSON, renders it using the diagrams library,
and returns the PNG path and optionally a base64-encoded image.
"""

from __future__ import annotations

import base64
import logging
import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, status

from backend.api.schemas.response import GenerateDiagramRequest, GenerateDiagramResponse
from backend.core.diagram.generator import generate_diagram

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/generate-diagram", tags=["Diagram"])

# Directory where diagrams are persisted between requests.
_OUTPUT_DIR = Path(os.getenv("DIAGRAM_OUTPUT_DIR", "/tmp/architectai_diagrams"))
_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@router.post(
    "",
    response_model=GenerateDiagramResponse,
    summary="Render an Architecture as a PNG diagram",
    status_code=status.HTTP_200_OK,
)
async def generate_diagram_route(
    body: GenerateDiagramRequest,
    inline: bool = Query(
        default=True,
        description="Include base64-encoded PNG in the response body.",
    ),
) -> GenerateDiagramResponse:
    """Render the supplied *architecture* as a PNG using Graphviz + diagrams.

    - **architecture**: validated Architecture JSON (from ``/parse-prompt`` or
      user-edited React Flow export).
    - **inline**: when ``true`` (default) the response includes a base64 PNG
      string suitable for ``<img src='data:image/png;base64,...'>`` rendering.

    Returns the server-side path and optional inline image data.
    """
    try:
        # Derive a stable filename from node ids for de-duplication.
        name_hint = "_".join(n.id for n in body.architecture.nodes[:4])
        output_base = str(_OUTPUT_DIR / f"arch_{name_hint}")
        png_path = generate_diagram(body.architecture, output_path=output_base)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Diagram generation failed: {exc}",
        ) from exc

    b64: str | None = None
    if inline:
        with open(png_path, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode()

    logger.info("Diagram generated: %s", png_path)
    return GenerateDiagramResponse(diagram_path=png_path, diagram_base64=b64)
