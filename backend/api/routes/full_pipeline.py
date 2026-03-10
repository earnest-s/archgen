"""
POST /generate

Full pipeline: prompt → Architecture → PNG diagram in a single request.

This is the primary endpoint consumed by the frontend.
"""

from __future__ import annotations

import base64
import logging
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, status

from backend.api.schemas.response import FullPipelineResponse, PromptRequest
from backend.core.diagram.generator import generate_diagram
from backend.core.prompt_parser.parser import parse_prompt

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/generate", tags=["Pipeline"])

_OUTPUT_DIR = Path(os.getenv("DIAGRAM_OUTPUT_DIR", "/tmp/architectai_diagrams"))
_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@router.post(
    "",
    response_model=FullPipelineResponse,
    summary="Full pipeline: prompt → Architecture + PNG diagram",
    status_code=status.HTTP_200_OK,
)
async def full_pipeline_route(
    body: PromptRequest,
    inline: bool = Query(
        default=True,
        description="Include base64-encoded PNG in the response body.",
    ),
) -> FullPipelineResponse:
    """Run the complete ArchitectAI pipeline for a given *prompt*.

    Steps (all server-side, no ML at this stage):

    1. Parse *prompt* → :class:`~backend.api.schemas.architecture.Architecture`
    2. Render Architecture → PNG (via Graphviz + ``diagrams``)
    3. Return architecture JSON + diagram path + optional base64 image

    - **prompt**: plain-English system description.
    - **inline**: include base64 PNG in response (default: ``true``).
    """
    # ── 1. Parse ─────────────────────────────────────────────────────────────
    try:
        arch = parse_prompt(body.prompt)
        logger.info(
            "Pipeline parsed prompt → %d nodes, %d edges",
            len(arch.nodes), len(arch.edges),
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Prompt parsing failed: {exc}",
        ) from exc

    # ── 2. Generate diagram ───────────────────────────────────────────────────
    try:
        name_hint = "_".join(n.id for n in arch.nodes[:4])
        output_base = str(_OUTPUT_DIR / f"pipeline_{name_hint}")
        png_path = generate_diagram(arch, output_path=output_base)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Diagram generation failed: {exc}",
        ) from exc

    # ── 3. Build response ─────────────────────────────────────────────────────
    b64: str | None = None
    if inline:
        with open(png_path, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode()

    return FullPipelineResponse(
        architecture=arch,
        diagram_path=png_path,
        diagram_base64=b64,
    )
