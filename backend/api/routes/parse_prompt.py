"""
POST /parse-prompt

Converts a natural-language prompt into a validated Architecture JSON object
using the deterministic rule-based prompt parser.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status

from backend.api.schemas.response import ParsePromptResponse, PromptRequest
from backend.core.prompt_parser.parser import parse_prompt

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/parse-prompt", tags=["Parsing"])


@router.post(
    "",
    response_model=ParsePromptResponse,
    summary="Parse a natural-language prompt into an Architecture",
    status_code=status.HTTP_200_OK,
)
async def parse_prompt_route(body: PromptRequest) -> ParsePromptResponse:
    """Accept a natural-language *prompt* and return the parsed Architecture.

    The parsing is fully deterministic (rule-based, no ML).

    - **prompt**: plain-English description of the desired system architecture.

    Returns a JSON-serialised :class:`~backend.api.schemas.architecture.Architecture`
    that can be passed directly to ``POST /generate-diagram``.
    """
    try:
        arch = parse_prompt(body.prompt)
        logger.info("Parsed prompt → %d nodes, %d edges", len(arch.nodes), len(arch.edges))
        return ParsePromptResponse(architecture=arch)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error in parse_prompt_route")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal parsing error: {exc}",
        ) from exc
