"""
Request/response schemas for the FastAPI routes (diagram-specific).

General architecture schemas live in
:mod:`backend.api.schemas.architecture`.  This module adds thin
request/response wrappers that are specific to the HTTP transport layer.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from backend.api.schemas.architecture import Architecture


class PromptRequest(BaseModel):
    """Input body for the ``POST /parse-prompt`` and ``POST /generate`` endpoints."""

    prompt: str = Field(
        ...,
        min_length=1,
        description="Natural-language description of the desired architecture.",
        examples=["React frontend, FastAPI backend, PostgreSQL database"],
    )


class ParsePromptResponse(BaseModel):
    """Response body for ``POST /parse-prompt``."""

    architecture: Architecture = Field(
        ...,
        description="Parsed architecture derived from the natural-language prompt.",
    )


class GenerateDiagramRequest(BaseModel):
    """Input body for the ``POST /generate-diagram`` endpoint."""

    architecture: Architecture = Field(
        ...,
        description="Validated architecture to render as a PNG diagram.",
    )


class GenerateDiagramResponse(BaseModel):
    """Response body for ``POST /generate-diagram``."""

    diagram_path: str = Field(
        ...,
        description="Absolute path of the generated PNG file on the server.",
    )
    diagram_base64: Optional[str] = Field(
        default=None,
        description=(
            "Base64-encoded PNG image data (included when the client "
            "requests inline delivery via the ``inline=true`` query param)."
        ),
    )


class FullPipelineResponse(BaseModel):
    """Response body for ``POST /generate`` (full pipeline)."""

    architecture: Architecture = Field(
        ...,
        description="Architecture parsed from the prompt.",
    )
    diagram_path: str = Field(
        ...,
        description="Server-side path of the generated PNG.",
    )
    diagram_base64: Optional[str] = Field(
        default=None,
        description="Base64-encoded PNG for inline display.",
    )
