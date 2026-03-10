"""
/explain API route for ArchitectAI.

POST /explain
    Input : { "architecture": <Architecture JSON>, "diagram_path": "<optional path>" }
    Output: { "explanation": "<plain-English explanation>" }

If *diagram_path* points to an existing PNG the diagram is encoded by
ConvNeXt-Tiny and the 768-dim embedding is projected to condition Qwen's
explanation.  Falls back to text-only if the file is missing or encoding fails.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.api.schemas.architecture import Architecture
from backend.core.vision.encoder import encode_diagram
from backend.core.vlm.explainer import generate_explanation

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Explanation"])


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class ExplainRequest(BaseModel):
    """Request body for POST /explain."""

    architecture: Architecture
    diagram_path: Optional[str] = None  # absolute or relative path to the PNG


class ExplainResponse(BaseModel):
    """Response from POST /explain."""

    explanation: str


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------


@router.post(
    "/explain",
    response_model=ExplainResponse,
    summary="Explain architecture diagram",
)
async def explain_endpoint(req: ExplainRequest) -> ExplainResponse:
    """Generate a plain-English explanation of *architecture*.

    1. Encode the diagram PNG with ConvNeXt-Tiny (if path exists).
    2. Project the 768-dim vision embedding with VisionProjector.
    3. Include embedding statistics in the Qwen prompt.
    4. Return the generated explanation text.
    """
    import torch  # local import keeps startup fast when GPU is absent

    # ── 1. Vision encoding (optional) ─────────────────────────────────────────
    vision_features: Optional[torch.Tensor] = None
    if req.diagram_path:
        img_path = Path(req.diagram_path)
        if img_path.exists():
            try:
                vision_features = encode_diagram(str(img_path))
                logger.info(
                    "Diagram encoded: %s  (features shape %s)",
                    img_path.name,
                    tuple(vision_features.shape),
                )
            except Exception as exc:
                logger.warning(
                    "Vision encoding failed (%s) — proceeding without image features.",
                    exc,
                )
        else:
            logger.warning(
                "diagram_path does not exist: %s — proceeding text-only.",
                img_path,
            )

    # ── 2. Generate explanation ────────────────────────────────────────────────
    try:
        explanation = generate_explanation(req.architecture, vision_features)
    except Exception as exc:
        logger.exception("Explanation generation failed.")
        raise HTTPException(
            status_code=500, detail=f"Explanation generation failed: {exc}"
        ) from exc

    return ExplainResponse(explanation=explanation)
