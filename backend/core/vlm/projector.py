"""
Vision-to-language projector for ArchitectAI.

Maps the 768-dimensional ConvNeXt-Tiny feature vector into the token embedding
space of Qwen2.5-3B (hidden size 2048) so the LLM can "see" the diagram.

The projector is a lightweight two-layer MLP with a GELU activation:

    768 → 1536 → 2048

Weights are randomly initialised until fine-tuned on paired
(diagram, explanation) data.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ConvNeXt-Tiny output dimension.
VISION_DIM: int = 768
# Qwen2.5-3B hidden / embedding dimension.
LM_DIM: int = 2048
# Intermediate MLP dimension.
HIDDEN_DIM: int = 1536


class VisionProjector(nn.Module):
    """Two-layer MLP that projects vision features into the LM embedding space.

    Architecture::

        Linear(768 → 1536) → GELU → LayerNorm(1536) → Linear(1536 → 2048)

    Args:
        vision_dim: Input dimension (ConvNeXt-Tiny output).  Default: 768.
        lm_dim:     Output dimension (Qwen hidden size).  Default: 2048.
        hidden_dim: Intermediate MLP dimension.  Default: 1536.
    """

    def __init__(
        self,
        vision_dim: int = VISION_DIM,
        lm_dim: int = LM_DIM,
        hidden_dim: int = HIDDEN_DIM,
    ) -> None:
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, lm_dim),
        )
        logger.debug(
            "VisionProjector: %d → %d → %d", vision_dim, hidden_dim, lm_dim
        )

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """Project *vision_features* into the language model embedding space.

        Args:
            vision_features: Float tensor of shape ``(B, vision_dim)`` or
                             ``(vision_dim,)`` (single sample).

        Returns:
            Float tensor of shape ``(B, lm_dim)`` or ``(lm_dim,)``,
            matching the input batch dimension.
        """
        return self.projector(vision_features)


def project_features(
    features: torch.Tensor,
    projector: Optional[VisionProjector] = None,
) -> torch.Tensor:
    """Convenience function: project vision features with a default projector.

    If no *projector* is supplied a new (untrained) :class:`VisionProjector`
    is created.  In production, pass the fine-tuned projector instance loaded
    from a checkpoint.

    Args:
        features:   Vision embedding tensor ``(B, 768)`` or ``(768,)``.
        projector:  Optional pre-loaded :class:`VisionProjector`.

    Returns:
        Projected tensor on the same device as *features*.
    """
    if projector is None:
        logger.warning(
            "project_features called without a trained projector — "
            "using randomly-initialised weights."
        )
        projector = VisionProjector().to(features.device)

    projector = projector.to(features.device)
    with torch.inference_mode():
        return projector(features)
