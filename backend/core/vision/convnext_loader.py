"""
ConvNeXt-Tiny model loader for the ArchitectAI vision encoder.

Loads a pretrained ConvNeXt-Tiny from ``timm`` and removes the classification
head so the model acts as a pure feature extractor returning a 768-dimensional
embedding vector.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Feature dimension output by ConvNeXt-Tiny before the classification head.
CONVNEXT_TINY_FEATURES: int = 768

# timm model identifier for ConvNeXt-Tiny with ImageNet-1k pretrained weights.
CONVNEXT_TIMM_ID: str = "convnext_tiny"


def load_convnext(
    *,
    pretrained: bool = True,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """Load ConvNeXt-Tiny as a headless feature extractor.

    The classification ``head`` (``nn.Linear(768, 1000)``) is replaced with
    ``nn.Identity()`` so the model's forward pass returns the 768-dim feature
    vector directly.

    Args:
        pretrained: When ``True`` (default), download ImageNet-1k weights from
                    the ``timm`` model hub on first call.
        device:     Target device.  If ``None``, selects CUDA if available,
                    otherwise CPU.

    Returns:
        An ``nn.Module`` whose ``forward(x)`` accepts a ``(B, 3, 224, 224)``
        tensor and returns a ``(B, 768)`` feature tensor.

    Raises:
        ImportError: If ``timm`` is not installed.
        RuntimeError: If the model weights cannot be downloaded.
    """
    try:
        import timm  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "timm is required for the vision encoder. "
            "Install it with: pip install timm"
        ) from exc

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(
        "Loading ConvNeXt-Tiny (pretrained=%s) on %s", pretrained, device
    )

    # num_classes=0 tells timm to return features instead of logits.
    model: nn.Module = timm.create_model(
        CONVNEXT_TIMM_ID,
        pretrained=pretrained,
        num_classes=0,       # remove classification head
        global_pool="avg",   # global average pooling → (B, 768)
    )
    model = model.to(device)
    model.eval()

    logger.info(
        "ConvNeXt-Tiny loaded — output features: %d", CONVNEXT_TINY_FEATURES
    )
    return model
