"""
Vision encoder for the ArchitectAI pipeline.

Extracts a dense feature vector from an architecture diagram PNG using a
pretrained ConvNeXt-Tiny backbone.  The resulting embedding is later
projected into the language model's embedding space by the VLM projector.

Public API::

    from backend.core.vision.encoder import encode_diagram

    embedding = encode_diagram("path/to/diagram.png")
    # → torch.Tensor of shape (768,)
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn

from backend.core.vision.convnext_loader import load_convnext
from backend.core.vision.preprocess import batch_preprocess, preprocess_image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Singleton model cache (loaded once per process)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _get_model(device_str: str) -> nn.Module:
    """Return the cached ConvNeXt-Tiny model for *device_str*."""
    return load_convnext(pretrained=True, device=torch.device(device_str))


def _default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def encode_diagram(
    image_path: Union[str, Path],
    *,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Extract a 768-dimensional feature vector from a diagram PNG.

    The image is preprocessed (resize → normalise) and forwarded through a
    headless ConvNeXt-Tiny.  The model is loaded (with pretrained weights) on
    the first call and cached for subsequent calls in the same process.

    Args:
        image_path: Path to the diagram PNG file.
        device:     Torch device to run inference on.  Defaults to CUDA if
                    available, otherwise CPU.

    Returns:
        1-D float32 tensor of shape ``(768,)`` on CPU.

    Raises:
        FileNotFoundError: If *image_path* does not exist.
        RuntimeError:      If model inference fails.

    Example::

        embedding = encode_diagram("/tmp/arch_react_fastapi.png")
        print(embedding.shape)   # torch.Size([768])
    """
    if device is None:
        device = _default_device()

    model = _get_model(str(device))

    tensor = preprocess_image(image_path, device=device)  # (1, 3, 224, 224)

    with torch.inference_mode():
        features: torch.Tensor = model(tensor)  # (1, 768)

    embedding = features.squeeze(0).cpu()  # (768,)
    logger.debug(
        "Encoded diagram %s → shape %s, norm %.4f",
        image_path,
        tuple(embedding.shape),
        embedding.norm().item(),
    )
    return embedding


def encode_diagrams_batch(
    image_paths: list[Union[str, Path]],
    *,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Batch-encode multiple diagram images in a single forward pass.

    Args:
        image_paths: List of paths to diagram PNG files.
        device:      Torch device for inference.

    Returns:
        Float32 tensor of shape ``(N, 768)`` on CPU.
    """
    if device is None:
        device = _default_device()

    model = _get_model(str(device))
    batch = batch_preprocess(image_paths, device=device)  # (N, 3, 224, 224)

    with torch.inference_mode():
        features: torch.Tensor = model(batch)  # (N, 768)

    return features.cpu()
