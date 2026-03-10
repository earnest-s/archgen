"""
Image preprocessing for the ArchitectAI vision encoder.

Applies the standard ImageNet normalisation pipeline used by ConvNeXt models
from ``timm``.  All transforms are pure-torch/PIL so no additional dependencies
beyond ``Pillow`` and ``torchvision`` are required.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import torch
from PIL import Image
from torchvision import transforms

# ---------------------------------------------------------------------------
# ImageNet statistics (used by timm ConvNeXt checkpoints)
# ---------------------------------------------------------------------------

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# Default input resolution for ConvNeXt-Tiny.
IMAGE_SIZE: int = 224

# ---------------------------------------------------------------------------
# Transform pipeline
# ---------------------------------------------------------------------------

_transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)


def preprocess_image(
    image: Union[str, Path, Image.Image],
    *,
    device: Union[str, torch.device] = "cpu",
) -> torch.Tensor:
    """Load, resize, and normalise *image* for ConvNeXt inference.

    Args:
        image:  Path to a PNG/JPEG file **or** an already-loaded
                :class:`~PIL.Image.Image` instance.
        device: Target torch device. The returned tensor is placed there.

    Returns:
        Float32 tensor of shape ``(1, 3, 224, 224)`` on *device*.

    Raises:
        FileNotFoundError: If *image* is a path and the file does not exist.
        ValueError:        If the image cannot be opened as RGB.
    """
    if isinstance(image, (str, Path)):
        path = Path(image)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")
        pil = Image.open(path).convert("RGB")
    elif isinstance(image, Image.Image):
        pil = image.convert("RGB")
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    tensor: torch.Tensor = _transform(pil)   # (3, H, W)
    return tensor.unsqueeze(0).to(device)     # (1, 3, H, W)


def batch_preprocess(
    images: list[Union[str, Path, Image.Image]],
    device: Union[str, torch.device] = "cpu",
) -> torch.Tensor:
    """Preprocess a list of images into a single batched tensor.

    Args:
        images: List of file paths or PIL images.
        device: Target torch device.

    Returns:
        Float32 tensor of shape ``(N, 3, 224, 224)`` on *device*.
    """
    tensors = [preprocess_image(img, device=device).squeeze(0) for img in images]
    return torch.stack(tensors, dim=0)
