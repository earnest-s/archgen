"""
Image preprocessing for the ArchitectAI vision encoder.

Provides two transform pipelines:

* **Eval / inference** — deterministic resize + ImageNet normalisation.
* **Train** — same base transforms plus data augmentation for diversity:
  random rotation (±5°), colour jitter (brightness/contrast), slight
  Gaussian blur, and random crop + resize.  Augmentations keep diagrams
  legible while increasing visual variety for the vision encoder.

All transforms are pure-torch/PIL; no additional dependencies beyond
``Pillow`` and ``torchvision`` are required.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Union

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
# Transform pipelines
# ---------------------------------------------------------------------------

# Inference / evaluation — deterministic, no augmentation.
_eval_transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)

# Training — adds visual diversity while keeping diagrams readable:
#   • RandomRotation ±5° — slight tilt variation
#   • ColorJitter      — mild brightness/contrast shift
#   • GaussianBlur     — very light smoothing noise (kernel 3, σ 0.1-0.5)
#   • RandomResizedCrop — zoom/crop variation (scale 0.85-1.0)
_train_transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE + 16, IMAGE_SIZE + 16), antialias=True),
        transforms.RandomRotation(degrees=5, fill=255),           # white fill
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
        transforms.RandomResizedCrop(
            size=IMAGE_SIZE,
            scale=(0.85, 1.0),
            ratio=(0.95, 1.05),
            antialias=True,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)

# Backwards-compatible alias keeps existing call sites working.
_transform = _eval_transform


def get_train_transform() -> transforms.Compose:
    """Return the augmented training transform pipeline.

    Returns:
        A :class:`torchvision.transforms.Compose` instance with augmentation.
    """
    return _train_transform


def get_eval_transform() -> transforms.Compose:
    """Return the deterministic inference/evaluation transform pipeline.

    Returns:
        A :class:`torchvision.transforms.Compose` instance without augmentation.
    """
    return _eval_transform


def preprocess_image(
    image: Union[str, Path, Image.Image],
    *,
    device: Union[str, torch.device] = "cpu",
    mode: Literal["eval", "train"] = "eval",
) -> torch.Tensor:
    """Load, resize, and normalise *image* for ConvNeXt inference or training.

    Args:
        image:  Path to a PNG/JPEG file **or** an already-loaded
                :class:`~PIL.Image.Image` instance.
        device: Target torch device. The returned tensor is placed there.
        mode:   ``"eval"`` (default) uses the deterministic inference pipeline.
                ``"train"`` applies augmentation (rotation, jitter, blur, crop).

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

    pipeline = _train_transform if mode == "train" else _eval_transform
    tensor: torch.Tensor = pipeline(pil)  # (3, H, W)
    return tensor.unsqueeze(0).to(device)  # (1, 3, H, W)


def batch_preprocess(
    images: list[Union[str, Path, Image.Image]],
    device: Union[str, torch.device] = "cpu",
    mode: Literal["eval", "train"] = "eval",
) -> torch.Tensor:
    """Preprocess a list of images into a single batched tensor.

    Args:
        images: List of file paths or PIL images.
        device: Target torch device.
        mode:   ``"eval"`` for deterministic inference, ``"train"`` for augmented.

    Returns:
        Float32 tensor of shape ``(N, 3, 224, 224)`` on *device*.
    """
    tensors = [preprocess_image(img, device=device, mode=mode).squeeze(0) for img in images]
    return torch.stack(tensors, dim=0)
