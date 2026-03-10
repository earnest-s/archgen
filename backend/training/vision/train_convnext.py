"""
ConvNeXt-Tiny fine-tuning on architecture diagram images.

This script fine-tunes a pretrained ConvNeXt-Tiny to predict
the set of NodeType labels present in a diagram (multi-label
binary cross-entropy classification).

Training data is read from the synthetic dataset produced by
``scripts/generate_dataset.py``.

Usage::

    python -m backend.training.vision.train_convnext \
        --data  data/synthetic \
        --out   checkpoints/convnext \
        --epochs 20 \
        --bs 32 \
        --lr 3e-4
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)

# NodeType label order (must match backend.api.schemas.architecture.NodeType).
NODE_TYPES: List[str] = [
    "Frontend", "Backend", "Service", "Database", "Cache", "Queue", "External"
]
NUM_LABELS = len(NODE_TYPES)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class DiagramDataset(Dataset):
    """PyTorch dataset over the synthetic JSONL manifest.

    Each sample returns ``(image_tensor, label_vector)`` where
    ``label_vector`` is a float32 multi-hot vector of length 7.
    """

    def __init__(self, manifest_path: Path) -> None:
        from backend.core.vision.preprocess import preprocess_image  # lazy import

        self.preprocess = preprocess_image
        self.samples: List[dict] = []

        with manifest_path.open(encoding="utf-8") as fh:
            for line in fh:
                entry = json.loads(line.strip())
                if entry.get("image") and Path(entry["image"]).exists():
                    self.samples.append(entry)

        logger.info("DiagramDataset: %d valid samples", len(self.samples))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        entry = self.samples[idx]

        # Image
        img = self.preprocess(entry["image"]).squeeze(0)  # (3, 224, 224)

        # Multi-hot label
        arch_types = {
            n["type"] for n in entry["architecture"]["nodes"]
        }
        label = torch.zeros(NUM_LABELS, dtype=torch.float32)
        for i, t in enumerate(NODE_TYPES):
            if t in arch_types:
                label[i] = 1.0

        return img, label


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(
    data_dir: Path,
    out_dir: Path,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 3e-4,
    val_split: float = 0.1,
) -> None:
    """Fine-tune ConvNeXt-Tiny for multi-label NodeType classification.

    Args:
        data_dir:   Root directory containing ``dataset.jsonl``.
        out_dir:    Directory to save model checkpoints.
        epochs:     Number of training epochs.
        batch_size: Mini-batch size.
        lr:         Initial learning rate (AdamW).
        val_split:  Fraction of data reserved for validation.
    """
    from backend.core.vision.convnext_loader import CONVNEXT_TINY_FEATURES, load_convnext

    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on %s", device)

    # ── Dataset & DataLoader ─────────────────────────────────────────────────
    manifest = data_dir / "dataset.jsonl"
    if not manifest.exists():
        raise FileNotFoundError(
            f"JSONL manifest not found: {manifest}. "
            "Run scripts/generate_dataset.py first."
        )

    full_ds = DiagramDataset(manifest)
    val_size  = max(1, int(len(full_ds) * val_split))
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2)

    # ── Model ────────────────────────────────────────────────────────────────
    # Load backbone with features only (num_classes=0).
    backbone = load_convnext(pretrained=True, device=device)
    # Add classification head for multi-label task.
    model = nn.Sequential(
        backbone,
        nn.Linear(CONVNEXT_TINY_FEATURES, NUM_LABELS),
    ).to(device)

    # ── Optimiser & loss ─────────────────────────────────────────────────────
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss()

    # ── Training ─────────────────────────────────────────────────────────────
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimiser.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimiser.step()
            total_loss += loss.item() * imgs.size(0)

        avg_train = total_loss / train_size

        # ── Validation ───────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.inference_mode():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                val_loss += criterion(logits, labels).item() * imgs.size(0)
        avg_val = val_loss / val_size

        scheduler.step()

        logger.info(
            "Epoch %3d / %d | train_loss=%.4f | val_loss=%.4f",
            epoch, epochs, avg_train, avg_val,
        )

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            ckpt_path = out_dir / "best_convnext.pth"
            torch.save(model.state_dict(), ckpt_path)
            logger.info("  ✓ New best model saved → %s", ckpt_path)

    logger.info("Training complete. Best val loss: %.4f", best_val_loss)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune ConvNeXt for ArchitectAI.")
    p.add_argument("--data",   default="data/synthetic",     help="Synthetic data root dir.")
    p.add_argument("--out",    default="checkpoints/convnext", help="Checkpoint output dir.")
    p.add_argument("--epochs", type=int,   default=20,   help="Training epochs.")
    p.add_argument("--bs",     type=int,   default=32,   help="Batch size.")
    p.add_argument("--lr",     type=float, default=3e-4, help="Learning rate.")
    p.add_argument("--val",    type=float, default=0.1,  help="Validation split fraction.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(
        data_dir=Path(args.data),
        out_dir=Path(args.out),
        epochs=args.epochs,
        batch_size=args.bs,
        lr=args.lr,
        val_split=args.val,
    )
