#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import List, Tuple

import timm
import torch
import torch.nn as nn
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms


class DiagramDataset(Dataset):
    def __init__(self, pairs: List[Tuple[Path, int]], transform: transforms.Compose):
        self.pairs = pairs
        self.transform = transform

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        image_path, label = self.pairs[index]
        image = Image.open(image_path).convert("RGB")
        return self.transform(image), torch.tensor(label, dtype=torch.long)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or export ConvNeXt checkpoint.")
    parser.add_argument("--dataset", default="data/synthetic/dataset.jsonl")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--output", default="checkpoints/convnext_best.pt")
    return parser.parse_args()


def load_image_pairs(dataset_path: Path) -> Tuple[List[Tuple[Path, int]], int]:
    if not dataset_path.exists():
        return [], 0

    pattern_to_id = {}
    pairs: List[Tuple[Path, int]] = []

    with dataset_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            arch = row.get("architecture", {})
            pattern = arch.get("pattern", "unknown")
            image_path = row.get("image_path")

            if image_path is None:
                continue

            image_file = Path(image_path)
            if not image_file.exists():
                continue

            if pattern not in pattern_to_id:
                pattern_to_id[pattern] = len(pattern_to_id)

            pairs.append((image_file, pattern_to_id[pattern]))

    return pairs, len(pattern_to_id)


def export_pretrained(model: nn.Module, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "num_classes": 0, "mode": "pretrained"}, output_path)
    print(f"No image dataset found. Saved pretrained ConvNeXt weights to {output_path}")


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset)
    output_path = Path(args.output)

    model = timm.create_model("convnext_tiny", pretrained=True)
    pairs, num_classes = load_image_pairs(dataset_path)

    if not pairs or num_classes < 2:
        export_pretrained(model, output_path)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.reset_classifier(num_classes)
    model.to(device)

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    dataset = DiagramDataset(pairs, transform)
    train_len = max(1, int(0.8 * len(dataset)))
    val_len = len(dataset) - train_len
    train_ds, val_ds = random_split(dataset, [train_len, val_len])

    bs = min(args.batch_size, 16)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=3e-4)

    best_loss = float("inf")
    stale = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        total_val = 0.0
        count = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                logits = model(images)
                loss = criterion(logits, labels)
                total_val += float(loss.item())
                count += 1

        val_loss = total_val / max(1, count)
        print(f"epoch={epoch + 1} val_loss={val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            stale = 0
            torch.save(
                {"state_dict": model.state_dict(), "num_classes": num_classes, "mode": "finetuned"},
                output_path,
            )
        else:
            stale += 1
            if stale >= args.patience:
                print("Early stopping triggered")
                break

    print(f"Saved best ConvNeXt checkpoint to {output_path}")


if __name__ == "__main__":
    main()
