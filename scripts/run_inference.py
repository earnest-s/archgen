#!/usr/bin/env python3
import json
import os
import time
from pathlib import Path

import torch
from backend.core.inference import generate_explanation


def load_samples(dataset_path: Path, limit: int = 5):
    samples = []
    with dataset_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if "architecture" in row and "explanation" in row:
                samples.append(row)
            if len(samples) >= limit:
                break
    return samples


def main() -> None:
    os.environ.setdefault("HF_HOME", "./.cache/huggingface")

    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required for inference")

    dataset_path = Path("data/synthetic/dataset.jsonl")

    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing dataset: {dataset_path}")

    samples = load_samples(dataset_path, limit=5)
    if not samples:
        raise RuntimeError("No valid samples found in dataset")

    for idx, sample in enumerate(samples, start=1):
        architecture = sample["architecture"]
        ground_truth = sample["explanation"]

        start = time.perf_counter()
        generated = generate_explanation(architecture)
        elapsed = time.perf_counter() - start

        mem_mb = torch.cuda.memory_allocated() / (1024 ** 2)

        print("----------------------------------------")
        print(f"Sample {idx}")
        print("Architecture")
        print(json.dumps(architecture, indent=2, ensure_ascii=True))
        print("Generated Explanation")
        print(generated)
        print("Ground Truth")
        print(ground_truth)
        print(f"Generation Time (s): {elapsed:.3f}")
        print(f"GPU Memory Allocated (MB): {mem_mb:.2f}")

    print("----------------------------------------")
    print("Inference completed")


if __name__ == "__main__":
    main()
