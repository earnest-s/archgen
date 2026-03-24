"""
Inference pipeline for ArchitectAI.

Loads trained ConvNeXt + Qwen LoRA models and generates explanations for
architecture diagrams using the combined vision + LLM pipeline.

Runs on the first N samples from the dataset and prints:
- Sample ID
- Architecture JSON
- Vision embedding shape
- Generated explanation
- Ground truth explanation
- Per-sample inference time

Usage::

    python scripts/run_inference.py --num-samples 5
    python scripts/run_inference.py --num-samples 100 --data data/synthetic
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import torch

# Allow running as a top-level script from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.api.schemas.architecture import Architecture
from backend.core.vision.encoder import encode_diagram
from backend.core.vlm.explainer import generate_explanation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)

# Model configuration
CONVNEXT_CHECKPOINT = "checkpoints/convnext_best.pt"
QWEN_LORA_CHECKPOINT = "checkpoints/qwen_lora/lora_adapter"


def _load_dataset(manifest_path: Path, num_samples: Optional[int] = None) -> list[dict]:
    """Load samples from JSONL manifest."""
    samples = []
    with manifest_path.open(encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            if num_samples and idx >= num_samples:
                break
            try:
                entry = json.loads(line.strip())
                samples.append(entry)
            except json.JSONDecodeError as e:
                logger.warning("Skipping invalid JSON at line %d: %s", idx, e)
    return samples


def _setup_device() -> torch.device:
    """Return a device (cuda if available, else cpu)."""
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        logger.info("Using GPU: %s", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    return device


def run_inference(
    data_dir: Path,
    num_samples: int = 5,
) -> None:
    """Run inference on the first N samples from the dataset.

    Args:
        data_dir:    Dataset root directory.
        num_samples: Number of samples to process.
    """
    # Load dataset
    manifest_path = data_dir / "dataset.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    samples = _load_dataset(manifest_path, num_samples)
    if not samples:
        logger.error("No samples loaded from manifest")
        return

    logger.info("Loaded %d samples from manifest", len(samples))

    # Setup device
    device = _setup_device()

    # Load Qwen + LoRA (warmup for GPU)
    logger.info("Loading Qwen2.5-1.5B with LoRA adapter...")
    try:
        from peft import PeftModel  # type: ignore
        from backend.core.vlm.qwen_loader import load_qwen

        model, tokenizer = load_qwen()
        
        # Try to load LoRA adapter if it exists
        adapter_dir = Path(QWEN_LORA_CHECKPOINT)
        if adapter_dir.exists():
            logger.info("Loading LoRA adapter from %s", adapter_dir)
            model = PeftModel.from_pretrained(model, str(adapter_dir))
            model.eval()
            logger.info("LoRA adapter loaded successfully")
        else:
            logger.warning("LoRA adapter not found at %s; using base model", adapter_dir)
    except Exception as e:
        logger.error("Failed to load Qwen + LoRA: %s", e)
        return

    # Run inference on each sample
    logger.info("\n" + "=" * 80)
    logger.info("INFERENCE RESULTS")
    logger.info("=" * 80)

    total_time = 0.0
    successful = 0

    for idx, sample in enumerate(samples):
        sample_id = sample.get("id", f"sample_{idx:04d}")
        image_path = sample.get("image", "")
        arch_dict = sample.get("architecture", {})
        ground_truth = sample.get("explanation", "")

        logger.info("\n" + "-" * 80)
        logger.info("Sample %d / %d: %s", idx + 1, len(samples), sample_id)
        logger.info("-" * 80)

        try:
            t0 = time.perf_counter()

            # Validate architecture
            try:
                arch = Architecture.model_validate(arch_dict)
            except Exception as e:
                logger.warning("Invalid architecture schema: %s", e)
                arch = None

            # Encode diagram
            if image_path and Path(image_path).exists():
                try:
                    vision_embedding = encode_diagram(image_path, device=device)
                    logger.info("Vision embedding shape: %s", vision_embedding.shape)
                except Exception as e:
                    logger.warning("Failed to encode diagram: %s", e)
                    vision_embedding = None
            else:
                logger.warning("Image not found: %s", image_path)
                vision_embedding = None

            # Generate explanation
            if arch:
                try:
                    generated = generate_explanation(
                        arch,
                        vision_features=vision_embedding,
                        max_new_tokens=256,
                        temperature=0.3,
                        top_p=0.9,
                    )
                except Exception as e:
                    logger.error("Explanation generation failed: %s", e)
                    generated = "[ERROR: Generation failed]"
            else:
                generated = "[SKIPPED: Invalid architecture]"

            elapsed = time.perf_counter() - t0
            total_time += elapsed

            # Print results
            logger.info("\nArchitecture:")
            logger.info(json.dumps(arch_dict, indent=2)[:500] + "...")

            logger.info("\nGenerated Explanation:")
            logger.info(generated[:500] if len(generated) > 500 else generated)

            logger.info("\nGround Truth Explanation:")
            logger.info(ground_truth[:500] if len(ground_truth) > 500 else ground_truth)

            logger.info("\nInference time: %.2f seconds", elapsed)
            successful += 1

        except Exception as e:
            logger.error("Unexpected error processing sample %s: %s", sample_id, e)
            import traceback
            traceback.print_exc()

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("INFERENCE SUMMARY")
    logger.info("=" * 80)
    logger.info("Total samples: %d", len(samples))
    logger.info("Successful: %d", successful)
    logger.info("Failed: %d", len(samples) - successful)
    if successful > 0:
        logger.info("Average time per sample: %.2f seconds", total_time / successful)
    logger.info("Total time: %.2f seconds", total_time)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference on synthetic ArchitectAI samples."
    )
    parser.add_argument(
        "--num-samples", type=int, default=5,
        metavar="N",
        help="Number of samples to process (default: 5)."
    )
    parser.add_argument(
        "--data", type=str, default="data/synthetic",
        metavar="DIR",
        help="Dataset root directory (default: data/synthetic)."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_inference(
        data_dir=Path(args.data),
        num_samples=args.num_samples,
    )
