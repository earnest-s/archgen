"""
Evaluation script for ArchitectAI inference.

Computes BLEU-4 and ROUGE-L metrics comparing generated explanations
against ground truth explanations for all samples in the dataset.

Metrics:
- BLEU-4: Bilingual Evaluation Understudy (4-gram precision)
- ROUGE-L: Recall-Oriented Understudy for Gisting Evaluation (longest common subsequence)

Results are saved to reports/inference_eval.json with:
- Per-sample scores
- Average metrics across all samples
- Sample count and timing information

Usage::

    python scripts/run_evaluation_inference.py
    python scripts/run_evaluation_inference.py --data data/synthetic --output reports/
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

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction  # type: ignore
    nltk_available = True
except ImportError:
    nltk_available = False

try:
    from rouge_score import rouge_scorer  # type: ignore
    rouge_available = True
except ImportError:
    rouge_available = False

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


def _load_dataset(manifest_path: Path) -> list[dict]:
    """Load all samples from JSONL manifest."""
    samples = []
    with manifest_path.open(encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
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


def _compute_bleu4(reference: str, hypothesis: str) -> float:
    """Compute BLEU-4 score between reference and hypothesis."""
    if not nltk_available:
        logger.warning("nltk not available; skipping BLEU-4 computation")
        return 0.0

    try:
        ref_tokens = reference.lower().split()
        hyp_tokens = hypothesis.lower().split()

        if not hyp_tokens:
            return 0.0

        smoothing_fn = SmoothingFunction().method1
        score = sentence_bleu(
            [ref_tokens],
            hyp_tokens,
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=smoothing_fn,
        )
        return float(score)
    except Exception as e:
        logger.warning("BLEU-4 computation failed: %s", e)
        return 0.0


def _compute_rouge_l(reference: str, hypothesis: str) -> float:
    """Compute ROUGE-L (F-score) between reference and hypothesis."""
    if not rouge_available:
        logger.warning("rouge_score not available; skipping ROUGE-L computation")
        return 0.0

    try:
        scorer = rouge_scorer.RougeScorer(
            ["rougeL"], use_stemmer=True  # type: ignore
        )
        scores = scorer.score(reference, hypothesis)
        f_score = scores["rougeL"].fmeasure  # type: ignore
        return float(f_score)
    except Exception as e:
        logger.warning("ROUGE-L computation failed: %s", e)
        return 0.0


def run_evaluation(
    data_dir: Path,
    output_dir: Path,
) -> None:
    """Evaluate inference quality on all samples.

    Computes BLEU-4 and ROUGE-L metrics for each sample, then aggregates.

    Args:
        data_dir:    Dataset root directory.
        output_dir:  Directory to save evaluation results.
    """
    # Check dependencies
    if not nltk_available:
        logger.error("NLTK not installed. Run: pip install nltk")
        return
    if not rouge_available:
        logger.error("rouge_score not installed. Run: pip install rouge_score")
        return

    # Load dataset
    manifest_path = data_dir / "dataset.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    samples = _load_dataset(manifest_path)
    if not samples:
        logger.error("No samples loaded from manifest")
        return

    logger.info("Loaded %d samples for evaluation", len(samples))

    # Setup device
    device = _setup_device()

    # Load Qwen + LoRA
    logger.info("Loading Qwen2.5-1.5B with LoRA adapter...")
    try:
        from peft import PeftModel  # type: ignore
        from backend.core.vlm.qwen_loader import load_qwen

        model, tokenizer = load_qwen()
        
        adapter_dir = Path(QWEN_LORA_CHECKPOINT)
        if adapter_dir.exists():
            logger.info("Loading LoRA adapter from %s", adapter_dir)
            model = PeftModel.from_pretrained(model, str(adapter_dir))
            model.eval()
        else:
            logger.warning("LoRA adapter not found; using base model")
    except Exception as e:
        logger.error("Failed to load Qwen + LoRA: %s", e)
        return

    # Evaluation loop
    logger.info("Starting evaluation on %d samples...\n", len(samples))

    results = {
        "samples": [],
        "metrics": {
            "bleu4_mean": 0.0,
            "bleu4_std": 0.0,
            "rouge_l_mean": 0.0,
            "rouge_l_std": 0.0,
        },
        "metadata": {
            "total_samples": len(samples),
            "successful_samples": 0,
            "evaluation_time_s": 0.0,
        },
    }

    eval_start = time.perf_counter()
    bleu4_scores = []
    rouge_l_scores = []

    for idx, sample in enumerate(samples):
        sample_id = sample.get("id", f"sample_{idx:04d}")
        image_path = sample.get("image", "")
        arch_dict = sample.get("architecture", {})
        ground_truth = sample.get("explanation", "")

        if (idx + 1) % 10 == 0 or idx == 0:
            logger.info("Processing sample %d / %d...", idx + 1, len(samples))

        try:
            # Validate architecture
            try:
                arch = Architecture.model_validate(arch_dict)
            except Exception:
                logger.debug("Invalid architecture for sample %s; skipping", sample_id)
                continue

            # Encode diagram (optional; not used in current LLM but loaded for completeness)
            vision_embedding = None
            if image_path and Path(image_path).exists():
                try:
                    vision_embedding = encode_diagram(image_path, device=device)
                except Exception:
                    pass

            # Generate explanation
            try:
                generated = generate_explanation(
                    arch,
                    vision_features=vision_embedding,
                    max_new_tokens=256,
                    temperature=0.3,
                    top_p=0.9,
                )
            except Exception as e:
                logger.debug("Explanation generation failed for %s: %s", sample_id, e)
                continue

            # Compute metrics
            bleu4 = _compute_bleu4(ground_truth, generated)
            rouge_l = _compute_rouge_l(ground_truth, generated)

            bleu4_scores.append(bleu4)
            rouge_l_scores.append(rouge_l)

            results["samples"].append({
                "id": sample_id,
                "bleu4": round(bleu4, 4),
                "rouge_l": round(rouge_l, 4),
            })

        except Exception as e:
            logger.warning("Evaluation failed for sample %s: %s", sample_id, e)

    eval_elapsed = time.perf_counter() - eval_start

    # Compute aggregate statistics
    if bleu4_scores:
        import statistics
        results["metadata"]["successful_samples"] = len(bleu4_scores)
        results["metrics"]["bleu4_mean"] = round(statistics.mean(bleu4_scores), 4)
        results["metrics"]["rouge_l_mean"] = round(statistics.mean(rouge_l_scores), 4)
        if len(bleu4_scores) > 1:
            results["metrics"]["bleu4_std"] = round(statistics.stdev(bleu4_scores), 4)
            results["metrics"]["rouge_l_std"] = round(statistics.stdev(rouge_l_scores), 4)
    results["metadata"]["evaluation_time_s"] = round(eval_elapsed, 2)

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "inference_eval.json"
    results_path.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 80)
    logger.info("Total samples: %d", results["metadata"]["total_samples"])
    logger.info("Evaluated: %d", results["metadata"]["successful_samples"])
    logger.info("BLEU-4 (avg): %.4f ± %.4f", 
               results["metrics"]["bleu4_mean"],
               results["metrics"]["bleu4_std"])
    logger.info("ROUGE-L (avg): %.4f ± %.4f",
               results["metrics"]["rouge_l_mean"],
               results["metrics"]["rouge_l_std"])
    logger.info("Evaluation time: %.2f seconds", results["metadata"]["evaluation_time_s"])
    logger.info("Results saved → %s", results_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate ArchitectAI inference quality."
    )
    parser.add_argument(
        "--data", type=str, default="data/synthetic",
        metavar="DIR",
        help="Dataset root directory (default: data/synthetic)."
    )
    parser.add_argument(
        "--output", type=str, default="reports",
        metavar="DIR",
        help="Output directory for evaluation results (default: reports)."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_evaluation(
        data_dir=Path(args.data),
        output_dir=Path(args.output),
    )
