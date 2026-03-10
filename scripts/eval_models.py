"""
Model evaluation script for ArchitectAI.

Evaluates:
  1. ConvNeXt-Tiny multi-label classification on validation diagram images.
  2. Qwen2.5-3B explanation quality against reference text (BLEU-4 + ROUGE-L).

Results are printed to stdout and saved to ``reports/evaluation.json``.

Usage::

    python scripts/eval_models.py \\
        --data        data/synthetic \\
        --convnext    checkpoints/convnext/convnext_best.pt \\
        --output      reports/evaluation.json \\
        --max-samples 200 \\
        [--use-llm]       # add to include LLM evaluation (slow)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Allow running from the repo root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)

NODE_TYPES: List[str] = [
    "Frontend", "Backend", "Service", "Database", "Cache", "Queue", "External"
]


# ---------------------------------------------------------------------------
# Vision evaluation
# ---------------------------------------------------------------------------


def evaluate_convnext(
    manifest_path: Path,
    checkpoint_path: Optional[Path],
    max_samples: int,
) -> Dict:
    """Evaluate ConvNeXt multi-label classification on validation diagrams.

    Returns a dict with overall accuracy, per-class accuracy, and F1 score.
    """
    import torch
    import torch.nn as nn
    from sklearn.metrics import f1_score  # type: ignore

    from backend.core.vision.convnext_loader import CONVNEXT_TINY_FEATURES, load_convnext
    from backend.core.vision.preprocess import preprocess_image

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("ConvNeXt evaluation on %s", device)

    # Build the same model architecture as training.
    backbone = load_convnext(pretrained=False, device=device)
    model = nn.Sequential(
        backbone,
        nn.Linear(CONVNEXT_TINY_FEATURES, len(NODE_TYPES)),
    ).to(device)

    if checkpoint_path and checkpoint_path.exists():
        state = torch.load(str(checkpoint_path), map_location=device)
        model.load_state_dict(state)
        logger.info("Loaded checkpoint: %s", checkpoint_path)
    else:
        logger.warning(
            "No checkpoint found at %s — evaluating with random weights.",
            checkpoint_path,
        )

    model.eval()

    # Load manifest.
    samples: List[dict] = []
    with manifest_path.open(encoding="utf-8") as fh:
        for line in fh:
            entry = json.loads(line.strip())
            if entry.get("image") and Path(entry["image"]).exists():
                samples.append(entry)
                if len(samples) >= max_samples:
                    break

    logger.info("Evaluating on %d samples", len(samples))

    all_preds:  List[List[int]] = []
    all_labels: List[List[int]] = []

    with torch.inference_mode():
        for entry in samples:
            # Ground-truth multi-hot vector.
            arch_types = {n["type"] for n in entry["architecture"]["nodes"]}
            label = [1 if t in arch_types else 0 for t in NODE_TYPES]

            img = preprocess_image(entry["image"]).to(device)
            logits = model(img)
            preds  = (torch.sigmoid(logits.squeeze(0)) > 0.5).int().cpu().tolist()

            all_preds.append(preds)
            all_labels.append(label)

    import numpy as np
    preds_arr  = np.array(all_preds,  dtype=int)
    labels_arr = np.array(all_labels, dtype=int)

    # Exact-match (all labels correct per sample).
    exact_match = float((preds_arr == labels_arr).all(axis=1).mean())

    # Per-class accuracy.
    per_class_acc = {
        NODE_TYPES[i]: float((preds_arr[:, i] == labels_arr[:, i]).mean())
        for i in range(len(NODE_TYPES))
    }

    # Macro and micro F1.
    macro_f1 = float(f1_score(labels_arr, preds_arr, average="macro",  zero_division=0))
    micro_f1 = float(f1_score(labels_arr, preds_arr, average="micro",  zero_division=0))

    # Per-class precision, recall, F1
    from sklearn.metrics import classification_report, confusion_matrix  # type: ignore
    report = classification_report(
        labels_arr, preds_arr,
        target_names=NODE_TYPES,
        zero_division=0,
        output_dict=True,
    )
    per_class_f1 = {
        t: {
            "precision": round(report[t]["precision"], 4),
            "recall":    round(report[t]["recall"],    4),
            "f1":        round(report[t]["f1-score"],  4),
        }
        for t in NODE_TYPES
    }

    # Confusion matrix (using argmax for dominant label per sample)
    dominant_pred  = preds_arr.argmax(axis=1)
    dominant_label = labels_arr.argmax(axis=1)
    cm = confusion_matrix(dominant_label, dominant_pred, labels=list(range(len(NODE_TYPES))))

    return {
        "exact_match_accuracy": round(exact_match, 4),
        "macro_f1":             round(macro_f1, 4),
        "micro_f1":             round(micro_f1, 4),
        "per_class_accuracy":   {k: round(v, 4) for k, v in per_class_acc.items()},
        "per_class_f1":         per_class_f1,
        "confusion_matrix":     {"labels": NODE_TYPES, "matrix": cm.tolist()},
        "n_samples":            len(samples),
    }


# ---------------------------------------------------------------------------
# Explanation evaluation (BLEU + ROUGE)
# ---------------------------------------------------------------------------


def evaluate_explanations(
    manifest_path: Path,
    max_samples: int,
    use_llm: bool = False,
) -> Dict:
    """Evaluate explanation quality vs reference text using BLEU-4 and ROUGE-L.

    If *use_llm* is False, uses the rule-based explainer as both the hypothesis
    and a sanity check (effectively measures template consistency).  Set
    ``--use-llm`` to use the Qwen LLM (slow, requires GPU).

    Returns a dict with bleu4 and rouge_l scores.
    """
    try:
        import nltk  # type: ignore
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction  # type: ignore
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
    except ImportError:
        logger.error("nltk not installed. Run: pip install nltk")
        return {"bleu4": None, "rouge_l": None, "error": "nltk not installed"}

    try:
        from rouge_score import rouge_scorer  # type: ignore
    except ImportError:
        logger.error("rouge-score not installed. Run: pip install rouge-score")
        return {"bleu4": None, "rouge_l": None, "error": "rouge-score not installed"}

    from backend.api.schemas.architecture import Architecture
    from backend.core.vlm.explainer import (
        generate_explanation,
        generate_explanation_rule_based,
    )

    samples: List[dict] = []
    with manifest_path.open(encoding="utf-8") as fh:
        for line in fh:
            entry = json.loads(line.strip())
            if entry.get("explanation"):
                samples.append(entry)
                if len(samples) >= max_samples:
                    break

    logger.info(
        "Explanation evaluation on %d samples (use_llm=%s)", len(samples), use_llm
    )

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    references_corpus: List[List[List[str]]] = []
    hypotheses_corpus: List[List[str]]       = []
    rouge_l_scores: List[float]              = []
    smoothing = SmoothingFunction().method1

    for entry in samples:
        try:
            arch = Architecture.model_validate(entry["architecture"])
        except Exception as exc:
            logger.warning("Skipping sample (invalid architecture): %s", exc)
            continue

        reference: str = entry["explanation"]

        if use_llm:
            hypothesis = generate_explanation(arch)
        else:
            hypothesis = generate_explanation_rule_based(arch)

        ref_tokens  = nltk.word_tokenize(reference.lower())
        hyp_tokens  = nltk.word_tokenize(hypothesis.lower())

        references_corpus.append([ref_tokens])
        hypotheses_corpus.append(hyp_tokens)

        rs = scorer.score(reference, hypothesis)
        rouge_l_scores.append(rs["rougeL"].fmeasure)

    if not hypotheses_corpus:
        return {"bleu4": 0.0, "rouge_l": 0.0, "n_samples": 0}

    bleu4   = float(corpus_bleu(references_corpus, hypotheses_corpus,
                                weights=(0.25, 0.25, 0.25, 0.25),
                                smoothing_function=smoothing))
    rouge_l = float(sum(rouge_l_scores) / len(rouge_l_scores))

    return {
        "bleu4":             round(bleu4, 4),
        "rouge_l":           round(rouge_l, 4),
        "n_samples":         len(hypotheses_corpus),
        "llm_used":          use_llm,
    }


# ---------------------------------------------------------------------------
# Rule-based vs LLM comparison
# ---------------------------------------------------------------------------


def compare_explainers(
    manifest_path: Path,
    max_samples: int,
) -> Dict:
    """Run both rule-based and LLM explainers on the same samples.

    Returns a side-by-side dict with BLEU-4 and ROUGE-L for each explainer.
    Requires the Qwen LLM to be in the Python path and importable.
    """
    try:
        import nltk  # type: ignore
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction  # type: ignore
        nltk.download("punkt",     quiet=True)
        nltk.download("punkt_tab", quiet=True)
    except ImportError:
        return {"error": "nltk not installed"}

    try:
        from rouge_score import rouge_scorer  # type: ignore
    except ImportError:
        return {"error": "rouge-score not installed"}

    from backend.api.schemas.architecture import Architecture
    from backend.core.vlm.explainer import (
        generate_explanation,
        generate_explanation_rule_based,
    )

    samples: List[dict] = []
    with manifest_path.open(encoding="utf-8") as fh:
        for line in fh:
            entry = json.loads(line.strip())
            if entry.get("explanation"):
                samples.append(entry)
                if len(samples) >= max_samples:
                    break

    logger.info("Explainer comparison on %d samples", len(samples))

    scorer    = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    smoothing = SmoothingFunction().method1

    def _score_mode(use_llm: bool) -> Dict:
        refs_corpus: List[List[List[str]]] = []
        hyps_corpus: List[List[str]]       = []
        rouge_scores: List[float]           = []

        for entry in samples:
            try:
                arch = Architecture.model_validate(entry["architecture"])
            except Exception:
                continue
            reference = entry["explanation"]

            try:
                if use_llm:
                    hypothesis = generate_explanation(arch)
                else:
                    hypothesis = generate_explanation_rule_based(arch)
            except Exception as exc:
                logger.warning("Explainer error (llm=%s): %s", use_llm, exc)
                continue

            ref_tokens = nltk.word_tokenize(reference.lower())
            hyp_tokens = nltk.word_tokenize(hypothesis.lower())
            refs_corpus.append([ref_tokens])
            hyps_corpus.append(hyp_tokens)
            rouge_scores.append(
                scorer.score(reference, hypothesis)["rougeL"].fmeasure
            )

        if not hyps_corpus:
            return {"bleu4": 0.0, "rouge_l": 0.0, "n_samples": 0}

        return {
            "bleu4":    round(float(corpus_bleu(
                refs_corpus, hyps_corpus,
                weights=(0.25, 0.25, 0.25, 0.25),
                smoothing_function=smoothing,
            )), 4),
            "rouge_l":  round(float(sum(rouge_scores) / len(rouge_scores)), 4),
            "n_samples": len(hyps_corpus),
        }

    rule_based_scores = _score_mode(use_llm=False)
    logger.info("Rule-based scores: %s", rule_based_scores)
    llm_scores = _score_mode(use_llm=True)
    logger.info("LLM scores: %s", llm_scores)

    return {
        "rule_based": rule_based_scores,
        "llm":        llm_scores,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate ArchitectAI models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data", default="data/synthetic",
        help="Root directory containing dataset.jsonl.",
    )
    p.add_argument(
        "--convnext", default="checkpoints/convnext/convnext_best.pt",
        help="Path to ConvNeXt checkpoint (.pt).",
    )
    p.add_argument(
        "--output", default="reports/evaluation.json",
        help="Path to save evaluation results JSON.",
    )
    p.add_argument(
        "--max-samples", type=int, default=200,
        help="Maximum samples to evaluate per metric.",
    )
    p.add_argument(
        "--use-llm", action="store_true",
        help="Use Qwen LLM for explanation evaluation (slow).",
    )
    p.add_argument(
        "--skip-vision", action="store_true",
        help="Skip ConvNeXt vision evaluation.",
    )
    p.add_argument(
        "--skip-explanation", action="store_true",
        help="Skip explanation quality evaluation.",
    )
    p.add_argument(
        "--compare-explainers", action="store_true",
        help="Run both rule-based and LLM explainers side-by-side and compare BLEU-4/ROUGE-L.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    data_dir      = Path(args.data)
    manifest_path = data_dir / "dataset.jsonl"
    output_path   = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        logger.error(
            "dataset.jsonl not found at %s. "
            "Run scripts/generate_dataset.py first.",
            manifest_path,
        )
        sys.exit(1)

    results: Dict = {}

    # ── 1. Vision evaluation ──────────────────────────────────────────────────
    if not args.skip_vision:
        logger.info("=== ConvNeXt Evaluation ===")
        try:
            vision_results = evaluate_convnext(
                manifest_path=manifest_path,
                checkpoint_path=Path(args.convnext),
                max_samples=args.max_samples,
            )
            results["vision"] = vision_results
            logger.info("Vision results: %s", vision_results)
        except Exception as exc:
            logger.exception("Vision evaluation failed: %s", exc)
            results["vision"] = {"error": str(exc)}

    # ── 2. Explanation evaluation ─────────────────────────────────────────────
    if not args.skip_explanation:
        logger.info("=== Explanation Evaluation ===")
        try:
            expl_results = evaluate_explanations(
                manifest_path=manifest_path,
                max_samples=args.max_samples,
                use_llm=args.use_llm,
            )
            results["explanation"] = expl_results
            logger.info("Explanation results: %s", expl_results)
        except Exception as exc:
            logger.exception("Explanation evaluation failed: %s", exc)
            results["explanation"] = {"error": str(exc)}

    # ── 3. Rule-based vs LLM comparison ──────────────────────────────────────
    if getattr(args, "compare_explainers", False):
        logger.info("=== Rule-based vs LLM Comparison ===")
        try:
            cmp_results = compare_explainers(
                manifest_path=manifest_path,
                max_samples=args.max_samples,
            )
            results["explainer_comparison"] = cmp_results
            logger.info("Comparison results: %s", cmp_results)
        except Exception as exc:
            logger.exception("Explainer comparison failed: %s", exc)
            results["explainer_comparison"] = {"error": str(exc)}

    # ── 4. Print summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ArchitectAI Evaluation Results")
    print("=" * 60)

    if "vision" in results and "error" not in results["vision"]:
        v = results["vision"]
        print(f"\n  Vision (ConvNeXt) — {v['n_samples']} samples")
        print(f"    Exact-match accuracy : {v['exact_match_accuracy']:.4f}")
        print(f"    Macro F1             : {v['macro_f1']:.4f}")
        print(f"    Micro F1             : {v['micro_f1']:.4f}")
        print("    Per-class accuracy:")
        for cls, acc in v.get("per_class_accuracy", {}).items():
            print(f"      {cls:<10} {acc:.4f}")
        if "per_class_f1" in v:
            print("    Per-class precision / recall / F1:")
            for cls, m in v["per_class_f1"].items():
                print(f"      {cls:<10}  P={m['precision']:.4f}  R={m['recall']:.4f}  F1={m['f1']:.4f}")

    if "explanation" in results and "error" not in results["explanation"]:
        e = results["explanation"]
        print(f"\n  Explanation (Qwen) — {e['n_samples']} samples")
        print(f"    BLEU-4     : {e['bleu4']:.4f}")
        print(f"    ROUGE-L    : {e['rouge_l']:.4f}")

    if "explainer_comparison" in results and "error" not in results["explainer_comparison"]:
        cmp = results["explainer_comparison"]
        rb  = cmp.get("rule_based", {})
        llm = cmp.get("llm", {})
        print("\n  Explainer Comparison")
        print(f"    {'Mode':<14} {'BLEU-4':>8} {'ROUGE-L':>9} {'Samples':>9}")
        print(f"    {'-'*14} {'-'*8} {'-'*9} {'-'*9}")
        print(f"    {'Rule-based':<14} {rb.get('bleu4', 0):>8.4f} {rb.get('rouge_l', 0):>9.4f} {rb.get('n_samples', 0):>9}")
        print(f"    {'LLM (Qwen)':<14} {llm.get('bleu4', 0):>8.4f} {llm.get('rouge_l', 0):>9.4f} {llm.get('n_samples', 0):>9}")

    print("=" * 60 + "\n")

    # ── 4. Save JSON ──────────────────────────────────────────────────────────
    output_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Results saved → %s", output_path)


if __name__ == "__main__":
    main()
