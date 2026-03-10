"""run_ablation.py — Compare text-only vs text+vision explanation quality.

Mode A: architecture JSON → Qwen2.5 (no vision features)
Mode B: architecture JSON + ConvNeXt vision embedding → Qwen2.5

Metrics: BLEU-4 and ROUGE-L (vs reference explanation in dataset.jsonl).

Usage
-----
    python scripts/run_ablation.py \\
        --data      data/synthetic \\
        --convnext  checkpoints/convnext/convnext_best.pt \\
        --output    reports/ablation_results.json \\
        --max-samples 50
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_samples(manifest_path: Path, max_samples: int) -> List[dict]:
    samples: List[dict] = []
    with manifest_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("explanation") and entry.get("architecture"):
                samples.append(entry)
                if len(samples) >= max_samples:
                    break
    return samples


def _init_nlp():
    """Import and initialise nltk + rouge_score; return (nltk, scorer, smoothing)."""
    try:
        import nltk  # type: ignore
        from nltk.translate.bleu_score import SmoothingFunction  # type: ignore
        nltk.download("punkt",     quiet=True)
        nltk.download("punkt_tab", quiet=True)
    except ImportError as exc:
        logger.error("nltk not installed: %s", exc)
        sys.exit(1)

    try:
        from rouge_score import rouge_scorer  # type: ignore
    except ImportError as exc:
        logger.error("rouge-score not installed: %s", exc)
        sys.exit(1)

    scorer    = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    smoothing = SmoothingFunction().method1
    return nltk, scorer, smoothing


def _vision_features(image_path: str, convnext_ckpt: Optional[Path]):
    """Return a float tensor of shape (768,) or None if anything fails."""
    if convnext_ckpt is None or not Path(image_path).exists():
        return None
    try:
        from backend.core.vision.encoder import encode_diagram  # type: ignore
        feats = encode_diagram(image_path, checkpoint_path=str(convnext_ckpt))
        return feats
    except Exception as exc:
        logger.debug("Vision encoding failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Core ablation logic
# ---------------------------------------------------------------------------

def run_ablation(
    manifest_path: Path,
    convnext_ckpt: Optional[Path],
    max_samples: int,
) -> Dict:
    """Run Mode A (no vision) and Mode B (with vision) and return metrics dict."""
    from backend.api.schemas.architecture import Architecture  # type: ignore
    from backend.core.vlm.explainer import generate_explanation  # type: ignore

    samples = _load_samples(manifest_path, max_samples)
    if not samples:
        logger.error("No valid samples found in %s", manifest_path)
        sys.exit(1)

    logger.info("Running ablation on %d samples …", len(samples))

    nltk, scorer, smoothing = _init_nlp()
    from nltk.translate.bleu_score import corpus_bleu  # type: ignore

    mode_a_refs:  List[List[List[str]]] = []
    mode_a_hyps:  List[List[str]]       = []
    mode_a_rouge: List[float]           = []

    mode_b_refs:  List[List[List[str]]] = []
    mode_b_hyps:  List[List[str]]       = []
    mode_b_rouge: List[float]           = []

    for idx, entry in enumerate(samples):
        logger.info("  Sample %d/%d …", idx + 1, len(samples))

        try:
            arch = Architecture.model_validate(entry["architecture"])
        except Exception as exc:
            logger.warning("Skipping sample %d (invalid architecture): %s", idx, exc)
            continue

        reference  = entry["explanation"]
        ref_tokens = nltk.word_tokenize(reference.lower())

        # ── Mode A: text-only ────────────────────────────────────────────────
        try:
            hyp_a = generate_explanation(arch, vision_features=None)
        except Exception as exc:
            logger.warning("Mode A failed on sample %d: %s", idx, exc)
            hyp_a = ""

        hyp_a_tokens = nltk.word_tokenize(hyp_a.lower())
        mode_a_refs.append([ref_tokens])
        mode_a_hyps.append(hyp_a_tokens)
        mode_a_rouge.append(scorer.score(reference, hyp_a)["rougeL"].fmeasure)

        # ── Mode B: text + vision ────────────────────────────────────────────
        vision_feat = _vision_features(entry.get("image", ""), convnext_ckpt)
        try:
            hyp_b = generate_explanation(arch, vision_features=vision_feat)
        except Exception as exc:
            logger.warning("Mode B failed on sample %d: %s", idx, exc)
            hyp_b = hyp_a  # fall back to Mode A output

        hyp_b_tokens = nltk.word_tokenize(hyp_b.lower())
        mode_b_refs.append([ref_tokens])
        mode_b_hyps.append(hyp_b_tokens)
        mode_b_rouge.append(scorer.score(reference, hyp_b)["rougeL"].fmeasure)

    def _metrics(refs, hyps, rouge_list) -> Dict:
        if not hyps:
            return {"bleu4": 0.0, "rouge_l": 0.0, "n_samples": 0}
        bleu4 = float(corpus_bleu(
            refs, hyps,
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=smoothing,
        ))
        return {
            "bleu4":     round(bleu4, 4),
            "rouge_l":   round(sum(rouge_list) / len(rouge_list), 4),
            "n_samples": len(hyps),
        }

    return {
        "mode_a_text_only":    _metrics(mode_a_refs, mode_a_hyps, mode_a_rouge),
        "mode_b_text_vision":  _metrics(mode_b_refs, mode_b_hyps, mode_b_rouge),
        "vision_checkpoint":   str(convnext_ckpt) if convnext_ckpt else None,
        "max_samples":         max_samples,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Ablation: text-only vs text+vision explanation quality.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data", default="data/synthetic",
        help="Root directory containing dataset.jsonl.",
    )
    p.add_argument(
        "--convnext", default="checkpoints/convnext/convnext_best.pt",
        help="Path to the ConvNeXt checkpoint used for vision features (Mode B).",
    )
    p.add_argument(
        "--output", default="reports/ablation_results.json",
        help="Path to write ablation results JSON.",
    )
    p.add_argument(
        "--max-samples", type=int, default=50,
        help="Maximum number of dataset samples to evaluate.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    manifest_path = Path(args.data) / "dataset.jsonl"
    output_path   = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    convnext_ckpt = Path(args.convnext) if Path(args.convnext).exists() else None
    if convnext_ckpt is None:
        logger.warning(
            "ConvNeXt checkpoint not found at %s — Mode B will run without vision.",
            args.convnext,
        )

    if not manifest_path.exists():
        logger.error(
            "dataset.jsonl not found at %s. "
            "Run scripts/generate_dataset.py first.",
            manifest_path,
        )
        sys.exit(1)

    results = run_ablation(
        manifest_path=manifest_path,
        convnext_ckpt=convnext_ckpt,
        max_samples=args.max_samples,
    )

    # ── Print comparison table ────────────────────────────────────────────────
    a = results["mode_a_text_only"]
    b = results["mode_b_text_vision"]
    print("\n" + "=" * 58)
    print("  Ablation: Text-Only vs Text+Vision")
    print("=" * 58)
    print(f"  {'Mode':<24} {'BLEU-4':>8} {'ROUGE-L':>9} {'N':>6}")
    print(f"  {'-'*24} {'-'*8} {'-'*9} {'-'*6}")
    print(f"  {'A — text-only':<24} {a['bleu4']:>8.4f} {a['rouge_l']:>9.4f} {a['n_samples']:>6}")
    print(f"  {'B — text+vision':<24} {b['bleu4']:>8.4f} {b['rouge_l']:>9.4f} {b['n_samples']:>6}")

    delta_bleu   = round(b["bleu4"]   - a["bleu4"],   4)
    delta_rouge  = round(b["rouge_l"] - a["rouge_l"], 4)
    sign_b = "+" if delta_bleu  >= 0 else ""
    sign_r = "+" if delta_rouge >= 0 else ""
    print(f"\n  Delta (B − A): BLEU-4 {sign_b}{delta_bleu:.4f}  ROUGE-L {sign_r}{delta_rouge:.4f}")
    print("=" * 58 + "\n")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    output_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Ablation results saved → %s", output_path)


if __name__ == "__main__":
    main()
