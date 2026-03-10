"""visualize_results.py — Generate charts from evaluation and ablation reports.

Loads:
  reports/evaluation.json      — ConvNeXt + explanation metrics
  reports/ablation_results.json — text-only vs text+vision comparison

Generates the following figures in reports/figures/:
  1. per_class_f1.png          — horizontal bar chart of F1 per NodeType
  2. confusion_matrix.png      — heatmap of dominant-label confusion matrix
  3. bleu_rouge_comparison.png — BLEU-4 / ROUGE-L bar chart (rule-based vs LLM)
  4. ablation_delta.png        — delta bar chart (Mode B − Mode A)

Usage
-----
    python scripts/visualize_results.py [OPTIONS]

Options
-------
    --eval       Path to evaluation.json  (default: reports/evaluation.json)
    --ablation   Path to ablation_results.json
    --output     Directory for figures    (default: reports/figures)
    --dpi        Resolution of saved figures (default: 150)
    --no-show    Do not open figures interactively (always True in CI)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Matplotlib guard
# ---------------------------------------------------------------------------

def _require_matplotlib():
    try:
        import matplotlib  # type: ignore
        matplotlib.use("Agg")  # non-interactive backend — safe for scripts
        import matplotlib.pyplot as plt  # type: ignore
        return plt
    except ImportError:
        logger.error("matplotlib not installed. Run: pip install matplotlib")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Individual chart generators
# ---------------------------------------------------------------------------

def plot_per_class_f1(
    vision: Dict[str, Any],
    output_dir: Path,
    dpi: int,
    plt,
) -> Optional[Path]:
    """Horizontal bar chart — F1 per NodeType."""
    per_f1 = vision.get("per_class_f1", {})
    if not per_f1:
        logger.warning("per_class_f1 not found in evaluation.json — skipping.")
        return None

    labels = list(per_f1.keys())
    f1_vals = [per_f1[k]["f1"] for k in labels]
    prec    = [per_f1[k]["precision"] for k in labels]
    rec     = [per_f1[k]["recall"]    for k in labels]

    x  = list(range(len(labels)))
    bw = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh([i + bw     for i in x], prec,    bw, label="Precision", color="#4C72B0")
    ax.barh([i          for i in x], rec,     bw, label="Recall",    color="#55A868")
    ax.barh([i - bw     for i in x], f1_vals, bw, label="F1",        color="#C44E52")
    ax.set_yticks(x)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Score")
    ax.set_title("Per-Class Precision / Recall / F1 (ConvNeXt)")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1.05)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()

    path = output_dir / "per_class_f1.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    logger.info("Saved: %s", path)
    return path


def plot_confusion_matrix(
    vision: Dict[str, Any],
    output_dir: Path,
    dpi: int,
    plt,
) -> Optional[Path]:
    """Heatmap of the dominant-label confusion matrix."""
    cm_data = vision.get("confusion_matrix", {})
    matrix  = cm_data.get("matrix")
    labels  = cm_data.get("labels")

    if not matrix or not labels:
        logger.warning("confusion_matrix not found in evaluation.json — skipping.")
        return None

    try:
        import numpy as np  # type: ignore
    except ImportError:
        logger.warning("numpy not installed — skipping confusion matrix.")
        return None

    cm = np.array(matrix, dtype=float)
    # Normalize by row (true label) — avoid division by zero.
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm / row_sums

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, label="Normalised fraction")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix — Dominant NodeType (ConvNeXt)")

    # Annotate cells.
    thresh = 0.5
    for i in range(len(labels)):
        for j in range(len(labels)):
            color = "white" if cm_norm[i, j] > thresh else "black"
            ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color=color)

    fig.tight_layout()
    path = output_dir / "confusion_matrix.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    logger.info("Saved: %s", path)
    return path


def plot_bleu_rouge_comparison(
    eval_data: Dict[str, Any],
    output_dir: Path,
    dpi: int,
    plt,
) -> Optional[Path]:
    """Grouped bar — BLEU-4 and ROUGE-L for rule-based vs LLM."""
    cmp = eval_data.get("explainer_comparison", {})
    expl = eval_data.get("explanation", {})

    # Build dataset: may come from comparison block or single explanation block.
    modes: Dict[str, Dict[str, float]] = {}
    if "rule_based" in cmp:
        modes["Rule-based"] = {
            "BLEU-4":  cmp["rule_based"].get("bleu4",   0.0),
            "ROUGE-L": cmp["rule_based"].get("rouge_l", 0.0),
        }
    if "llm" in cmp:
        modes["LLM (Qwen)"] = {
            "BLEU-4":  cmp["llm"].get("bleu4",   0.0),
            "ROUGE-L": cmp["llm"].get("rouge_l", 0.0),
        }
    if not modes and expl:
        label = "LLM (Qwen)" if expl.get("llm_used") else "Rule-based"
        modes[label] = {
            "BLEU-4":  expl.get("bleu4",   0.0),
            "ROUGE-L": expl.get("rouge_l", 0.0),
        }

    if not modes:
        logger.warning("No explanation comparison data found — skipping BLEU/ROUGE chart.")
        return None

    metrics = ["BLEU-4", "ROUGE-L"]
    colors  = ["#4C72B0", "#DD8452"]
    x = list(range(len(metrics)))
    bw = 0.8 / max(len(modes), 1)

    fig, ax = plt.subplots(figsize=(7, 5))
    for i, (mode_name, scores) in enumerate(modes.items()):
        offsets = [xi + (i - len(modes) / 2 + 0.5) * bw for xi in x]
        vals    = [scores[m] for m in metrics]
        bars    = ax.bar(offsets, vals, bw * 0.9, label=mode_name, color=colors[i % len(colors)])
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("BLEU-4 / ROUGE-L: Rule-based vs LLM")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    path = output_dir / "bleu_rouge_comparison.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    logger.info("Saved: %s", path)
    return path


def plot_ablation_delta(
    ablation: Dict[str, Any],
    output_dir: Path,
    dpi: int,
    plt,
) -> Optional[Path]:
    """Delta bar chart — Mode B − Mode A (text+vision minus text-only)."""
    a = ablation.get("mode_a_text_only",   {})
    b = ablation.get("mode_b_text_vision", {})

    if not a or not b:
        logger.warning("Ablation data missing modes — skipping ablation delta chart.")
        return None

    metrics = ["BLEU-4", "ROUGE-L"]
    a_vals  = [a.get("bleu4", 0), a.get("rouge_l", 0)]
    b_vals  = [b.get("bleu4", 0), b.get("rouge_l", 0)]
    deltas  = [round(bv - av, 4) for av, bv in zip(a_vals, b_vals)]
    colors  = ["#4CAF50" if d >= 0 else "#F44336" for d in deltas]

    x = list(range(len(metrics)))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: absolute scores side-by-side.
    bw = 0.35
    axes[0].bar([xi - bw / 2 for xi in x], a_vals, bw, label="A — text-only",   color="#4C72B0")
    axes[0].bar([xi + bw / 2 for xi in x], b_vals, bw, label="B — text+vision", color="#55A868")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics)
    axes[0].set_ylim(0, max(max(a_vals + b_vals) * 1.3, 0.1))
    axes[0].set_ylabel("Score")
    axes[0].set_title("Ablation: Absolute Scores")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    # Right: deltas.
    bars = axes[1].bar(x, deltas, color=colors, width=0.5)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(metrics)
    axes[1].set_ylabel("Δ Score (B − A)")
    axes[1].set_title("Ablation: Vision Contribution (Δ)")
    for bar, d in zip(bars, deltas):
        sign = "+" if d >= 0 else ""
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                     d + (0.001 if d >= 0 else -0.003),
                     f"{sign}{d:.4f}", ha="center",
                     va="bottom" if d >= 0 else "top", fontsize=10)
    axes[1].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = output_dir / "ablation_delta.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    logger.info("Saved: %s", path)
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    plt  = _require_matplotlib()

    eval_path    = Path(args.eval)
    ablation_path = Path(args.ablation)
    output_dir   = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data.
    eval_data:    Dict = {}
    ablation_data: Dict = {}

    if eval_path.exists():
        eval_data = json.loads(eval_path.read_text(encoding="utf-8"))
        logger.info("Loaded evaluation data from %s", eval_path)
    else:
        logger.warning("evaluation.json not found at %s", eval_path)

    if ablation_path.exists():
        ablation_data = json.loads(ablation_path.read_text(encoding="utf-8"))
        logger.info("Loaded ablation data from %s", ablation_path)
    else:
        logger.warning("ablation_results.json not found at %s", ablation_path)

    vision = eval_data.get("vision", {})
    saved: list[str] = []

    # Chart 1: per-class F1.
    p = plot_per_class_f1(vision, output_dir, args.dpi, plt)
    if p:
        saved.append(str(p))

    # Chart 2: confusion matrix.
    p = plot_confusion_matrix(vision, output_dir, args.dpi, plt)
    if p:
        saved.append(str(p))

    # Chart 3: BLEU / ROUGE comparison.
    p = plot_bleu_rouge_comparison(eval_data, output_dir, args.dpi, plt)
    if p:
        saved.append(str(p))

    # Chart 4: ablation delta.
    p = plot_ablation_delta(ablation_data, output_dir, args.dpi, plt)
    if p:
        saved.append(str(p))

    print(f"\nSaved {len(saved)} figure(s) to {output_dir}/")
    for s in saved:
        print(f"  {s}")
    print()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate visualisation charts from ArchitectAI evaluation reports.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--eval",     default="reports/evaluation.json")
    p.add_argument("--ablation", default="reports/ablation_results.json")
    p.add_argument("--output",   default="reports/figures")
    p.add_argument("--dpi",      type=int, default=150)
    return p.parse_args()


if __name__ == "__main__":
    main()
