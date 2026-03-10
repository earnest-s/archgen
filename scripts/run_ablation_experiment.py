"""run_ablation_experiment.py — Vision ablation experiment runner for ArchitectAI.

Wraps run_ablation.py via subprocess, loads the resulting JSON, and prints a
formatted comparison table:
    Mode A — architecture text only → Qwen
    Mode B — architecture text + ConvNeXt vision features → Qwen

Saves results to reports/ablation_results.json.

Usage
-----
    python scripts/run_ablation_experiment.py [OPTIONS]

Options
-------
    --data         Dataset directory            (default: data/synthetic)
    --convnext     ConvNeXt checkpoint path     (default: checkpoints/convnext/convnext_best.pt)
    --max-samples  Samples to evaluate          (default: 50)
    --output       Report path                  (default: reports/ablation_results.json)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_SEP  = "=" * 60
_DASH = "-" * 60


def _run_ablation(args: argparse.Namespace) -> int:
    cmd = [
        sys.executable, "scripts/run_ablation.py",
        "--data",        args.data,
        "--convnext",    args.convnext,
        "--max-samples", str(args.max_samples),
        "--output",      args.output,
    ]
    print(f"  Running: {' '.join(cmd)}\n")
    return subprocess.run(cmd, text=True).returncode


def _fmt(val) -> str:
    if val is None:
        return "N/A"
    try:
        return f"{float(val):.4f}"
    except (TypeError, ValueError):
        return str(val)


def _delta_str(a, b) -> str:
    try:
        d = float(b) - float(a)
        sign = "+" if d >= 0 else ""
        return f"{sign}{d:.4f}"
    except (TypeError, ValueError):
        return "N/A"


def _print_comparison(report: dict) -> None:
    mode_a = report.get("mode_a", {})
    mode_b = report.get("mode_b", {})
    delta  = report.get("delta",  {})

    a_bleu  = mode_a.get("bleu4",   mode_a.get("bleu"))
    a_rouge = mode_a.get("rouge_l", mode_a.get("rougeL"))
    b_bleu  = mode_b.get("bleu4",   mode_b.get("bleu"))
    b_rouge = mode_b.get("rouge_l", mode_b.get("rougeL"))
    d_bleu  = delta.get("bleu4",    delta.get("bleu"))
    d_rouge = delta.get("rouge_l",  delta.get("rougeL"))

    # Fallback: compute delta if not pre-computed
    if d_bleu is None and a_bleu is not None and b_bleu is not None:
        try:
            d_bleu  = round(float(b_bleu)  - float(a_bleu),  4)
            d_rouge = round(float(b_rouge) - float(a_rouge), 4)
        except (TypeError, ValueError):
            pass

    print()
    print(_SEP)
    print("  Ablation Results — Text-Only vs. Text + Vision")
    print(_SEP)
    print(f"  {'Mode':<40} {'BLEU-4':>10} {'ROUGE-L':>10}")
    print(f"  {_DASH}")
    print(f"  {'Mode A: architecture text → Qwen':<40} {_fmt(a_bleu):>10} {_fmt(a_rouge):>10}")
    print(f"  {'Mode B: text + ConvNeXt features → Qwen':<40} {_fmt(b_bleu):>10} {_fmt(b_rouge):>10}")
    print(f"  {_DASH}")

    db = _delta_str(a_bleu, b_bleu)
    dr = _delta_str(a_rouge, b_rouge)
    print(f"  {'Delta  (B − A)':<40} {db:>10} {dr:>10}")
    print(_SEP)

    # Interpretation
    try:
        if float(b_bleu) > float(a_bleu):
            print("  ✓ Vision encoder IMPROVES explanation quality (BLEU-4).")
        else:
            print("  ⚠ Vision encoder does NOT improve BLEU-4 in this run.")
    except (TypeError, ValueError):
        pass
    print()


def main() -> None:
    args = _parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    print(_SEP)
    print("  ArchitectAI — Vision Ablation Experiment")
    print(_SEP)
    print(f"  Mode A : text-only  →  Qwen2.5-3B")
    print(f"  Mode B : text + ConvNeXt features  →  Qwen2.5-3B")
    print(f"  Samples: {args.max_samples}")
    print(_SEP)

    rc = _run_ablation(args)

    out_path = Path(args.output)
    if not out_path.exists():
        print(f"ERROR: run_ablation.py did not produce {out_path}")
        sys.exit(1)

    report = json.loads(out_path.read_text(encoding="utf-8"))
    _print_comparison(report)

    print(f"Report → {out_path}")
    sys.exit(rc)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run ArchitectAI vision ablation experiment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data",        default="data/synthetic")
    p.add_argument("--convnext",    default="checkpoints/convnext/convnext_best.pt")
    p.add_argument("--max-samples", type=int, default=50)
    p.add_argument("--output",      default="reports/ablation_results.json")
    return p.parse_args()


if __name__ == "__main__":
    main()
