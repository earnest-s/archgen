"""run_visualizations.py — Visualization runner for ArchitectAI metrics.

Calls visualize_results.py via subprocess to generate four figures:
    1. per_class_f1.png      — bar chart of per-class F1 scores
    2. confusion_matrix.png  — heatmap of the 7×7 confusion matrix
    3. bleu_rouge_comparison.png — grouped bar: rule-based vs LLM
    4. ablation_delta.png    — delta chart: Mode B − Mode A

Figures are saved to reports/figures/ (default).

Usage
-----
    python scripts/run_visualizations.py [OPTIONS]

Options
-------
    --eval      Path to evaluation.json     (default: reports/evaluation.json)
    --ablation  Path to ablation_results.json (default: reports/ablation_results.json)
    --output    Output directory for figures (default: reports/figures)
    --dpi       Figure DPI                  (default: 150)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_SEP  = "=" * 60
_DASH = "-" * 60

_EXPECTED_FIGURES = [
    "per_class_f1.png",
    "confusion_matrix.png",
    "bleu_rouge_comparison.png",
    "ablation_delta.png",
]


def _run_visualizer(args: argparse.Namespace) -> int:
    cmd = [
        sys.executable, "scripts/visualize_results.py",
        "--eval",     args.eval,
        "--ablation", args.ablation,
        "--output",   args.output,
        "--dpi",      str(args.dpi),
    ]
    print(f"  Running: {' '.join(cmd)}\n")
    return subprocess.run(cmd, text=True).returncode


def _check_outputs(out_dir: Path) -> list[dict]:
    results = []
    for fname in _EXPECTED_FIGURES:
        path = out_dir / fname
        if path.exists():
            mb = path.stat().st_size / 1_048_576
            results.append({"file": fname, "exists": True, "mb": round(mb, 3)})
        else:
            results.append({"file": fname, "exists": False, "mb": 0})
    return results


def main() -> None:
    args = _parse_args()
    Path(args.output).mkdir(parents=True, exist_ok=True)

    print(_SEP)
    print("  ArchitectAI — Visualization Generator")
    print(_SEP)
    print(f"  Evaluation data : {args.eval}")
    print(f"  Ablation data   : {args.ablation}")
    print(f"  Output dir      : {args.output}")
    print(f"  DPI             : {args.dpi}")
    print(_SEP)

    rc = _run_visualizer(args)

    out_dir = Path(args.output)
    checks  = _check_outputs(out_dir)

    print()
    print(_SEP)
    print("  Generated Figures")
    print(_DASH)
    all_ok = True
    for c in checks:
        icon = "✓" if c["exists"] else "✗"
        size = f"({c['mb']:.3f} MB)" if c["exists"] else "(MISSING)"
        print(f"  {icon}  {c['file']:<40} {size}")
        if not c["exists"]:
            all_ok = False
    print(_SEP)

    if all_ok:
        print("  All figures generated successfully.")
    else:
        print("  WARNING: Some figures are missing.")

    print(f"\n  Figures saved → {out_dir}/")
    sys.exit(rc if all_ok else 1)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate ArchitectAI result visualizations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--eval",     default="reports/evaluation.json")
    p.add_argument("--ablation", default="reports/ablation_results.json")
    p.add_argument("--output",   default="reports/figures")
    p.add_argument("--dpi",      type=int, default=150)
    return p.parse_args()


if __name__ == "__main__":
    main()
