"""run_evaluation.py — Evaluation command runner for ArchitectAI.

Executes eval_models.py via subprocess, captures its JSON output, and prints
a formatted summary table covering:
    • ConvNeXt: macro F1, micro F1, per-class F1, confusion matrix count
    • Explainer: BLEU-4, ROUGE-L (rule-based and optional LLM)

Saves results to reports/evaluation.json.

Usage
-----
    python scripts/run_evaluation.py [OPTIONS]

Options
-------
    --data              Dataset directory            (default: data/synthetic)
    --convnext          ConvNeXt checkpoint path     (default: checkpoints/convnext/convnext_best.pt)
    --output            Report path                  (default: reports/evaluation.json)
    --skip-vision       Skip vision encoder eval
    --use-llm           Evaluate Qwen LLM explainer
    --compare-explainers  Side-by-side rule-based vs LLM
    --max-samples       Max eval samples             (default: 200)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_SEP  = "=" * 60
_DASH = "-" * 60


def _run_eval(args: argparse.Namespace) -> int:
    cmd = [
        sys.executable, "scripts/eval_models.py",
        "--data",        args.data,
        "--convnext",    args.convnext,
        "--output",      args.output,
        "--max-samples", str(args.max_samples),
    ]
    if args.skip_vision:
        cmd.append("--skip-vision")
    if args.use_llm:
        cmd.append("--use-llm")
    if args.compare_explainers:
        cmd.append("--compare-explainers")
    print(f"  Running: {' '.join(cmd)}\n")
    return subprocess.run(cmd, text=True).returncode


def _print_convnext(vis: dict) -> None:
    print(_DASH)
    print("  Vision Model  —  ConvNeXt-Tiny (multi-label, 7 classes)")
    print(_DASH)
    print(f"  {'Exact-match accuracy':<30} {vis.get('exact_match', 'N/A')!r}")
    print(f"  {'Macro F1':<30} {vis.get('macro_f1', 'N/A')!r}")
    print(f"  {'Micro F1':<30} {vis.get('micro_f1', 'N/A')!r}")

    per_class: dict = vis.get("per_class_f1", {})
    if per_class:
        print()
        print(f"  {'Class':<25} {'F1':>8}")
        print(f"  {'-'*25} {'----':>8}")
        for cls, f1 in sorted(per_class.items(), key=lambda x: -x[1]):
            bar = "█" * int(f1 * 20)
            print(f"  {cls:<25} {f1:>8.4f}  {bar}")

    cm = vis.get("confusion_matrix")
    if cm:
        n_cells = sum(sum(row) for row in cm)
        print(f"\n  Confusion matrix: {len(cm)}×{len(cm)} ({n_cells:,} predictions)")


def _print_explainer(exp: dict, label: str = "Explainer") -> None:
    bleu  = exp.get("bleu4",   exp.get("bleu",   "N/A"))
    rouge = exp.get("rouge_l", exp.get("rougeL", "N/A"))
    print(f"  {label:<35} BLEU-4={bleu!r:<10} ROUGE-L={rouge!r}")


def _print_summary(report: dict) -> None:
    print()
    print(_SEP)
    print("  Evaluation Results Summary")
    print(_SEP)

    vis = report.get("convnext", {})
    if vis:
        _print_convnext(vis)
    else:
        print("  (Vision evaluation skipped)")

    print()
    print(_DASH)
    print("  Explanation Metrics")
    print(_DASH)

    compare = report.get("compare_explainers", {})
    if compare:
        _print_explainer(compare.get("rule_based", {}), "Rule-based baseline")
        _print_explainer(compare.get("qwen", {}),       "Qwen2.5-3B + LoRA")
    else:
        exp = report.get("explainer", report.get("qwen", {}))
        if exp:
            _print_explainer(exp, "Explainer")

    print()
    print(_SEP)


def main() -> None:
    args = _parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    print(_SEP)
    print("  ArchitectAI — Model Evaluation")
    print(_SEP)

    rc = _run_eval(args)

    out_path = Path(args.output)
    if not out_path.exists():
        print(f"ERROR: eval_models.py did not produce {out_path}")
        sys.exit(1)

    report = json.loads(out_path.read_text(encoding="utf-8"))
    _print_summary(report)

    print(f"Report → {out_path}")
    sys.exit(rc)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run ArchitectAI model evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data",               default="data/synthetic")
    p.add_argument("--convnext",           default="checkpoints/convnext/convnext_best.pt")
    p.add_argument("--output",             default="reports/evaluation.json")
    p.add_argument("--skip-vision",        action="store_true")
    p.add_argument("--use-llm",            action="store_true")
    p.add_argument("--compare-explainers", action="store_true")
    p.add_argument("--max-samples",        type=int, default=200)
    return p.parse_args()


if __name__ == "__main__":
    main()
