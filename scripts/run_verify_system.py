"""run_verify_system.py — CLI entry point for ArchitectAI system verification.

Imports and executes verify_system.py checks, prints a formatted PASS/FAIL
summary table, and saves results to reports/verify_system.json.

Exit code 0 only if all checks pass.

Usage
-----
    python scripts/run_verify_system.py [OPTIONS]

Options
-------
    --dataset      Path to dataset JSONL       (default: data/synthetic/dataset.jsonl)
    --convnext     Path to ConvNeXt checkpoint (default: checkpoints/convnext/convnext_best.pt)
    --qwen-lora    Path to LoRA adapter dir    (default: checkpoints/qwen_lora/lora_adapter)
    --skip-vision  Skip vision encoder check
    --output       JSON report path            (default: reports/verify_system.json)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Ensure project root is importable
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ---------------------------------------------------------------------------
# Re-import check functions from verify_system (no subprocess overhead)
# ---------------------------------------------------------------------------
from scripts.verify_system import (  # type: ignore
    check_dataset,
    check_convnext_checkpoint,
    check_qwen_lora,
    check_imports,
    check_parser,
    check_diagram_generator,
    check_vision_encoder,
    check_explainer,
)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

class CheckResult:
    __slots__ = ("name", "status", "elapsed_s", "detail", "error")

    def __init__(self, name: str) -> None:
        self.name      = name
        self.status    = "pending"
        self.elapsed_s = 0.0
        self.detail: Any = None
        self.error: str | None = None


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _run(name: str, fn: Callable[[], Any], results: list[CheckResult]) -> bool:
    rec = CheckResult(name)
    results.append(rec)
    t0 = time.perf_counter()
    try:
        rec.detail    = fn()
        rec.elapsed_s = time.perf_counter() - t0
        rec.status    = "PASS"
        return True
    except Exception as exc:
        rec.elapsed_s = time.perf_counter() - t0
        rec.status    = "FAIL"
        rec.error     = str(exc)
        return False


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------

_COL_NAME  = 42
_COL_TIME  = 9
_COL_EXTRA = 0  # dynamic


def _header() -> None:
    print("\n" + "=" * 70)
    print("  ArchitectAI — System Verification")
    print("=" * 70)
    print(f"  {'Check':<{_COL_NAME}} {'Time':>{_COL_TIME}}  Detail")
    print("-" * 70)


def _row(rec: CheckResult) -> None:
    icon   = "✓" if rec.status == "PASS" else "✗"
    extra  = str(rec.detail) if rec.detail and rec.status == "PASS" else ""
    err    = f"  ← {rec.error}" if rec.error else ""
    time_s = f"{rec.elapsed_s * 1000:>7.1f} ms"
    print(f"  {icon}  {rec.name:<{_COL_NAME}} {time_s}  {extra}{err}")


def _footer(passed: int, total: int, wall: float) -> None:
    print("=" * 70)
    icon = "ALL CHECKS PASSED ✓" if passed == total else f"{total - passed} CHECK(S) FAILED ✗"
    print(f"  {icon}  ({passed}/{total})  —  wall {wall:.2f}s")
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    dataset_path   = Path(args.dataset)
    convnext_path  = Path(args.convnext)
    qwen_lora_path = Path(args.qwen_lora)

    results: list[CheckResult] = []
    passed = 0
    total  = 0

    def run(name: str, fn: Callable[[], Any]) -> None:
        nonlocal passed, total
        total += 1
        if _run(name, fn, results):
            passed += 1

    _header()
    wall_start = time.perf_counter()

    # ── Checks ────────────────────────────────────────────────────────────────
    run("Dataset file exists",
        lambda: check_dataset(dataset_path))

    run("ConvNeXt checkpoint exists",
        lambda: check_convnext_checkpoint(convnext_path))

    run("Qwen LoRA adapter exists",
        lambda: check_qwen_lora(qwen_lora_path))

    run("Backend imports resolve",
        check_imports)

    run("Prompt parser produces nodes",
        check_parser)

    run("Diagram generator produces output",
        check_diagram_generator)

    if not args.skip_vision:
        run("Vision encoder → 768-dim embedding",
            check_vision_encoder)

    run("Rule-based explainer produces text",
        check_explainer)

    wall_elapsed = time.perf_counter() - wall_start

    # Print each row (after all checks so the table is contiguous)
    for rec in results:
        _row(rec)

    _footer(passed, total, wall_elapsed)

    # ── Save JSON report ─────────────────────────────────────────────────────
    report = {
        "n_total":   total,
        "n_passed":  passed,
        "n_failed":  total - passed,
        "all_pass":  passed == total,
        "wall_s":    round(wall_elapsed, 3),
        "checks": [
            {
                "name":      r.name,
                "status":    r.status,
                "elapsed_s": round(r.elapsed_s, 4),
                "detail":    r.detail,
                "error":     r.error,
            }
            for r in results
        ],
    }
    out = Path(args.output)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report → {out}")

    sys.exit(0 if passed == total else 1)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run ArchitectAI system verification checks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset",     default="data/synthetic/dataset.jsonl")
    p.add_argument("--convnext",    default="checkpoints/convnext/convnext_best.pt")
    p.add_argument("--qwen-lora",   default="checkpoints/qwen_lora/lora_adapter")
    p.add_argument("--skip-vision", action="store_true",
                   help="Skip vision encoder check.")
    p.add_argument("--output",      default="reports/verify_system.json")
    return p.parse_args()


if __name__ == "__main__":
    main()
