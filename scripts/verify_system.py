"""verify_system.py — End-to-end system verification for ArchitectAI.

Checks:
  1. Dataset file exists and is non-empty
  2. ConvNeXt checkpoint exists
  3. Qwen LoRA adapter directory exists
  4. Core imports resolve (backend packages)
  5. Prompt parser produces a valid architecture
  6. Diagram generator produces a non-empty output
  7. Vision encoder produces a 768-dim embedding
  8. Rule-based explainer produces a non-empty explanation

Each check reports PASS / FAIL with elapsed time.
Exit code 0 if all checks pass, 1 if any fail.

Usage
-----
    python scripts/verify_system.py [OPTIONS]

Options
-------
    --dataset      Path to dataset JSONL (default: data/synthetic/dataset.jsonl)
    --convnext     Path to ConvNeXt checkpoint (default: checkpoints/convnext/convnext_best.pt)
    --qwen-lora    Path to LoRA adapter dir (default: checkpoints/qwen_lora/lora_adapter)
    --skip-vision  Skip vision encoder check (faster)
    --output       JSON report path (default: reports/verify_system.json)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

# Ensure project root is on the path so backend imports work.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ---------------------------------------------------------------------------
# Check runner
# ---------------------------------------------------------------------------


class Result:
    __slots__ = ("name", "status", "elapsed_s", "detail", "error")

    def __init__(self, name: str) -> None:
        self.name      = name
        self.status    = "pending"
        self.elapsed_s = 0.0
        self.detail: Any = None
        self.error: str | None = None


def _run_check(name: str, fn: Callable[[], Any], results: list[Result]) -> bool:
    rec = Result(name)
    results.append(rec)
    t0 = time.perf_counter()
    try:
        detail = fn()
        rec.elapsed_s = time.perf_counter() - t0
        rec.status    = "PASS"
        rec.detail    = detail
        _print_row(rec)
        return True
    except Exception as exc:
        rec.elapsed_s = time.perf_counter() - t0
        rec.status    = "FAIL"
        rec.error     = str(exc)
        _print_row(rec)
        return False


def _print_row(rec: Result) -> None:
    icon  = "✓" if rec.status == "PASS" else "✗"
    extra = f"  {rec.detail}" if rec.detail and rec.status == "PASS" else ""
    err   = f"  ERROR: {rec.error}" if rec.error else ""
    print(f"  {icon}  {rec.name:<40} {rec.elapsed_s*1000:6.1f} ms{extra}{err}")


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def check_dataset(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")
    lines = sum(1 for _ in path.open("r", encoding="utf-8"))
    if lines == 0:
        raise ValueError("Dataset file is empty")
    return f"({lines:,} lines)"


def check_convnext_checkpoint(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")
    size_mb = path.stat().st_size / 1_048_576
    return f"({size_mb:.1f} MB)"


def check_qwen_lora(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")
    cfg = path / "adapter_config.json"
    if not cfg.exists():
        raise FileNotFoundError(f"adapter_config.json missing in {path}")
    return f"({sum(1 for _ in path.rglob('*') if _.is_file())} files)"


def check_imports() -> str:
    # These must import without error
    from backend.core.prompt_parser.parser import PromptParser  # noqa: F401
    from backend.core.diagram.generator    import DiagramGenerator  # noqa: F401
    from backend.core.vlm.explainer        import ArchitectureExplainer  # noqa: F401
    return "prompt_parser, diagram.generator, vlm.explainer"


def check_parser() -> str:
    from backend.core.prompt_parser.parser import PromptParser
    parser = PromptParser()
    arch   = parser.parse("Design a three-tier web app with React, FastAPI, and PostgreSQL.")
    nodes  = arch.get("nodes") or arch.get("components") or []
    if not nodes:
        raise ValueError(f"Parser returned no nodes: {arch}")
    return f"({len(nodes)} nodes parsed)"


def check_diagram_generator() -> str:
    from backend.core.prompt_parser.parser import PromptParser
    from backend.core.diagram.generator   import DiagramGenerator

    parser    = PromptParser()
    arch      = parser.parse("Design a REST API with FastAPI and PostgreSQL.")
    generator = DiagramGenerator()
    result    = generator.generate(arch)

    # Result may be bytes (PNG), a path string, or a dict with 'diagram'
    if isinstance(result, bytes) and len(result) > 0:
        return f"(PNG, {len(result):,} bytes)"
    if isinstance(result, str) and Path(result).exists():
        return f"(file: {result})"
    if isinstance(result, dict):
        nodes = result.get("nodes") or []
        return f"(layout dict, {len(nodes)} nodes)"
    raise ValueError(f"Unexpected generator output: {type(result)}")


def check_vision_encoder() -> str:
    import torch
    from backend.core.vision.encoder import VisionEncoder

    enc  = VisionEncoder()
    # Create a fake 224×224 RGB image tensor (batch=1)
    img  = torch.zeros(1, 3, 224, 224)
    with torch.no_grad():
        emb = enc(img)
    if emb.shape[-1] != 768:
        raise ValueError(f"Expected 768-dim embedding, got {emb.shape[-1]}")
    return f"(shape {tuple(emb.shape)})"


def check_explainer() -> str:
    from backend.core.prompt_parser.parser import PromptParser
    from backend.core.vlm.explainer        import ArchitectureExplainer

    parser  = PromptParser()
    arch    = parser.parse("React frontend connected to FastAPI and Redis cache.")
    exp     = ArchitectureExplainer(use_llm=False)  # rule-based only
    text    = exp.explain(arch)
    if not text or len(text.strip()) < 20:
        raise ValueError(f"Explanation too short: {repr(text)}")
    return f"({len(text)} chars)"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = _parse_args()

    dataset_path   = Path(args.dataset)
    convnext_path  = Path(args.convnext)
    qwen_lora_path = Path(args.qwen_lora)
    output_path    = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  ArchitectAI — System Verification")
    print("=" * 60)
    print(f"  {'Check':<40} {'Time':>8}")
    print("-" * 60)

    results: list[Result] = []
    checks_passed = 0
    checks_total  = 0

    def run(name: str, fn: Callable[[], Any]) -> None:
        nonlocal checks_passed, checks_total
        checks_total += 1
        if _run_check(name, fn, results):
            checks_passed += 1

    # ── File / directory checks ───────────────────────────────────────────────
    run("Dataset file exists",          lambda: check_dataset(dataset_path))
    run("ConvNeXt checkpoint exists",   lambda: check_convnext_checkpoint(convnext_path))
    run("Qwen LoRA adapter exists",     lambda: check_qwen_lora(qwen_lora_path))

    # ── Import checks ────────────────────────────────────────────────────────
    run("Backend imports resolve",      check_imports)

    # ── Functional checks ────────────────────────────────────────────────────
    run("Prompt parser output",         check_parser)
    run("Diagram generator output",     check_diagram_generator)

    if not args.skip_vision:
        run("Vision encoder (768-dim)", check_vision_encoder)

    run("Rule-based explainer output",  check_explainer)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("=" * 60)
    all_pass = checks_passed == checks_total
    icon = "✓ ALL CHECKS PASSED" if all_pass else f"✗ {checks_total - checks_passed} CHECK(S) FAILED"
    print(f"  {icon}  ({checks_passed}/{checks_total})")
    print("=" * 60)

    # ── Write JSON report ────────────────────────────────────────────────────
    report = {
        "n_total":   checks_total,
        "n_passed":  checks_passed,
        "n_failed":  checks_total - checks_passed,
        "all_pass":  all_pass,
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
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport written → {output_path}")

    sys.exit(0 if all_pass else 1)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Verify ArchitectAI system end-to-end.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset",      default="data/synthetic/dataset.jsonl")
    p.add_argument("--convnext",     default="checkpoints/convnext/convnext_best.pt")
    p.add_argument("--qwen-lora",    default="checkpoints/qwen_lora/lora_adapter")
    p.add_argument("--skip-vision",  action="store_true",
                   help="Skip the vision encoder check (avoids loading ConvNeXt).")
    p.add_argument("--output",       default="reports/verify_system.json")
    return p.parse_args()


if __name__ == "__main__":
    main()
