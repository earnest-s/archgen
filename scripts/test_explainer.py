#!/usr/bin/env python3
"""
LLM explanation validation script for ArchitectAI.

Loads a sample architecture, generates a dummy vision embedding, and
calls generate_explanation() (or the rule-based fallback). Prints the
generated explanation along with timing and prompt visibility.

Usage::

    # Rule-based explanation (fast, no GPU required):
    python scripts/test_explainer.py

    # Full LLM explanation with Qwen2.5-3B-Instruct (requires ~4 GB download):
    python scripts/test_explainer.py --use-llm

    # Custom architecture JSON file:
    python scripts/test_explainer.py --arch path/to/arch.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _print_separator(char: str = "─", width: int = 62) -> None:
    print(char * width)


def _load_architecture(arch_path: str | None):
    """Load Architecture from a JSON file, or build a default sample."""
    from backend.api.schemas.architecture import Architecture
    from backend.core.prompt_parser.parser import parse_prompt

    if arch_path:
        path = Path(arch_path)
        if not path.exists():
            logger.error("Architecture file not found: %s", arch_path)
            sys.exit(1)
        with path.open() as fh:
            data = json.load(fh)
        arch = Architecture.model_validate(data)
        logger.info("Loaded architecture from %s", arch_path)
    else:
        prompt = "React frontend, FastAPI backend, PostgreSQL database, Redis cache"
        logger.info("Using default prompt: %r", prompt)
        arch = parse_prompt(prompt)

    return arch


def _dummy_vision_embedding(dim: int = 768) -> torch.Tensor:
    """Return a plausible fake ConvNeXt embedding (unit normal noise)."""
    torch.manual_seed(42)
    return torch.randn(dim)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_explainer(arch_path: str | None, use_llm: bool) -> None:
    from backend.core.vlm.explainer import generate_explanation_rule_based

    arch = _load_architecture(arch_path)
    embedding = _dummy_vision_embedding()

    _print_separator("═")
    print("  ArchitectAI — Explanation Validator")
    _print_separator("═")

    # ── Architecture summary ──────────────────────────────────────────────
    print("\n  Architecture:")
    for node in arch.nodes:
        print(f"    • [{node.type.value:10s}] {node.id} → {node.label!r}")
    if arch.edges:
        print("\n  Edges:")
        for edge in arch.edges:
            proto = f" [{edge.protocol}]" if edge.protocol else ""
            print(f"    {edge.from_node} → {edge.to_node}{proto}")

    # ── Vision embedding summary ──────────────────────────────────────────
    print(f"\n  Vision embedding  : shape={tuple(embedding.shape)}  norm={embedding.norm():.4f}")
    _print_separator()

    # ── Rule-based (always run) ───────────────────────────────────────────
    print("\n  [Rule-based explanation]")
    t0 = time.perf_counter()
    rule_explanation = generate_explanation_rule_based(arch)
    rule_latency = time.perf_counter() - t0

    print(f"\n  {rule_explanation}")
    print(f"\n  Latency : {rule_latency * 1000:.1f} ms")

    # ── Validation checks ─────────────────────────────────────────────────
    _print_separator()
    print("\n  Validation checks (rule-based):")
    checks_passed = 0
    checks_total  = 0

    def check(label: str, condition: bool) -> None:
        nonlocal checks_passed, checks_total
        checks_total += 1
        status = "✓" if condition else "✗"
        print(f"    {status} {label}")
        if condition:
            checks_passed += 1

    check("Output is a non-empty string",        bool(rule_explanation.strip()))
    check("Length > 20 chars",                   len(rule_explanation.strip()) > 20)
    check("Latency < 1 s",                       rule_latency < 1.0)

    explanation_lower = rule_explanation.lower()
    architectural_terms = {
        "frontend", "backend", "database", "service", "cache", "queue",
        "api", "server", "tier", "data", "layer", "componen",
    }
    found_terms = [t for t in architectural_terms if t in explanation_lower]
    check(
        f"Mentions architectural terms ({found_terms[:3]})",
        len(found_terms) > 0,
    )

    # ── LLM explanation (optional) ────────────────────────────────────────
    if use_llm:
        _print_separator()
        print("\n  [LLM explanation — Qwen2.5-3B-Instruct 4-bit]")
        print("  Loading model (first run may download ~4 GB) …")

        from backend.core.vlm.explainer import generate_explanation

        t0 = time.perf_counter()
        llm_explanation = generate_explanation(arch, embedding)
        llm_latency = time.perf_counter() - t0

        print(f"\n  {llm_explanation}")
        print(f"\n  Latency : {llm_latency:.2f} s")

        _print_separator()
        print("\n  Validation checks (LLM):")
        checks_total += 1
        if llm_explanation.strip():
            checks_passed += 1
            print("    ✓ LLM returned non-empty string")
        else:
            print("    ✗ LLM returned empty string")

    _print_separator()
    print(f"\n  {checks_passed}/{checks_total} checks passed")
    _print_separator()

    if checks_passed < checks_total:
        logger.error("Some validation checks failed.")
        sys.exit(1)
    else:
        print("\n  ✓ Explainer validation PASSED\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the ArchitectAI explanation generator."
    )
    parser.add_argument(
        "--arch",
        type=str,
        default=None,
        help="Path to an architecture JSON file. Defaults to a built-in sample.",
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Also run the full Qwen2.5-3B-Instruct LLM (requires model download).",
    )
    args = parser.parse_args()
    run_explainer(arch_path=args.arch, use_llm=args.use_llm)


if __name__ == "__main__":
    main()
