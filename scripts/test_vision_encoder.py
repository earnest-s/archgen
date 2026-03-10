#!/usr/bin/env python3
"""
Vision encoder validation script for ArchitectAI.

Loads (or generates) a sample architecture diagram PNG, runs it through the
ConvNeXt-Tiny encoder, and prints embedding statistics.

Usage::

    python scripts/test_vision_encoder.py
    python scripts/test_vision_encoder.py --image path/to/diagram.png
    python scripts/test_vision_encoder.py --generate-sample
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
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


def _generate_sample_diagram() -> str:
    """Generate a sample architecture PNG and return its path."""
    from backend.core.diagram.generator import generate_diagram
    from backend.core.prompt_parser.parser import parse_prompt

    prompt = "React frontend, FastAPI backend, PostgreSQL database, Redis cache"
    logger.info("Generating sample diagram for prompt: %r", prompt)
    arch = parse_prompt(prompt)

    tmpdir = tempfile.mkdtemp(prefix="architectai_vision_test_")
    output_base = str(Path(tmpdir) / "sample_arch")
    png_path = generate_diagram(arch, output_path=output_base)
    logger.info("Sample diagram written to: %s", png_path)
    return png_path


def _print_separator(char: str = "─", width: int = 60) -> None:
    print(char * width)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def validate_encoder(image_path: str) -> None:
    """Run the ConvNeXt encoder on *image_path* and print diagnostics."""
    from backend.core.vision.encoder import encode_diagram

    path = Path(image_path)
    if not path.exists():
        logger.error("Image not found: %s", image_path)
        sys.exit(1)

    file_size_kb = path.stat().st_size / 1024
    _print_separator()
    print(f"  Image path  : {image_path}")
    print(f"  File size   : {file_size_kb:.1f} KB")
    _print_separator()

    # ── Verify image is loadable ──────────────────────────────────────────
    from PIL import Image
    with Image.open(image_path) as img:
        print(f"  PIL mode    : {img.mode}")
        print(f"  PIL size    : {img.size[0]}×{img.size[1]} px")

    _print_separator()
    print("  Running ConvNeXt-Tiny encoder …")
    t0 = time.perf_counter()
    embedding = encode_diagram(image_path)
    elapsed = time.perf_counter() - t0

    # ── Embedding statistics ──────────────────────────────────────────────
    mean_val  = embedding.mean().item()
    std_val   = embedding.std().item()
    min_val   = embedding.min().item()
    max_val   = embedding.max().item()
    norm_val  = embedding.norm().item()
    nonzero   = (embedding != 0).sum().item()
    device    = embedding.device.type

    _print_separator()
    print(f"  Shape       : {tuple(embedding.shape)}")
    print(f"  Device      : {device}")
    print(f"  Dtype       : {embedding.dtype}")
    print(f"  Mean        : {mean_val:+.6f}")
    print(f"  Std         : {std_val:.6f}")
    print(f"  Min         : {min_val:+.6f}")
    print(f"  Max         : {max_val:+.6f}")
    print(f"  L2 norm     : {norm_val:.4f}")
    print(f"  Non-zero    : {nonzero} / {embedding.numel()}")
    print(f"  Latency     : {elapsed * 1000:.1f} ms")
    _print_separator()

    # ── GPU info ──────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        dev = torch.cuda.get_device_properties(0)
        print(f"  GPU         : {dev.name}")
        print(f"  VRAM used   : {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        _print_separator()
    else:
        print("  GPU         : not available (running on CPU)")
        _print_separator()

    # ── Validation checks ─────────────────────────────────────────────────
    print("\n  Validation checks:")
    checks_passed = 0
    checks_total  = 0

    def check(label: str, condition: bool) -> None:
        nonlocal checks_passed, checks_total
        checks_total += 1
        status = "✓" if condition else "✗"
        print(f"    {status} {label}")
        if condition:
            checks_passed += 1

    check("Shape is (768,)",                    embedding.shape == (768,))
    check("Dtype is float32",                   embedding.dtype == torch.float32)
    check("Device is cpu (result moved)",       device == "cpu")
    check("All values are finite",              torch.isfinite(embedding).all().item())
    check("L2 norm > 0",                        norm_val > 0.0)
    check("Std > 0 (non-constant output)",      std_val > 1e-6)
    check("Latency < 30 s",                     elapsed < 30.0)

    print(f"\n  {checks_passed}/{checks_total} checks passed")
    _print_separator()

    if checks_passed < checks_total:
        logger.error("Some validation checks failed.")
        sys.exit(1)
    else:
        print("\n  ✓ Vision encoder validation PASSED\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the ConvNeXt-Tiny vision encoder."
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to an existing diagram PNG.  If omitted, a sample is generated.",
    )
    parser.add_argument(
        "--generate-sample",
        action="store_true",
        help="Force generation of a fresh sample diagram even if --image is given.",
    )
    args = parser.parse_args()

    if args.generate_sample or args.image is None:
        image_path = _generate_sample_diagram()
    else:
        image_path = args.image

    validate_encoder(image_path)


if __name__ == "__main__":
    main()
