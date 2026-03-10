"""export_examples.py — Export random dataset samples as demo artifacts.

Picks N random samples from dataset.jsonl and writes:
  docs/examples/png/<idx>_<prompt_slug>.png
  docs/examples/json/<idx>_architecture.json
  docs/examples/text/<idx>_explanation.txt
  docs/examples/README.md          — markdown table of all exported samples

Usage
-----
    python scripts/export_examples.py \\
        --data     data/synthetic \\
        --output   docs/examples \\
        --n        20 \\
        --seed     42
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List

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

def _slugify(text: str, max_len: int = 40) -> str:
    """Convert arbitrary text to a safe filename slug."""
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug[:max_len] or "example"


def _load_samples(manifest_path: Path) -> List[dict]:
    samples: List[dict] = []
    with manifest_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Only keep samples with at minimum an architecture and explanation.
            if entry.get("architecture") and entry.get("explanation"):
                samples.append(entry)
    return samples


# ---------------------------------------------------------------------------
# Export logic
# ---------------------------------------------------------------------------

def export_examples(
    manifest_path: Path,
    output_dir: Path,
    n: int,
    seed: int,
) -> List[Dict]:
    """Pick *n* random samples and write demo artifacts to *output_dir*."""
    samples = _load_samples(manifest_path)
    if not samples:
        logger.error("No valid samples found in %s", manifest_path)
        sys.exit(1)

    if n > len(samples):
        logger.warning(
            "Requested %d samples but only %d available; exporting all.", n, len(samples)
        )
        n = len(samples)

    rng = random.Random(seed)
    chosen = rng.sample(samples, n)

    png_dir  = output_dir / "png"
    json_dir = output_dir / "json"
    text_dir = output_dir / "text"
    for d in (png_dir, json_dir, text_dir):
        d.mkdir(parents=True, exist_ok=True)

    exported: List[Dict] = []

    for idx, entry in enumerate(chosen, start=1):
        # Derive a short slug from the prompt (if present) or architecture name.
        prompt_text = (
            entry.get("prompt")
            or entry.get("architecture", {}).get("name", "")
            or "example"
        )
        slug = _slugify(str(prompt_text))
        prefix = f"{idx:03d}_{slug}"

        # ── PNG ──────────────────────────────────────────────────────────────
        src_png   = Path(entry.get("image", ""))
        dest_png  = png_dir / f"{prefix}.png"
        png_copied = False
        if src_png.exists():
            shutil.copy2(src_png, dest_png)
            png_copied = True
            logger.info("  [%d] PNG  → %s", idx, dest_png)
        else:
            logger.warning("  [%d] PNG not found: %s", idx, src_png)

        # ── JSON ─────────────────────────────────────────────────────────────
        dest_json = json_dir / f"{prefix}_architecture.json"
        dest_json.write_text(
            json.dumps(entry["architecture"], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("  [%d] JSON → %s", idx, dest_json)

        # ── Explanation text ──────────────────────────────────────────────────
        dest_txt = text_dir / f"{prefix}_explanation.txt"
        dest_txt.write_text(entry["explanation"], encoding="utf-8")
        logger.info("  [%d] TXT  → %s", idx, dest_txt)

        exported.append({
            "index":           idx,
            "slug":            slug,
            "prompt":          prompt_text,
            "png_exported":    png_copied,
            "png_dest":        str(dest_png)  if png_copied else None,
            "json_dest":       str(dest_json),
            "text_dest":       str(dest_txt),
            "n_nodes":         len(entry["architecture"].get("nodes", [])),
            "n_edges":         len(entry["architecture"].get("edges", [])),
            "explanation_len": len(entry["explanation"]),
        })

    return exported


def _write_readme(output_dir: Path, exported: List[Dict]) -> None:
    """Generate a markdown summary of all exported examples."""
    lines = [
        "# ArchitectAI — Example Gallery",
        "",
        f"Generated {len(exported)} examples from the synthetic training dataset.",
        "",
        "## Samples",
        "",
        "| # | Prompt / Name | Nodes | Edges | PNG | Explanation length |",
        "|---|---------------|------:|------:|:---:|-------------------:|",
    ]

    for e in exported:
        prompt_display = str(e["prompt"])[:60].replace("|", "\\|")
        png_link = f"[view](png/{e['slug']}.png)" if e["png_exported"] else "—"
        lines.append(
            f"| {e['index']:3d} "
            f"| {prompt_display} "
            f"| {e['n_nodes']:5d} "
            f"| {e['n_edges']:5d} "
            f"| {png_link} "
            f"| {e['explanation_len']:6d} chars |"
        )

    lines += [
        "",
        "## Directory layout",
        "",
        "```",
        "docs/examples/",
        "├── png/     # Architecture diagrams (PNG)",
        "├── json/    # Architecture definitions (JSON)",
        "└── text/    # LLM explanations (plain text)",
        "```",
        "",
    ]

    readme_path = output_dir / "README.md"
    readme_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("README written → %s", readme_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export random dataset samples as demo artifacts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data", default="data/synthetic",
        help="Root directory containing dataset.jsonl.",
    )
    p.add_argument(
        "--output", default="docs/examples",
        help="Output directory for exported examples.",
    )
    p.add_argument(
        "--n", type=int, default=20,
        help="Number of samples to export.",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible sample selection.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    manifest_path = Path(args.data) / "dataset.jsonl"
    output_dir    = Path(args.output)

    if not manifest_path.exists():
        logger.error(
            "dataset.jsonl not found at %s. "
            "Run scripts/generate_dataset.py first.",
            manifest_path,
        )
        sys.exit(1)

    logger.info("Exporting %d examples from %s …", args.n, manifest_path)

    exported = export_examples(
        manifest_path=manifest_path,
        output_dir=output_dir,
        n=args.n,
        seed=args.seed,
    )

    _write_readme(output_dir, exported)

    # Print summary.
    png_ok    = sum(1 for e in exported if e["png_exported"])
    total     = len(exported)
    print(f"\nExported {total} sample(s) to {output_dir}/")
    print(f"  PNGs copied : {png_ok}/{total}")
    print(f"  JSON files  : {total}")
    print(f"  Text files  : {total}")
    print(f"  README      : {output_dir}/README.md\n")


if __name__ == "__main__":
    main()
