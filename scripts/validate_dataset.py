"""
Dataset quality validator for ArchitectAI.

Loads ``data/synthetic/dataset.jsonl`` and verifies structural correctness,
file availability, and statistical balance before training.

Usage::

    python scripts/validate_dataset.py
    python scripts/validate_dataset.py --data data/synthetic --output reports/dataset_summary.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)

NODE_TYPES = ["Frontend", "Backend", "Service", "Database", "Cache", "Queue", "External"]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_dataset(manifest_path: Path, output_path: Path) -> Dict:
    """Load and validate the JSONL manifest.

    Returns a summary dict written to *output_path*.
    """
    if not manifest_path.exists():
        logger.error("Manifest not found: %s", manifest_path)
        sys.exit(1)

    entries: List[dict] = []
    with manifest_path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning("Line %d: JSON parse error — %s", lineno, exc)

    n_total = len(entries)
    logger.info("Loaded %d entries from %s", n_total, manifest_path)

    # ── Per-entry validation ──────────────────────────────────────────────────
    missing_image        = 0
    missing_architecture = 0
    missing_explanation  = 0
    missing_png_file     = 0
    small_diagram        = 0   # < 2 nodes
    empty_explanation    = 0

    node_type_counter: Counter = Counter()
    node_counts: List[int]     = []
    edge_counts: List[int]     = []
    valid_count                = 0

    for i, entry in enumerate(entries):
        stem = entry.get("id", f"entry_{i}")

        # ── Required field checks ─────────────────────────────────────────────
        if not entry.get("image"):
            missing_image += 1
            logger.warning("%s: missing 'image' field", stem)

        arch_data = entry.get("architecture")
        if not arch_data:
            missing_architecture += 1
            logger.warning("%s: missing 'architecture' field", stem)
            continue

        expl = entry.get("explanation", "")
        if not expl:
            missing_explanation += 1
            logger.warning("%s: missing 'explanation' field", stem)
        elif not expl.strip():
            empty_explanation += 1
            logger.warning("%s: explanation is empty/whitespace", stem)

        # ── PNG existence ─────────────────────────────────────────────────────
        img_path = Path(entry.get("image", ""))
        if img_path and not img_path.exists():
            missing_png_file += 1
            logger.warning("%s: PNG not found at %s", stem, img_path)

        # ── Architecture stats ────────────────────────────────────────────────
        nodes = arch_data.get("nodes", [])
        edges = arch_data.get("edges", [])
        n_nodes = len(nodes)
        n_edges = len(edges)

        if n_nodes < 2:
            small_diagram += 1
            logger.warning("%s: only %d node(s) — too few", stem, n_nodes)

        node_counts.append(n_nodes)
        edge_counts.append(n_edges)

        for node in nodes:
            ntype = node.get("type", "Unknown")
            node_type_counter[ntype] += 1

        valid_count += 1

    # ── Statistics ────────────────────────────────────────────────────────────
    avg_nodes = sum(node_counts) / len(node_counts) if node_counts else 0.0
    avg_edges = sum(edge_counts) / len(edge_counts) if edge_counts else 0.0

    # Node type distribution (as % of total node instances)
    total_node_instances = sum(node_type_counter.values())
    node_type_distribution = {
        t: {
            "count":   node_type_counter.get(t, 0),
            "percent": round(
                100.0 * node_type_counter.get(t, 0) / max(total_node_instances, 1), 2
            ),
        }
        for t in NODE_TYPES
    }

    # Sample-level: how many samples contain each NodeType at least once
    sample_type_presence: Counter = Counter()
    for entry in entries:
        arch_data = entry.get("architecture", {})
        seen = set()
        for node in arch_data.get("nodes", []):
            seen.add(node.get("type"))
        for t in seen:
            sample_type_presence[t] += 1

    sample_type_percent = {
        t: round(100.0 * sample_type_presence.get(t, 0) / max(n_total, 1), 2)
        for t in NODE_TYPES
    }

    # ── Warnings ──────────────────────────────────────────────────────────────
    warnings: List[str] = []

    for t in NODE_TYPES:
        pct = sample_type_percent[t]
        if pct < 5.0:
            msg = f"NodeType '{t}' appears in only {pct:.1f}% of samples (threshold: 5%)"
            logger.warning(msg)
            warnings.append(msg)

    if small_diagram > 0:
        msg = f"{small_diagram} diagram(s) have fewer than 2 nodes"
        logger.warning(msg)
        warnings.append(msg)

    if empty_explanation > 0 or missing_explanation > 0:
        msg = f"{empty_explanation + missing_explanation} sample(s) have empty/missing explanations"
        logger.warning(msg)
        warnings.append(msg)

    if missing_png_file > 0:
        msg = f"{missing_png_file} PNG file(s) are missing from disk"
        logger.warning(msg)
        warnings.append(msg)

    # ── Summary ───────────────────────────────────────────────────────────────
    summary = {
        "manifest":              str(manifest_path),
        "n_total":               n_total,
        "n_valid":               valid_count,
        "n_missing_image":       missing_image,
        "n_missing_architecture":missing_architecture,
        "n_missing_explanation": missing_explanation,
        "n_empty_explanation":   empty_explanation,
        "n_missing_png_file":    missing_png_file,
        "n_small_diagram":       small_diagram,
        "avg_nodes_per_sample":  round(avg_nodes, 2),
        "avg_edges_per_sample":  round(avg_edges, 2),
        "node_type_distribution":node_type_distribution,
        "sample_type_presence_pct": sample_type_percent,
        "warnings":              warnings,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Summary saved → %s", output_path)

    # ── Print report ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ArchitectAI Dataset Validation Report")
    print("=" * 60)
    print(f"  Total entries      : {n_total}")
    print(f"  Valid entries      : {valid_count}")
    print(f"  Avg nodes/sample   : {avg_nodes:.2f}")
    print(f"  Avg edges/sample   : {avg_edges:.2f}")
    print()
    print("  NodeType presence in samples (%):")
    for t, pct in sample_type_percent.items():
        bar   = "█" * int(pct / 2)
        flag  = " ⚠" if pct < 5.0 else ""
        print(f"    {t:<10}  {pct:5.1f}%  {bar}{flag}")
    if warnings:
        print()
        print(f"  ⚠ {len(warnings)} warning(s):")
        for w in warnings:
            print(f"    - {w}")
    else:
        print("  ✓ No warnings — dataset looks healthy.")
    print("=" * 60 + "\n")

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validate the ArchitectAI synthetic dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data", default="data/synthetic",
        help="Root directory containing dataset.jsonl.",
    )
    p.add_argument(
        "--output", default="reports/dataset_summary.json",
        help="Path to save the summary JSON report.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args  = _parse_args()
    manifest = Path(args.data) / "dataset.jsonl"
    validate_dataset(manifest_path=manifest, output_path=Path(args.output))
