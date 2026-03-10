"""run_dataset_validation.py — Automated dataset validation for ArchitectAI.

Calls validate_dataset.py via subprocess, captures its JSON output, prints
a formatted statistics table, warns on imbalance, and saves the summary to
reports/dataset_summary.json.

Usage
-----
    python scripts/run_dataset_validation.py [OPTIONS]

Options
-------
    --data     Path to dataset directory   (default: data/synthetic)
    --output   Path to save summary JSON   (default: reports/dataset_summary.json)
    --imbalance-threshold
               Warn when a node type is under-represented below this fraction
               of the expected uniform share (default: 0.5)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SEP  = "=" * 60
_DASH = "-" * 60


def _run_validator(data_dir: str, output: str) -> int:
    """Invoke validate_dataset.py and return its exit code."""
    cmd = [
        sys.executable,
        "scripts/validate_dataset.py",
        "--data",   data_dir,
        "--output", output,
    ]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, text=True)
    return result.returncode


def _print_summary(summary: dict, threshold: float) -> None:
    n_total   = summary.get("n_total",   "N/A")
    n_valid   = summary.get("n_valid",   "N/A")
    n_invalid = summary.get("n_invalid", "N/A")
    avg_nodes = summary.get("avg_nodes_per_arch", "N/A")
    avg_edges = summary.get("avg_edges_per_arch", "N/A")

    print(_SEP)
    print("  Dataset Validation Summary")
    print(_SEP)
    print(f"  {'Total samples':<30} {n_total}")
    print(f"  {'Valid samples':<30} {n_valid}")
    print(f"  {'Invalid samples':<30} {n_invalid}")
    if isinstance(avg_nodes, (int, float)):
        print(f"  {'Avg nodes / architecture':<30} {avg_nodes:.2f}")
    if isinstance(avg_edges, (int, float)):
        print(f"  {'Avg edges / architecture':<30} {avg_edges:.2f}")

    node_dist: dict = summary.get("node_type_distribution", {})
    if node_dist:
        print(_DASH)
        print("  Node Type Distribution")
        print(_DASH)
        print(f"  {'Type':<25} {'Count':>8}  {'Share':>7}")
        total_nodes = sum(node_dist.values())
        n_types     = len(node_dist)
        uniform_share = 1 / n_types if n_types else 1
        has_warning = False

        for ntype, count in sorted(node_dist.items(), key=lambda x: -x[1]):
            share = count / total_nodes if total_nodes else 0
            ratio = share / uniform_share if uniform_share else 1
            flag  = "  ⚠ UNDER-REPRESENTED" if ratio < threshold else ""
            if flag:
                has_warning = True
            print(f"  {ntype:<25} {count:>8}  {share:>6.1%}{flag}")

        if has_warning:
            print(_DASH)
            print("  WARNING: Some node types are under-represented.")
            print("  Consider re-generating with --num-samples or adjusting weights.")

    print(_SEP)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    print(_SEP)
    print("  ArchitectAI — Dataset Validation")
    print(_SEP)

    rc = _run_validator(args.data, args.output)

    out_path = Path(args.output)
    if not out_path.exists():
        print(f"ERROR: validate_dataset.py did not produce {out_path}")
        sys.exit(1)

    summary: dict = json.loads(out_path.read_text(encoding="utf-8"))
    _print_summary(summary, args.imbalance_threshold)

    print(f"\nSummary saved → {out_path}")
    sys.exit(rc)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run ArchitectAI dataset validation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data",   default="data/synthetic")
    p.add_argument("--output", default="reports/dataset_summary.json")
    p.add_argument("--imbalance-threshold", type=float, default=0.5,
                   dest="imbalance_threshold",
                   help="Warn when a type's share < threshold × (1/n_types).")
    return p.parse_args()


if __name__ == "__main__":
    main()
