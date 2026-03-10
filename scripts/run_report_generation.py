"""run_report_generation.py — Final project report generator for ArchitectAI.

Loads experiment result JSON files from reports/ and delegates to
generate_report.py to produce reports/project_report.md, then prints
a summary of the sections written.

Source files consumed (all optional):
    reports/dataset_summary.json
    reports/evaluation.json
    reports/ablation_results.json
    reports/pipeline_profile.json
    reports/api_stress_test.json

Usage
-----
    python scripts/run_report_generation.py [OPTIONS]

Options
-------
    --reports-dir   Directory containing JSON report files (default: reports)
    --output        Output Markdown path                  (default: reports/project_report.md)
    --title         Report title
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_SEP  = "=" * 60
_DASH = "-" * 60

_SOURCE_FILES = {
    "dataset_summary.json":    "Dataset statistics",
    "evaluation.json":         "Vision model + explanation metrics",
    "ablation_results.json":   "Ablation study (text-only vs text+vision)",
    "pipeline_profile.json":   "Pipeline latency profile",
    "api_stress_test.json":    "API stress test results",
}


def _check_sources(reports_dir: Path) -> None:
    print(_DASH)
    print("  Source report files")
    print(_DASH)
    for fname, desc in _SOURCE_FILES.items():
        path = reports_dir / fname
        if path.exists():
            kb = path.stat().st_size / 1024
            print(f"  ✓  {fname:<35} {desc}  ({kb:.1f} KB)")
        else:
            print(f"  ⚠  {fname:<35} MISSING — section will note 'data unavailable'")
    print()


def _run_generator(args: argparse.Namespace) -> int:
    cmd = [
        sys.executable, "scripts/generate_report.py",
        "--reports-dir", args.reports_dir,
        "--output",      args.output,
        "--title",       args.title,
    ]
    print(f"  Running: {' '.join(cmd)}\n")
    return subprocess.run(cmd, text=True).returncode


def _print_report_summary(output_path: Path) -> None:
    if not output_path.exists():
        return
    text   = output_path.read_text(encoding="utf-8")
    lines  = text.splitlines()
    n_h2   = sum(1 for l in lines if l.startswith("## "))
    n_rows = sum(1 for l in lines if l.startswith("|") and "---" not in l)
    kb     = output_path.stat().st_size / 1024

    print(_DASH)
    print("  Report Summary")
    print(_DASH)
    print(f"  File   : {output_path}")
    print(f"  Size   : {kb:.1f} KB  ({len(lines)} lines)")
    print(f"  Sections (##): {n_h2}")
    print(f"  Table rows   : {n_rows}")

    print()
    print("  Sections written:")
    for line in lines:
        if line.startswith("## "):
            print(f"    • {line[3:]}")


def main() -> None:
    args = _parse_args()
    reports_dir = Path(args.reports_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(_SEP)
    print("  ArchitectAI — Project Report Generation")
    print(_SEP)

    _check_sources(reports_dir)

    rc = _run_generator(args)

    print()
    print(_SEP)
    if output_path.exists():
        _print_report_summary(output_path)
        print()
        print(f"  Report → {output_path}")
    else:
        print(f"  ERROR: generate_report.py did not produce {output_path}")
        rc = 1
    print(_SEP)

    sys.exit(rc)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate ArchitectAI final project report.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--reports-dir", default="reports")
    p.add_argument("--output",      default="reports/project_report.md")
    p.add_argument("--title",       default="ArchitectAI — Project Report")
    return p.parse_args()


if __name__ == "__main__":
    main()
