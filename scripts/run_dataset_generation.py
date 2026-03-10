"""run_dataset_generation.py — Dataset generation command wrapper for ArchitectAI.

Wraps generate_dataset.py with CLI argument forwarding, progress logging, and
metadata persistence to reports/dataset_generation.json.

Usage
-----
    python scripts/run_dataset_generation.py [OPTIONS]

Options
-------
    --num-samples  Number of samples to generate   (default: 10000)
    --seed         Random seed for reproducibility (default: 42)
    --output-dir   Output directory                (default: data/synthetic)
    --output-meta  Metadata report path            (default: reports/dataset_generation.json)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

_SEP = "=" * 60


def main() -> None:
    args = _parse_args()
    Path(args.output_meta).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(_SEP)
    print("  ArchitectAI — Dataset Generation")
    print(_SEP)
    print(f"  Samples   : {args.num_samples:,}")
    print(f"  Seed      : {args.seed}")
    print(f"  Output dir: {args.output_dir}")
    print(_SEP)

    cmd = [
        sys.executable,
        "scripts/generate_dataset.py",
        "--num-samples", str(args.num_samples),
        "--seed",        str(args.seed),
        "--output-dir",  args.output_dir,
    ]
    print(f"  Running: {' '.join(cmd)}\n")

    wall_start = time.perf_counter()
    result = subprocess.run(cmd, text=True)
    wall_elapsed = time.perf_counter() - wall_start

    rc = result.returncode

    # ── Count generated diagrams ──────────────────────────────────────────────
    out_dir   = Path(args.output_dir)
    jsonl     = out_dir / "dataset.jsonl"
    n_created = 0
    if jsonl.exists():
        with jsonl.open("r", encoding="utf-8") as f:
            n_created = sum(1 for line in f if line.strip())

    png_count = len(list(out_dir.glob("*.png")))

    print()
    print(_SEP)
    print("  Generation Complete")
    print(_SEP)
    print(f"  Exit code         : {rc}")
    print(f"  Wall time         : {wall_elapsed:.1f}s")
    print(f"  JSONL records     : {n_created:,}")
    print(f"  PNG diagrams found: {png_count:,}")
    print(_SEP)

    # ── Save metadata ─────────────────────────────────────────────────────────
    meta = {
        "num_samples_requested": args.num_samples,
        "seed":                  args.seed,
        "output_dir":            args.output_dir,
        "wall_elapsed_s":        round(wall_elapsed, 2),
        "exit_code":             rc,
        "jsonl_records":         n_created,
        "png_diagrams":          png_count,
        "status":                "ok" if rc == 0 else "failed",
    }
    out = Path(args.output_meta)
    out.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nMetadata saved → {out}")

    sys.exit(rc)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate ArchitectAI training dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--num-samples",  type=int, default=10_000)
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--output-dir",   default="data/synthetic")
    p.add_argument("--output-meta",  default="reports/dataset_generation.json")
    return p.parse_args()


if __name__ == "__main__":
    main()
