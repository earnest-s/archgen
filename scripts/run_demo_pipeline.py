"""run_demo_pipeline.py — Demo execution runner for ArchitectAI.

Loads prompts from docs/demo_prompts.json and drives the full pipeline via the
running API server:
    POST /generate  → architecture JSON + diagram PNG
    POST /explain   → structured 4-section explanation

For each prompt, saves:
    docs/demo_outputs/<n>_<pattern>_architecture.json
    docs/demo_outputs/<n>_<pattern>_diagram.png
    docs/demo_outputs/<n>_<pattern>_explanation.txt

Prints per-prompt progress and a final summary table.

Usage
-----
    python scripts/run_demo_pipeline.py [OPTIONS]

Options
-------
    --url     Base API URL                (default: http://localhost:8000)
    --input   Path to demo_prompts.json   (default: docs/demo_prompts.json)
    --output  Output directory            (default: docs/demo_outputs)
    --ids     Comma-separated prompt IDs  (default: all)
    --timeout Request timeout seconds     (default: 60)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_SEP  = "=" * 60
_DASH = "-" * 60


def main() -> None:
    args = _parse_args()

    # Generate demo prompts if not present
    prompts_path = Path(args.input)
    if not prompts_path.exists():
        print(f"  Prompts file not found: {prompts_path}")
        print("  Generating prompts with scripts/generate_demo_prompts.py …")
        gen_rc = subprocess.run(
            [sys.executable, "scripts/generate_demo_prompts.py",
             "--output", str(prompts_path.with_suffix(""))],
            text=True,
        ).returncode
        if gen_rc != 0 or not prompts_path.exists():
            print("  ERROR: Could not generate demo prompts file.")
            sys.exit(1)

    Path(args.output).mkdir(parents=True, exist_ok=True)

    print(_SEP)
    print("  ArchitectAI — Demo Pipeline Runner")
    print(_SEP)
    print(f"  API URL : {args.url}")
    print(f"  Prompts : {args.input}")
    print(f"  Output  : {args.output}")
    if args.ids:
        print(f"  IDs     : {args.ids}")
    print(_SEP + "\n")

    cmd = [
        sys.executable, "scripts/run_demo.py",
        "--url",     args.url,
        "--input",   args.input,
        "--output",  args.output,
        "--timeout", str(args.timeout),
    ]
    if args.ids:
        cmd += ["--ids", args.ids]

    print(f"  Running: {' '.join(cmd)}\n")
    rc = subprocess.run(cmd, text=True).returncode

    # List outputs produced
    out_dir = Path(args.output)
    produced = sorted(out_dir.glob("*"))
    if produced:
        print()
        print(_SEP)
        print(f"  Outputs in {out_dir}/")
        print(_DASH)
        for p in produced:
            if p.is_file():
                mb = p.stat().st_size / 1_048_576
                print(f"  {'✓':<3} {p.name:<50} {mb:.3f} MB")
        print(_SEP)

    sys.exit(rc)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run ArchitectAI demo pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--url",     default="http://localhost:8000")
    p.add_argument("--input",   default="docs/demo_prompts.json")
    p.add_argument("--output",  default="docs/demo_outputs")
    p.add_argument("--ids",     default="",
                   help="Comma-separated prompt IDs (e.g. '1,3,5'). Default: all.")
    p.add_argument("--timeout", type=int, default=60)
    return p.parse_args()


if __name__ == "__main__":
    main()
