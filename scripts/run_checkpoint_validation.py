"""run_checkpoint_validation.py — Model checkpoint validator for ArchitectAI.

Delegates to check_model_checkpoints.py via subprocess, then loads and
pretty-prints the resulting JSON report.

Usage
-----
    python scripts/run_checkpoint_validation.py [OPTIONS]

Options
-------
    --convnext   Path to ConvNeXt checkpoint   (default: checkpoints/convnext/convnext_best.pt)
    --qwen-lora  Path to LoRA adapter directory (default: checkpoints/qwen_lora/lora_adapter)
    --output     Report path                   (default: reports/checkpoints_report.json)
    --no-load    Skip model weight loading (size/existence only)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_SEP  = "=" * 60
_DASH = "-" * 60


def _run_checker(convnext: str, qwen_lora: str, output: str, no_load: bool) -> int:
    cmd = [
        sys.executable,
        "scripts/check_model_checkpoints.py",
        "--convnext",  convnext,
        "--qwen-lora", qwen_lora,
        "--output",    output,
    ]
    if no_load:
        cmd.append("--no-load")
    print(f"  Running: {' '.join(cmd)}\n")
    return subprocess.run(cmd, text=True).returncode


def _print_report(report: dict) -> None:
    ckpts = report.get("checkpoints", {})

    print(_SEP)
    print("  Checkpoint Validation Report")
    print(_SEP)
    print(f"  CUDA available: {report.get('cuda_available', 'N/A')}")
    print(_DASH)

    for name, rec in ckpts.items():
        status = rec.get("status", "unknown")
        icon   = "✓" if status == "ok" else ("⚠" if status in {"exists", "partial"} else "✗")

        print(f"\n  {icon}  {name.upper()}")
        print(f"       Path   : {rec.get('path', 'N/A')}")
        print(f"       Status : {status}")
        print(f"       Size   : {rec.get('mb', 0):.1f} MB")

        if "total_M" in rec:
            print(f"       Params : {rec['total_M']} M total / "
                  f"{rec.get('trainable_M', '?')} M trainable")

        if name == "qwen_lora":
            cfg = rec.get("adapter_config", {})
            if cfg:
                print(f"       LoRA r : {cfg.get('r', 'N/A')}  "
                      f"alpha={cfg.get('lora_alpha', 'N/A')}  "
                      f"targets={cfg.get('target_modules', [])}")

        if rec.get("error"):
            print(f"       Error  : {rec['error']}")

        files = rec.get("files", {})
        if files:
            print("       Files  :", end="")
            parts = []
            for fname, info in files.items():
                icon2 = "✓" if info.get("exists") else "✗"
                parts.append(f"{icon2} {fname}")
            print("  " + "  |  ".join(parts))

    print()
    print(_SEP)

    # Overall verdict
    any_bad = any(
        r.get("status") in {"missing", "corrupted"}
        for r in ckpts.values()
    )
    verdict = "ALL CHECKPOINTS OK ✓" if not any_bad else "SOME CHECKPOINTS MISSING / CORRUPTED ✗"
    print(f"  {verdict}")
    print(_SEP)


def main() -> None:
    args = _parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    print(_SEP)
    print("  ArchitectAI — Checkpoint Validation")
    print(_SEP)

    rc = _run_checker(args.convnext, args.qwen_lora, args.output, args.no_load)

    out_path = Path(args.output)
    if not out_path.exists():
        print(f"ERROR: check_model_checkpoints.py did not produce {out_path}")
        sys.exit(1)

    report = json.loads(out_path.read_text(encoding="utf-8"))
    _print_report(report)

    print(f"\nReport → {out_path}")
    sys.exit(rc)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validate ArchitectAI model checkpoints.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--convnext",  default="checkpoints/convnext/convnext_best.pt")
    p.add_argument("--qwen-lora", default="checkpoints/qwen_lora/lora_adapter")
    p.add_argument("--output",    default="reports/checkpoints_report.json")
    p.add_argument("--no-load",   action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    main()
