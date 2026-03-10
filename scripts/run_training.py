"""run_training.py — Sequential training pipeline executor for ArchitectAI.

Runs both model training stages in order:
  Stage 1 — ConvNeXt vision encoder  (backend/training/vision/train_convnext.py)
  Stage 2 — Qwen LoRA fine-tuning     (backend/training/qwen/lora_train.py)

Tracks per-stage and total wall time, verifies checkpoint creation, and saves
a summary to reports/training_summary.json.

Usage
-----
    python scripts/run_training.py [OPTIONS]

Options
-------
    --data          Dataset directory           (default: data/synthetic)
    --convnext-out  ConvNeXt checkpoint dir     (default: checkpoints/convnext)
    --qwen-out      Qwen LoRA output dir        (default: checkpoints/qwen_lora)
    --epochs-conv   ConvNeXt training epochs    (default: 30)
    --bs-conv       ConvNeXt batch size         (default: 32)
    --patience      ConvNeXt early-stop patience(default: 5)
    --epochs-qwen   Qwen LoRA epochs            (default: 3)
    --bs-qwen       Qwen LoRA batch size        (default: 2)
    --accum         Qwen LoRA gradient accum.   (default: 8)
    --skip-convnext Skip Stage 1
    --skip-qwen     Skip Stage 2
    --output        Report path                 (default: reports/training_summary.json)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

_SEP  = "=" * 60
_DASH = "-" * 60


# ---------------------------------------------------------------------------
# Stage runner
# ---------------------------------------------------------------------------

def _run_stage(label: str, cmd: list[str]) -> tuple[int, float]:
    print()
    print(_DASH)
    print(f"  Stage: {label}")
    print(f"  Cmd  : {' '.join(cmd)}")
    print(_DASH)
    t0     = time.perf_counter()
    result = subprocess.run(cmd, text=True)
    elapsed = time.perf_counter() - t0
    status  = "ok" if result.returncode == 0 else "failed"
    print(f"\n  → {label}: {status.upper()} ({elapsed:.1f}s)")
    return result.returncode, elapsed


# ---------------------------------------------------------------------------
# Checkpoint presence check
# ---------------------------------------------------------------------------

def _check_checkpoint(path: Path) -> dict:
    if path.is_file():
        return {"exists": True, "mb": round(path.stat().st_size / 1_048_576, 2)}
    if path.is_dir():
        files = list(path.rglob("*"))
        total = sum(f.stat().st_size for f in files if f.is_file())
        return {"exists": True, "mb": round(total / 1_048_576, 2),
                "n_files": sum(1 for f in files if f.is_file())}
    return {"exists": False}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    print(_SEP)
    print("  ArchitectAI — Training Pipeline")
    print(_SEP)

    stages: list[dict] = []
    wall_start = time.perf_counter()

    # ── Stage 1: ConvNeXt ────────────────────────────────────────────────────
    if not args.skip_convnext:
        cmd = [
            sys.executable,
            "backend/training/vision/train_convnext.py",
            "--data",    args.data,
            "--out",     args.convnext_out,
            "--epochs",  str(args.epochs_conv),
            "--bs",      str(args.bs_conv),
            "--patience",str(args.patience),
        ]
        rc, elapsed = _run_stage("ConvNeXt Vision Encoder", cmd)
        stages.append({
            "name": "convnext", "status": "ok" if rc == 0 else "failed",
            "elapsed_s": round(elapsed, 2), "returncode": rc,
        })
        if rc != 0:
            print("  ConvNeXt training failed — aborting pipeline.")
            _save_and_exit(stages, args, wall_start, aborted=True)
    else:
        print("  Skipping ConvNeXt training (--skip-convnext).")
        stages.append({"name": "convnext", "status": "skipped"})

    # ── Stage 2: Qwen LoRA ───────────────────────────────────────────────────
    if not args.skip_qwen:
        cmd = [
            sys.executable,
            "backend/training/qwen/lora_train.py",
            "--data",   f"{args.data}/dataset.jsonl",
            "--out",    args.qwen_out,
            "--epochs", str(args.epochs_qwen),
            "--bs",     str(args.bs_qwen),
            "--accum",  str(args.accum),
        ]
        rc, elapsed = _run_stage("Qwen LoRA Fine-Tuning", cmd)
        stages.append({
            "name": "qwen_lora", "status": "ok" if rc == 0 else "failed",
            "elapsed_s": round(elapsed, 2), "returncode": rc,
        })
    else:
        print("  Skipping Qwen LoRA training (--skip-qwen).")
        stages.append({"name": "qwen_lora", "status": "skipped"})

    _save_and_exit(stages, args, wall_start, aborted=False)


def _save_and_exit(stages: list[dict], args: argparse.Namespace,
                   wall_start: float, aborted: bool) -> None:
    wall_elapsed = time.perf_counter() - wall_start

    # ── Checkpoint inventory ─────────────────────────────────────────────────
    checkpoints = {
        "convnext_best_pt": _check_checkpoint(
            Path(args.convnext_out) / "convnext_best.pt"),
        "qwen_lora_adapter": _check_checkpoint(
            Path(args.qwen_out) / "lora_adapter"),
    }

    # ── Print summary ─────────────────────────────────────────────────────────
    print()
    print(_SEP)
    print("  Training Summary")
    print(_SEP)
    for s in stages:
        status = s.get("status", "?")
        elapsed = s.get("elapsed_s")
        time_str = f"{elapsed:.1f}s" if elapsed is not None else "—"
        icon = "✓" if status == "ok" else ("—" if status == "skipped" else "✗")
        print(f"  {icon}  {s['name']:<20} {status:<10} {time_str}")
    print(_DASH)
    print(f"  Wall time: {wall_elapsed:.1f}s")
    print(_DASH)
    print("  Checkpoints:")
    for name, info in checkpoints.items():
        exists = info.get("exists", False)
        mb     = info.get("mb", 0)
        icon   = "✓" if exists else "✗"
        print(f"    {icon}  {name:<30} {'%.1f MB' % mb if exists else 'NOT FOUND'}")
    print(_SEP)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    summary = {
        "aborted":         aborted,
        "wall_elapsed_s":  round(wall_elapsed, 2),
        "stages":          stages,
        "checkpoints":     checkpoints,
    }
    out = Path(args.output)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSummary saved → {out}")

    any_failed = any(s.get("status") == "failed" for s in stages)
    sys.exit(1 if (aborted or any_failed) else 0)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run ArchitectAI training pipeline (ConvNeXt + Qwen LoRA).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data",          default="data/synthetic")
    p.add_argument("--convnext-out",  default="checkpoints/convnext")
    p.add_argument("--qwen-out",      default="checkpoints/qwen_lora")
    p.add_argument("--epochs-conv",   type=int, default=30)
    p.add_argument("--bs-conv",       type=int, default=32)
    p.add_argument("--patience",      type=int, default=5)
    p.add_argument("--epochs-qwen",   type=int, default=3)
    p.add_argument("--bs-qwen",       type=int, default=2)
    p.add_argument("--accum",         type=int, default=8)
    p.add_argument("--skip-convnext", action="store_true")
    p.add_argument("--skip-qwen",     action="store_true")
    p.add_argument("--output",        default="reports/training_summary.json")
    return p.parse_args()


if __name__ == "__main__":
    main()
