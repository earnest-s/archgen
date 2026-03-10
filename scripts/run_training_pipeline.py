"""run_training_pipeline.py — Full ML training orchestrator for ArchitectAI.

Runs the complete pipeline sequentially:

  Stage 1  validate_dataset.py   — check existing data quality
  Stage 2  generate_dataset.py   — generate / top-up synthetic samples
  Stage 3  train_convnext.py     — train vision encoder
  Stage 4  lora_train.py         — fine-tune Qwen LoRA adapter
  Stage 5  eval_models.py        — evaluate both models
  Stage 6  run_ablation.py       — text-only vs text+vision ablation

Progress and timing are written to both stdout and
reports/training_pipeline.log.

Usage
-----
    python scripts/run_training_pipeline.py [OPTIONS]

Options
-------
    --skip-generate   Skip dataset generation (use existing data)
    --skip-convnext   Skip ConvNeXt training
    --skip-qwen       Skip Qwen LoRA training
    --skip-eval       Skip evaluation
    --skip-ablation   Skip ablation experiment
    --num-samples N   Samples to generate (default: 10000)
    --dry-run         Print commands but do not execute them
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Setup logging — console + rotating file handler
# ---------------------------------------------------------------------------

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = REPORTS_DIR / "training_pipeline.log"

_fmt = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

_file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
_file_handler.setFormatter(_fmt)

_console_handler = logging.StreamHandler(sys.stdout)
_console_handler.setFormatter(_fmt)

logger = logging.getLogger("pipeline")
logger.setLevel(logging.DEBUG)
logger.addHandler(_file_handler)
logger.addHandler(_console_handler)

# ---------------------------------------------------------------------------
# Stage runner
# ---------------------------------------------------------------------------

SEPARATOR = "=" * 68


def _banner(title: str) -> None:
    logger.info(SEPARATOR)
    logger.info("  %s", title)
    logger.info(SEPARATOR)


def run_stage(
    name: str,
    cmd: list[str],
    *,
    dry_run: bool = False,
    allow_failure: bool = False,
) -> tuple[bool, float]:
    """Run *cmd* as a subprocess.

    Args:
        name:          Human-readable stage name (used in log messages).
        cmd:           Command + arguments list.
        dry_run:       If True, print the command without executing it.
        allow_failure: If True, a non-zero exit code is logged as a warning
                       instead of causing the pipeline to abort.

    Returns:
        ``(success: bool, elapsed_seconds: float)``
    """
    _banner(f"Stage: {name}")
    cmd_str = " ".join(cmd)
    logger.info("Command: %s", cmd_str)

    if dry_run:
        logger.info("[DRY-RUN] Skipping execution.")
        return True, 0.0

    start = time.perf_counter()
    try:
        result = subprocess.run(
            cmd,
            check=False,          # we handle return code ourselves
            stdout=sys.stdout,    # stream to terminal + captured by tee-logging
            stderr=sys.stderr,
            text=True,
        )
        elapsed = time.perf_counter() - start

        if result.returncode == 0:
            logger.info("✓ Stage '%s' completed in %.1fs", name, elapsed)
            return True, elapsed
        else:
            msg = f"Stage '{name}' exited with code {result.returncode} (elapsed {elapsed:.1f}s)"
            if allow_failure:
                logger.warning("⚠  %s", msg)
                return False, elapsed
            else:
                logger.error("✗  %s", msg)
                return False, elapsed

    except FileNotFoundError as exc:
        elapsed = time.perf_counter() - start
        logger.error("✗ Stage '%s' — command not found: %s", name, exc)
        return False, elapsed


# ---------------------------------------------------------------------------
# Pipeline definition
# ---------------------------------------------------------------------------

def build_stages(args: argparse.Namespace) -> list[tuple[str, list[str], bool, bool]]:
    """Return ordered list of (name, cmd, skip, allow_failure) tuples."""
    python = sys.executable

    stages: list[tuple[str, list[str], bool, bool]] = [
        (
            "Validate Dataset",
            [python, "scripts/validate_dataset.py",
             "--data", "data/synthetic",
             "--output", "reports/dataset_summary.json"],
            False,       # never skip — always validate first
            True,        # allow failure if no dataset exists yet
        ),
        (
            "Generate Dataset",
            [python, "scripts/generate_dataset.py",
             "--num-samples", str(args.num_samples),
             "--output-dir", "data/synthetic",
             "--seed", "42"],
            args.skip_generate,
            False,
        ),
        (
            "Train ConvNeXt Vision Encoder",
            [python, "backend/training/vision/train_convnext.py",
             "--data",    "data/synthetic",
             "--out",     "checkpoints/convnext",
             "--epochs",  "30",
             "--bs",      "32",
             "--patience", "5"],
            args.skip_convnext,
            False,
        ),
        (
            "Fine-tune Qwen LoRA Adapter",
            [python, "backend/training/qwen/lora_train.py",
             "--data",   "data/synthetic/dataset.jsonl",
             "--out",    "checkpoints/qwen_lora",
             "--epochs", "3",
             "--bs",     "2",
             "--accum",  "8"],
            args.skip_qwen,
            False,
        ),
        (
            "Evaluate Models",
            [python, "scripts/eval_models.py",
             "--data",        "data/synthetic",
             "--convnext",    "checkpoints/convnext/convnext_best.pt",
             "--output",      "reports/evaluation.json",
             "--max-samples", "200"],
            args.skip_eval,
            True,   # allow failure if checkpoints don't exist yet
        ),
        (
            "Ablation Experiment",
            [python, "scripts/run_ablation.py",
             "--data",        "data/synthetic",
             "--convnext",    "checkpoints/convnext/convnext_best.pt",
             "--output",      "reports/ablation_results.json",
             "--max-samples", "50"],
            args.skip_ablation,
            True,
        ),
    ]

    return stages


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    _banner("ArchitectAI — Training Pipeline")
    logger.info("Log file  : %s", LOG_PATH.resolve())
    logger.info("Dry-run   : %s", args.dry_run)
    logger.info("Python    : %s", sys.executable)

    stages     = build_stages(args)
    results:   list[dict] = []
    pipeline_start = time.perf_counter()
    any_failure    = False

    for stage_name, cmd, skip, allow_failure in stages:
        if skip:
            logger.info("Skipping stage: %s", stage_name)
            results.append({"stage": stage_name, "status": "skipped", "elapsed_s": 0.0})
            continue

        success, elapsed = run_stage(
            stage_name, cmd,
            dry_run=args.dry_run,
            allow_failure=allow_failure,
        )
        results.append({
            "stage":     stage_name,
            "status":    "ok" if success else ("warning" if allow_failure else "failed"),
            "elapsed_s": round(elapsed, 2),
        })
        if not success and not allow_failure:
            logger.error("Pipeline aborted after stage '%s'.", stage_name)
            any_failure = True
            break

    # ── Final summary ─────────────────────────────────────────────────────────
    total_elapsed = time.perf_counter() - pipeline_start
    _banner("Pipeline Summary")
    col_w = max(len(r["stage"]) for r in results) + 2
    for r in results:
        icon = {"ok": "✓", "failed": "✗", "skipped": "—", "warning": "⚠"}.get(r["status"], "?")
        logger.info(
            "  %s  %-*s  %s  %.1fs",
            icon, col_w, r["stage"], r["status"].upper(), r["elapsed_s"],
        )
    logger.info(SEPARATOR)
    logger.info("Total wall-clock time: %.1fs", total_elapsed)
    logger.info("Log saved to: %s", LOG_PATH.resolve())

    # Write JSON summary alongside log.
    import json
    summary = {
        "total_elapsed_s": round(total_elapsed, 2),
        "stages": results,
    }
    summary_path = REPORTS_DIR / "pipeline_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Summary JSON: %s", summary_path.resolve())

    sys.exit(1 if any_failure else 0)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ArchitectAI full ML training pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--skip-generate",  action="store_true", help="Skip dataset generation.")
    p.add_argument("--skip-convnext",  action="store_true", help="Skip ConvNeXt training.")
    p.add_argument("--skip-qwen",      action="store_true", help="Skip Qwen LoRA fine-tuning.")
    p.add_argument("--skip-eval",      action="store_true", help="Skip model evaluation.")
    p.add_argument("--skip-ablation",  action="store_true", help="Skip ablation study.")
    p.add_argument(
        "--num-samples", type=int, default=10_000,
        help="Number of synthetic samples to generate.",
    )
    p.add_argument("--dry-run", action="store_true",
                   help="Print commands without executing them.")
    return p.parse_args()


if __name__ == "__main__":
    main()
