"""run_experiments.py — Automated experiment runner for ArchitectAI.

Executes all evaluation experiments in the correct order and saves a
consolidated JSON summary to reports/experiment_results.json.

Stages
------
  1  validate_dataset.py      — data quality gate
  2  generate_dataset.py      — synthesise 10 000 samples if missing
  3  train_convnext.py        — train vision encoder
  4  lora_train.py            — fine-tune Qwen LoRA adapter
  5  eval_models.py           — evaluation metrics (F1, BLEU, ROUGE)
  6  run_ablation.py          — text-only vs text+vision ablation
  7  export_examples.py       — export 20 demo artifacts
  8  visualize_results.py     — generate charts from reports

Usage
-----
    python scripts/run_experiments.py [OPTIONS]

Options
-------
    --num-samples N    Samples to generate if dataset is missing (default 10000)
    --skip-train       Skip ConvNeXt + Qwen training stages
    --skip-generate    Skip dataset generation
    --dry-run          Print commands without executing them
    --output           Path to summary JSON (default reports/experiment_results.json)
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging — dual console + file
# ---------------------------------------------------------------------------

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = REPORTS_DIR / "experiment_runner.log"

_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
_fh  = logging.FileHandler(LOG_PATH, encoding="utf-8")
_ch  = logging.StreamHandler(sys.stdout)
_fh.setFormatter(_fmt)
_ch.setFormatter(_fmt)

logger = logging.getLogger("experiments")
logger.setLevel(logging.DEBUG)
logger.addHandler(_fh)
logger.addHandler(_ch)

SEP = "=" * 70

# ---------------------------------------------------------------------------
# Stage runner
# ---------------------------------------------------------------------------


def _banner(title: str) -> None:
    logger.info(SEP)
    logger.info("  %s", title)
    logger.info(SEP)


def run_stage(
    name: str,
    cmd: list[str],
    *,
    dry_run: bool = False,
    allow_failure: bool = False,
) -> dict:
    """Execute *cmd* and return a result dict with status and elapsed time."""
    _banner(f"Experiment Stage: {name}")
    logger.info("Command: %s", " ".join(cmd))

    if dry_run:
        logger.info("[DRY-RUN] Skipping execution.")
        return {"stage": name, "status": "dry-run", "elapsed_s": 0.0, "returncode": 0}

    start = time.perf_counter()
    try:
        result = subprocess.run(cmd, check=False, text=True,
                                stdout=sys.stdout, stderr=sys.stderr)
        elapsed = round(time.perf_counter() - start, 2)
        if result.returncode == 0:
            logger.info("✓ '%s' completed in %.1fs", name, elapsed)
            status = "ok"
        else:
            msg = f"Stage '{name}' exited with code {result.returncode}"
            if allow_failure:
                logger.warning("⚠  %s", msg)
                status = "warning"
            else:
                logger.error("✗  %s", msg)
                status = "failed"
        return {"stage": name, "status": status, "elapsed_s": elapsed,
                "returncode": result.returncode}
    except FileNotFoundError as exc:
        elapsed = round(time.perf_counter() - start, 2)
        logger.error("✗ '%s' — command not found: %s", name, exc)
        return {"stage": name, "status": "failed", "elapsed_s": elapsed, "error": str(exc)}


# ---------------------------------------------------------------------------
# Stage definitions
# ---------------------------------------------------------------------------


def build_stages(args: argparse.Namespace) -> list[tuple[str, list[str], bool, bool]]:
    """Return list of (name, cmd, skip, allow_failure) tuples."""
    py = sys.executable
    data_dir   = "data/synthetic"
    convnext   = "checkpoints/convnext/convnext_best.pt"
    jsonl      = f"{data_dir}/dataset.jsonl"

    return [
        (
            "1 — Validate Dataset",
            [py, "scripts/validate_dataset.py",
             "--data", data_dir, "--output", "reports/dataset_summary.json"],
            False, True,   # always run; allow failure if data missing
        ),
        (
            "2 — Generate Dataset",
            [py, "scripts/generate_dataset.py",
             "--num-samples", str(args.num_samples),
             "--output-dir", data_dir, "--seed", "42"],
            args.skip_generate, False,
        ),
        (
            "3 — Train ConvNeXt Vision Encoder",
            [py, "backend/training/vision/train_convnext.py",
             "--data", data_dir, "--out", "checkpoints/convnext",
             "--epochs", "30", "--bs", "32", "--patience", "5"],
            args.skip_train, False,
        ),
        (
            "4 — Fine-tune Qwen LoRA Adapter",
            [py, "backend/training/qwen/lora_train.py",
             "--data", jsonl, "--out", "checkpoints/qwen_lora",
             "--epochs", "3", "--bs", "2", "--accum", "8"],
            args.skip_train, False,
        ),
        (
            "5 — Evaluate Models",
            [py, "scripts/eval_models.py",
             "--data", data_dir, "--convnext", convnext,
             "--output", "reports/evaluation.json",
             "--max-samples", "200", "--compare-explainers"],
            False, True,
        ),
        (
            "6 — Ablation Experiment",
            [py, "scripts/run_ablation.py",
             "--data", data_dir, "--convnext", convnext,
             "--output", "reports/ablation_results.json",
             "--max-samples", "50"],
            False, True,
        ),
        (
            "7 — Export Demo Examples",
            [py, "scripts/export_examples.py",
             "--data", data_dir, "--output", "docs/examples",
             "--n", "20", "--seed", "42"],
            False, True,
        ),
        (
            "8 — Generate Visualizations",
            [py, "scripts/visualize_results.py",
             "--eval", "reports/evaluation.json",
             "--ablation", "reports/ablation_results.json",
             "--output", "reports/figures"],
            False, True,
        ),
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = _parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    run_id    = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    wall_t0   = time.perf_counter()

    _banner("ArchitectAI — Experiment Runner")
    logger.info("Run ID   : %s", run_id)
    logger.info("Log file : %s", LOG_PATH.resolve())
    logger.info("Dry-run  : %s", args.dry_run)

    stages  = build_stages(args)
    results = []
    aborted = False

    for name, cmd, skip, allow_failure in stages:
        if skip:
            logger.info("Skipping: %s", name)
            results.append({"stage": name, "status": "skipped", "elapsed_s": 0.0})
            continue

        rec = run_stage(name, cmd, dry_run=args.dry_run, allow_failure=allow_failure)
        results.append(rec)

        if rec["status"] == "failed":
            logger.error("Pipeline aborted at stage: %s", name)
            aborted = True
            break

    wall_elapsed = round(time.perf_counter() - wall_t0, 2)

    # ── Summary ───────────────────────────────────────────────────────────────
    _banner("Experiment Summary")
    col_w = max(len(r["stage"]) for r in results) + 2
    for r in results:
        icon = {"ok": "✓", "failed": "✗", "skipped": "—",
                "warning": "⚠", "dry-run": "⊡"}.get(r["status"], "?")
        logger.info("  %s  %-*s  %s  %.1fs",
                    icon, col_w, r["stage"], r["status"].upper(), r["elapsed_s"])
    logger.info(SEP)
    logger.info("Wall time: %.1fs | Status: %s",
                wall_elapsed, "ABORTED" if aborted else "COMPLETE")

    summary = {
        "run_id":          run_id,
        "wall_elapsed_s":  wall_elapsed,
        "aborted":         aborted,
        "dry_run":         args.dry_run,
        "stages":          results,
    }
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Summary → %s", output_path.resolve())

    sys.exit(1 if aborted else 0)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ArchitectAI automated experiment runner.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--num-samples", type=int, default=10_000)
    p.add_argument("--skip-train",    action="store_true",
                   help="Skip ConvNeXt + Qwen training stages.")
    p.add_argument("--skip-generate", action="store_true",
                   help="Skip dataset generation.")
    p.add_argument("--dry-run",       action="store_true",
                   help="Print commands without executing.")
    p.add_argument("--output", default="reports/experiment_results.json")
    return p.parse_args()


if __name__ == "__main__":
    main()
