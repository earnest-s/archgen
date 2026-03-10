"""run_demo.py — Run the ArchitectAI demo pipeline against the API.

For each prompt in docs/demo_prompts.json this script:
  1. POST /parse       → parsed architecture JSON
  2. POST /generate    → diagram PNG bytes (base64)
  3. POST /explain     → natural-language explanation

Outputs saved under --output directory (default: docs/demo_outputs/):
  <n>_architecture.json
  <n>_diagram.png
  <n>_explanation.txt

Usage
-----
    # Requires the API server to be running (uvicorn or docker-compose)
    python scripts/run_demo.py [OPTIONS]

Options
-------
    --url     Base URL of the running API  (default: http://localhost:8000)
    --input   Path to demo_prompts.json    (default: docs/demo_prompts.json)
    --output  Output directory             (default: docs/demo_outputs)
    --ids     Comma-separated prompt IDs to run (default: all)
    --timeout Request timeout in seconds   (default: 60)
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SEP = "-" * 60


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def _post(session, url: str, payload: dict, timeout: int) -> tuple[int, dict | bytes]:
    """Return (status_code, parsed_json_or_bytes)."""
    resp = session.post(url, json=payload, timeout=timeout)
    content_type = resp.headers.get("content-type", "")
    if "image" in content_type or "octet-stream" in content_type:
        return resp.status_code, resp.content
    try:
        return resp.status_code, resp.json()
    except Exception:
        return resp.status_code, {"raw": resp.text}


# ---------------------------------------------------------------------------
# Per-prompt pipeline
# ---------------------------------------------------------------------------


def run_prompt(
    session,
    base_url: str,
    entry: dict,
    output_dir: Path,
    timeout: int,
) -> dict:
    prompt_id    = entry["id"]
    prompt_text  = entry["prompt"]
    pattern      = entry["pattern"]
    prefix       = f"{prompt_id:02d}_{pattern}"

    logger.info(SEP)
    logger.info("[%d/%s] %s", prompt_id, pattern, entry.get("title", ""))
    logger.info("Prompt: %s…", prompt_text[:80])

    result: dict = {
        "id":      prompt_id,
        "pattern": pattern,
        "title":   entry.get("title"),
        "steps":   {},
    }

    # ── Step 1: Parse ────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    try:
        status, data = _post(
            session, f"{base_url}/parse",
            {"prompt": prompt_text},
            timeout,
        )
        elapsed = time.perf_counter() - t0
        if status == 200:
            arch = data  # architecture dict
            arch_path = output_dir / f"{prefix}_architecture.json"
            arch_path.write_text(json.dumps(arch, indent=2), encoding="utf-8")
            logger.info("  ✓ parse  (%.2fs) → %s", elapsed, arch_path.name)
            result["steps"]["parse"] = {"status": "ok", "elapsed_s": round(elapsed, 3)}
        else:
            logger.error("  ✗ parse  HTTP %d: %s", status, data)
            result["steps"]["parse"] = {"status": "error", "http": status}
            arch = None
    except Exception as exc:
        logger.error("  ✗ parse  exception: %s", exc)
        result["steps"]["parse"] = {"status": "exception", "error": str(exc)}
        arch = None

    # ── Step 2: Generate diagram ──────────────────────────────────────────────
    if arch is not None:
        t0 = time.perf_counter()
        try:
            status, data = _post(
                session, f"{base_url}/generate",
                {"architecture": arch},
                timeout,
            )
            elapsed = time.perf_counter() - t0
            if status == 200:
                # API may return base64 PNG inside JSON or raw bytes
                if isinstance(data, bytes):
                    png_bytes = data
                elif isinstance(data, dict) and "diagram" in data:
                    png_bytes = base64.b64decode(data["diagram"])
                else:
                    png_bytes = None

                if png_bytes:
                    png_path = output_dir / f"{prefix}_diagram.png"
                    png_path.write_bytes(png_bytes)
                    logger.info("  ✓ generate (%.2fs) → %s", elapsed, png_path.name)
                    result["steps"]["generate"] = {
                        "status": "ok", "elapsed_s": round(elapsed, 3),
                    }
                else:
                    logger.warning("  ⚠ generate returned unexpected shape; saving JSON")
                    (output_dir / f"{prefix}_diagram.json").write_text(
                        json.dumps(data, indent=2), encoding="utf-8")
                    result["steps"]["generate"] = {"status": "no_png", "elapsed_s": round(elapsed, 3)}
            else:
                logger.error("  ✗ generate HTTP %d: %s", status, data)
                result["steps"]["generate"] = {"status": "error", "http": status}
        except Exception as exc:
            logger.error("  ✗ generate exception: %s", exc)
            result["steps"]["generate"] = {"status": "exception", "error": str(exc)}
    else:
        result["steps"]["generate"] = {"status": "skipped"}

    # ── Step 3: Explain ───────────────────────────────────────────────────────
    if arch is not None:
        t0 = time.perf_counter()
        try:
            status, data = _post(
                session, f"{base_url}/explain",
                {"architecture": arch},
                timeout,
            )
            elapsed = time.perf_counter() - t0
            if status == 200:
                explanation = (
                    data.get("explanation") or
                    data.get("text") or
                    json.dumps(data, indent=2)
                )
                txt_path = output_dir / f"{prefix}_explanation.txt"
                txt_path.write_text(explanation, encoding="utf-8")
                logger.info("  ✓ explain  (%.2fs) → %s", elapsed, txt_path.name)
                result["steps"]["explain"] = {
                    "status": "ok", "elapsed_s": round(elapsed, 3),
                    "chars": len(explanation),
                }
            else:
                logger.error("  ✗ explain  HTTP %d: %s", status, data)
                result["steps"]["explain"] = {"status": "error", "http": status}
        except Exception as exc:
            logger.error("  ✗ explain  exception: %s", exc)
            result["steps"]["explain"] = {"status": "exception", "error": str(exc)}
    else:
        result["steps"]["explain"] = {"status": "skipped"}

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = _parse_args()

    # Check that requests is available
    try:
        import requests  # type: ignore
    except ImportError:
        logger.error("'requests' is not installed. Run: pip install requests")
        sys.exit(1)

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Prompts file not found: %s", input_path)
        logger.error("Run scripts/generate_demo_prompts.py first.")
        sys.exit(1)

    prompts: list[dict] = json.loads(input_path.read_text(encoding="utf-8"))

    # Filter by --ids if provided
    if args.ids:
        wanted = {int(x.strip()) for x in args.ids.split(",")}
        prompts = [p for p in prompts if p["id"] in wanted]
        if not prompts:
            logger.error("No prompts matched --ids=%s", args.ids)
            sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_url = args.url.rstrip("/")
    timeout  = args.timeout

    logger.info("Base URL : %s", base_url)
    logger.info("Prompts  : %d", len(prompts))
    logger.info("Output   : %s", output_dir)

    # Verify API is reachable
    session = requests.Session()
    try:
        r = session.get(f"{base_url}/health", timeout=5)
        logger.info("API health: HTTP %d", r.status_code)
    except Exception as exc:
        logger.warning("Could not reach %s/health: %s (continuing anyway)", base_url, exc)

    # ── Run all prompts ───────────────────────────────────────────────────────
    wall_start = time.perf_counter()
    results: list[dict] = []

    for entry in prompts:
        rec = run_prompt(session, base_url, entry, output_dir, timeout)
        results.append(rec)

    wall_elapsed = time.perf_counter() - wall_start

    # ── Summary ───────────────────────────────────────────────────────────────
    ok_count = sum(
        1 for r in results
        if all(s.get("status") == "ok" for s in r["steps"].values()
               if s.get("status") != "skipped")
    )
    print("\n" + "=" * 60)
    print("  Demo Run Summary")
    print("=" * 60)
    for rec in results:
        statuses = "/".join(s.get("status", "?") for s in rec["steps"].values())
        print(f"  [{rec['id']:02d}] {rec['pattern']:<20} {statuses}")
    print("-" * 60)
    print(f"  Prompts run: {len(results)}  |  Fully OK: {ok_count}")
    print(f"  Wall time:   {wall_elapsed:.1f}s")
    print("=" * 60)

    # ── Save summary JSON ────────────────────────────────────────────────────
    summary_path = output_dir / "demo_summary.json"
    summary_path.write_text(
        json.dumps({
            "base_url":       base_url,
            "n_prompts":      len(results),
            "n_ok":           ok_count,
            "wall_elapsed_s": round(wall_elapsed, 2),
            "results":        results,
        }, indent=2),
        encoding="utf-8",
    )
    logger.info("Summary → %s", summary_path)

    # Exit non-zero if any prompt failed completely
    any_failed = any(
        any(s.get("status") in {"error", "exception"}
            for s in r["steps"].values())
        for r in results
    )
    sys.exit(1 if any_failed else 0)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run ArchitectAI demo pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--url",     default="http://localhost:8000")
    p.add_argument("--input",   default="docs/demo_prompts.json")
    p.add_argument("--output",  default="docs/demo_outputs")
    p.add_argument("--ids",     default="",
                   help="Comma-separated prompt IDs to run (e.g. '1,3,5'). Default: all.")
    p.add_argument("--timeout", type=int, default=60)
    return p.parse_args()


if __name__ == "__main__":
    main()
