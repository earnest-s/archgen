"""profile_pipeline.py — Measure per-stage runtime of the ArchitectAI pipeline.

Runs each of the four pipeline stages on 20 sample prompts and records the
average, min, max, and p95 latency for each stage individually.

Stages measured
---------------
  1. parse_prompt      — rule-based NLP prompt → Architecture
  2. generate_diagram  — Architecture → PNG (Graphviz)
  3. encode_diagram    — PNG → 768-dim ConvNeXt embedding
  4. generate_explanation — Architecture [+ embedding] → text (Qwen LLM)

Results are printed to stdout and saved to reports/pipeline_profile.json.

Usage
-----
    python scripts/profile_pipeline.py [OPTIONS]

Options
-------
    --n          Number of prompts to profile (default: 20)
    --output     Output JSON path (default: reports/pipeline_profile.json)
    --skip-llm   Skip the LLM explanation stage (slow; needs GPU)
    --skip-encode Skip the vision encoding stage (needs ConvNeXt checkpoint)
    --convnext   Path to ConvNeXt checkpoint
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sample prompts
# ---------------------------------------------------------------------------

SAMPLE_PROMPTS: List[str] = [
    "React frontend, FastAPI backend, PostgreSQL database",
    "Microservices with order service, payment service, and notification service",
    "Event-driven pipeline with Kafka, stream processor, and data warehouse",
    "CQRS system with command API, event store, and read model",
    "CDN-backed static site with CloudFront, S3, and Lambda",
    "Three-tier web app with Angular, Node.js API, and MySQL",
    "gRPC backend with load balancer, two workers, and a shared Redis cache",
    "Real-time chat app with WebSocket gateway, message broker, and user service",
    "API gateway routing to auth, billing, and notification microservices",
    "Serverless backend with API Gateway, Lambda functions, and DynamoDB",
    "Data lake pipeline with S3 ingestion, Spark ETL, and Redshift warehouse",
    "IoT telemetry with MQTT broker, time-series DB, and a dashboard service",
    "Blue/green deployment with load balancer and two identical service stacks",
    "Search platform with Elasticsearch, ingestion service, and query frontend",
    "Recommendation engine with event stream, Spark processor, and Redis store",
    "Vue.js SPA, Django REST API, PostgreSQL, and a Redis cache layer",
    "Next.js frontend, GraphQL gateway, and three domain microservices",
    "Background job system with API, job queue, worker pool, and result store",
    "Two-region active-active setup with Route 53 and cross-region RDS",
    "Video streaming platform with CDN, transcoder service, and metadata DB",
]


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def _percentile(data: List[float], pct: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    k = (len(s) - 1) * pct / 100.0
    lo, hi = int(math.floor(k)), int(math.ceil(k))
    return s[lo] if lo == hi else s[lo] + (s[hi] - s[lo]) * (k - lo)


def _stats(times: List[float]) -> Dict:
    if not times:
        return {"avg": 0, "min": 0, "max": 0, "p95": 0, "n": 0}
    return {
        "avg_s": round(statistics.mean(times),  4),
        "min_s": round(min(times),              4),
        "max_s": round(max(times),              4),
        "p95_s": round(_percentile(times, 95),  4),
        "n":     len(times),
    }


# ---------------------------------------------------------------------------
# Profiling stages
# ---------------------------------------------------------------------------

def profile_parse(prompts: List[str]) -> Dict:
    logger.info("Profiling: parse_prompt (%d prompts)…", len(prompts))
    from backend.core.prompt_parser.parser import parse_prompt  # type: ignore

    times: List[float] = []
    for p in prompts:
        t0 = time.perf_counter()
        try:
            parse_prompt(p)
        except Exception as exc:
            logger.warning("parse_prompt failed: %s", exc)
            continue
        times.append(time.perf_counter() - t0)

    return _stats(times)


def profile_generate(prompts: List[str], tmpdir: str) -> tuple[Dict, List[Optional[str]]]:
    """Returns stats dict + list of generated PNG paths (None on failure)."""
    logger.info("Profiling: generate_diagram (%d prompts)…", len(prompts))
    from backend.core.prompt_parser.parser import parse_prompt  # type: ignore
    from backend.core.diagram.generator import generate_diagram  # type: ignore

    times: List[float] = []
    paths: List[Optional[str]] = []

    for idx, p in enumerate(prompts):
        try:
            arch = parse_prompt(p)
            out  = str(Path(tmpdir) / f"arch_{idx:03d}")
            t0   = time.perf_counter()
            png  = generate_diagram(arch, output_path=out)
            times.append(time.perf_counter() - t0)
            paths.append(png)
        except Exception as exc:
            logger.warning("generate_diagram failed for prompt %d: %s", idx, exc)
            paths.append(None)

    return _stats(times), paths


def profile_encode(
    png_paths: List[Optional[str]],
    convnext_ckpt: Optional[str],
) -> tuple[Dict, list]:
    logger.info("Profiling: encode_diagram (%d images)…", len(png_paths))
    from backend.core.vision.encoder import encode_diagram  # type: ignore

    times: List[float]  = []
    features: list      = []
    kwargs = {"checkpoint_path": convnext_ckpt} if convnext_ckpt else {}

    for png in png_paths:
        if png is None or not Path(png).exists():
            features.append(None)
            continue
        try:
            t0   = time.perf_counter()
            feat = encode_diagram(png, **kwargs)
            times.append(time.perf_counter() - t0)
            features.append(feat)
        except Exception as exc:
            logger.warning("encode_diagram failed: %s", exc)
            features.append(None)

    return _stats(times), features


def profile_explain(
    prompts: List[str],
    features: list,
) -> Dict:
    logger.info("Profiling: generate_explanation (%d samples)…", len(prompts))
    from backend.core.prompt_parser.parser import parse_prompt  # type: ignore
    from backend.core.vlm.explainer import generate_explanation  # type: ignore

    times: List[float] = []

    for idx, p in enumerate(prompts):
        try:
            arch = parse_prompt(p)
            feat = features[idx] if idx < len(features) else None
            t0   = time.perf_counter()
            generate_explanation(arch, vision_features=feat)
            times.append(time.perf_counter() - t0)
        except Exception as exc:
            logger.warning("generate_explanation failed for prompt %d: %s", idx, exc)

    return _stats(times)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    n       = args.n
    prompts = (SAMPLE_PROMPTS * math.ceil(n / len(SAMPLE_PROMPTS)))[:n]
    output  = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    results: Dict = {}

    with tempfile.TemporaryDirectory() as tmpdir:

        # Stage 1: parse
        results["parse_prompt"] = profile_parse(prompts)
        logger.info("  parse_prompt: avg=%.4fs", results["parse_prompt"].get("avg_s", 0))

        # Stage 2: generate
        gen_stats, png_paths = profile_generate(prompts, tmpdir)
        results["generate_diagram"] = gen_stats
        logger.info("  generate_diagram: avg=%.4fs", gen_stats.get("avg_s", 0))

        # Stage 3: encode
        if not args.skip_encode:
            enc_stats, features = profile_encode(png_paths, args.convnext)
            results["encode_diagram"] = enc_stats
            logger.info("  encode_diagram: avg=%.4fs", enc_stats.get("avg_s", 0))
        else:
            features = [None] * len(prompts)
            results["encode_diagram"] = {"skipped": True}
            logger.info("  encode_diagram: SKIPPED")

        # Stage 4: explain (optional — slow)
        if not args.skip_llm:
            exp_stats = profile_explain(prompts, features)
            results["generate_explanation"] = exp_stats
            logger.info("  generate_explanation: avg=%.4fs", exp_stats.get("avg_s", 0))
        else:
            results["generate_explanation"] = {"skipped": True}
            logger.info("  generate_explanation: SKIPPED")

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 66)
    print("  ArchitectAI Pipeline Profile")
    print("=" * 66)
    print(f"  {'Stage':<26} {'avg':>8} {'min':>8} {'p95':>8} {'max':>8}  n")
    print(f"  {'-'*26} {'-'*8} {'-'*8} {'-'*8} {'-'*8} --")
    for stage, s in results.items():
        if s.get("skipped"):
            print(f"  {stage:<26} {'SKIPPED':>8}")
        else:
            print(
                f"  {stage:<26} "
                f"{s['avg_s']:>7.3f}s "
                f"{s['min_s']:>7.3f}s "
                f"{s['p95_s']:>7.3f}s "
                f"{s['max_s']:>7.3f}s "
                f"{s['n']:>3}"
            )
    print("=" * 66 + "\n")

    output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Profile saved → %s", output)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Profile each stage of the ArchitectAI pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--n",          type=int, default=20, help="Number of prompts.")
    p.add_argument("--output",     default="reports/pipeline_profile.json")
    p.add_argument("--skip-llm",   action="store_true", help="Skip LLM explanation stage.")
    p.add_argument("--skip-encode",action="store_true", help="Skip vision encoding stage.")
    p.add_argument(
        "--convnext", default=None,
        help="Path to ConvNeXt checkpoint (omit to use random weights).",
    )
    return p.parse_args()


if __name__ == "__main__":
    main()
