"""stress_test_api.py — Send N prompts to the /generate endpoint and measure latency.

Metrics collected per request:
  - HTTP status code
  - Total wall-clock latency (seconds)
  - Diagram generation time (from response body, if present)
  - Explanation time (from response body, if present)

Aggregate statistics written to JSON:
  - success_rate, n_requests, n_errors
  - avg_latency_s, min_latency_s, max_latency_s
  - p50_latency_s, p95_latency_s, p99_latency_s
  - per_request list with index, status, latency, error (if any)

Usage
-----
    python scripts/stress_test_api.py \\
        --url          http://localhost:8000 \\
        --n            100 \\
        --concurrency  1 \\
        --output       reports/api_stress_test.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    "Build a three-tier web application with a React frontend, Node.js API, and PostgreSQL database.",
    "Design a microservices architecture for an e-commerce platform with order, product, and user services.",
    "Create an event-driven pipeline with Kafka, a stream processor, and a data warehouse.",
    "Architect a CQRS system with a command API, event store, and read model service.",
    "Set up a CDN-backed static site with CloudFront, S3, and a Lambda API.",
    "Design a real-time analytics system with Kinesis, Lambda, and DynamoDB.",
    "Build a serverless authentication service with API Gateway, Cognito, and Lambda.",
    "Create a distributed cache layer with Redis Cluster in front of a MySQL database.",
    "Architect a GraphQL gateway that aggregates three downstream REST microservices.",
    "Design a CI/CD pipeline with GitHub Actions, a Docker registry, and Kubernetes.",
    "Build a data lake ingestion pipeline with S3, Glue ETL, and Athena.",
    "Create a messaging system with RabbitMQ connecting a producer service to multiple consumers.",
    "Design a gRPC backend with a load balancer, two worker services, and a shared Postgres cluster.",
    "Architect a two-region active-active setup with Route 53, two API clusters, and cross-region RDS.",
    "Build a webhook delivery system with a queue, worker pool, retry store, and status API.",
    "Design a search platform with Elasticsearch, a data ingestion service, and a query frontend.",
    "Create a recommendation engine that reads user events, processes them in Spark, and stores results in Redis.",
    "Architect an IoT telemetry pipeline: MQTT broker, stream processor, time-series DB, and dashboard.",
    "Build a monorepo backend with an API gateway routing to auth, billing, and notification services.",
    "Design a blue/green deployment setup with a load balancer, two identical service stacks, and a DB.",
]


def _get_prompt(index: int) -> str:
    return SAMPLE_PROMPTS[index % len(SAMPLE_PROMPTS)]


# ---------------------------------------------------------------------------
# Single request
# ---------------------------------------------------------------------------

def _send_request(
    session,
    base_url: str,
    index: int,
    timeout: float,
) -> Dict:
    """Fire one POST /generate request and return a result dict."""
    import urllib.parse

    prompt = _get_prompt(index)
    url    = base_url.rstrip("/") + "/generate"

    result: Dict = {
        "index":            index,
        "prompt":           prompt[:60],
        "status":           None,
        "latency_s":        None,
        "diagram_time_s":   None,
        "error":            None,
    }

    start = time.perf_counter()
    try:
        response = session.post(
            url,
            json={"prompt": prompt},
            timeout=timeout,
        )
        elapsed = time.perf_counter() - start
        result["status"]    = response.status_code
        result["latency_s"] = round(elapsed, 4)

        if response.status_code == 200:
            try:
                body = response.json()
                if "diagram_time_s" in body:
                    result["diagram_time_s"] = body["diagram_time_s"]
            except Exception:
                pass
        else:
            result["error"] = f"HTTP {response.status_code}: {response.text[:200]}"

    except Exception as exc:
        elapsed = time.perf_counter() - start
        result["latency_s"] = round(elapsed, 4)
        result["error"]     = str(exc)[:300]

    return result


# ---------------------------------------------------------------------------
# Percentile helper (no numpy dependency)
# ---------------------------------------------------------------------------

def _percentile(sorted_data: List[float], pct: float) -> float:
    """Return the *pct*-th percentile of a pre-sorted list."""
    if not sorted_data:
        return 0.0
    k = (len(sorted_data) - 1) * pct / 100.0
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return sorted_data[lo]
    return sorted_data[lo] + (sorted_data[hi] - sorted_data[lo]) * (k - lo)


# ---------------------------------------------------------------------------
# Main stress-test
# ---------------------------------------------------------------------------

def stress_test(
    base_url: str,
    n: int,
    concurrency: int,
    request_timeout: float,
    output_path: Path,
) -> None:
    try:
        import requests  # type: ignore
    except ImportError:
        logger.error("requests not installed. Run: pip install requests")
        sys.exit(1)

    from requests.adapters import HTTPAdapter  # type: ignore

    logger.info(
        "Stress test: %d requests → %s  (concurrency=%d)", n, base_url, concurrency
    )

    # Build a session with connection pooling.
    session = requests.Session()
    adapter = HTTPAdapter(pool_connections=concurrency, pool_maxsize=concurrency + 4)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    per_request: List[Dict] = []
    wall_start = time.perf_counter()

    if concurrency == 1:
        # Sequential — simplest, predictable order.
        for i in range(n):
            result = _send_request(session, base_url, i, request_timeout)
            per_request.append(result)
            status_str = result["status"] or "ERR"
            err_str    = f" [{result['error'][:60]}]" if result["error"] else ""
            logger.info(
                "  [%3d/%d] status=%s  latency=%.3fs%s",
                i + 1, n, status_str, result["latency_s"] or 0, err_str,
            )
    else:
        # Concurrent with ThreadPoolExecutor.
        futures = {}
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            for i in range(n):
                fut = executor.submit(_send_request, session, base_url, i, request_timeout)
                futures[fut] = i

            completed = 0
            for fut in as_completed(futures):
                result = fut.result()
                per_request.append(result)
                completed += 1
                status_str = result["status"] or "ERR"
                err_str    = f" [{result['error'][:60]}]" if result["error"] else ""
                logger.info(
                    "  [%3d/%d] index=%d  status=%s  latency=%.3fs%s",
                    completed, n, result["index"], status_str,
                    result["latency_s"] or 0, err_str,
                )

        # Sort by original index for deterministic output.
        per_request.sort(key=lambda r: r["index"])

    wall_elapsed = round(time.perf_counter() - wall_start, 3)

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    successes       = [r for r in per_request if r["status"] == 200]
    errors          = [r for r in per_request if r["error"] is not None]
    latencies       = sorted(r["latency_s"] for r in per_request if r["latency_s"] is not None)

    n_success       = len(successes)
    n_errors        = len(errors)
    success_rate    = round(n_success / n, 4) if n else 0.0

    avg_latency     = round(statistics.mean(latencies),   4) if latencies else 0.0
    min_latency     = round(min(latencies),               4) if latencies else 0.0
    max_latency     = round(max(latencies),               4) if latencies else 0.0
    p50_latency     = round(_percentile(latencies,  50),  4) if latencies else 0.0
    p95_latency     = round(_percentile(latencies,  95),  4) if latencies else 0.0
    p99_latency     = round(_percentile(latencies,  99),  4) if latencies else 0.0
    stdev_latency   = round(statistics.stdev(latencies),  4) if len(latencies) > 1 else 0.0

    throughput_rps  = round(n / wall_elapsed, 3) if wall_elapsed > 0 else 0.0

    report = {
        "base_url":          base_url,
        "n_requests":        n,
        "concurrency":       concurrency,
        "wall_time_s":       wall_elapsed,
        "success_rate":      success_rate,
        "n_success":         n_success,
        "n_errors":          n_errors,
        "throughput_rps":    throughput_rps,
        "avg_latency_s":     avg_latency,
        "min_latency_s":     min_latency,
        "max_latency_s":     max_latency,
        "stdev_latency_s":   stdev_latency,
        "p50_latency_s":     p50_latency,
        "p95_latency_s":     p95_latency,
        "p99_latency_s":     p99_latency,
        "error_details":     [
            {"index": r["index"], "status": r["status"], "error": r["error"]}
            for r in errors
        ],
        "per_request":       per_request,
    }

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  API Stress Test Results")
    print("=" * 60)
    print(f"  Target       : {base_url}")
    print(f"  Requests     : {n}  (concurrency={concurrency})")
    print(f"  Wall time    : {wall_elapsed:.2f}s  ({throughput_rps:.2f} req/s)")
    print(f"  Success rate : {success_rate:.1%}  ({n_success}/{n})")
    print(f"  Errors       : {n_errors}")
    print(f"\n  Latency (all requests)")
    print(f"    avg  : {avg_latency:.3f}s")
    print(f"    min  : {min_latency:.3f}s")
    print(f"    max  : {max_latency:.3f}s")
    print(f"    p50  : {p50_latency:.3f}s")
    print(f"    p95  : {p95_latency:.3f}s")
    print(f"    p99  : {p99_latency:.3f}s")
    if errors:
        print(f"\n  First 5 errors:")
        for r in errors[:5]:
            print(f"    [{r['index']}] status={r['status']}  {r['error'][:80]}")
    print("=" * 60 + "\n")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Report saved → %s", output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stress-test the /generate API endpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--url", default="http://localhost:8000",
        help="Base URL of the API server (no trailing slash).",
    )
    p.add_argument(
        "--n", type=int, default=100,
        help="Total number of requests to send.",
    )
    p.add_argument(
        "--concurrency", type=int, default=1,
        help="Number of parallel worker threads (1 = sequential).",
    )
    p.add_argument(
        "--timeout", type=float, default=120.0,
        help="Per-request timeout in seconds.",
    )
    p.add_argument(
        "--output", default="reports/api_stress_test.json",
        help="Path to write the results JSON.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    stress_test(
        base_url        = args.url,
        n               = args.n,
        concurrency     = args.concurrency,
        request_timeout = args.timeout,
        output_path     = Path(args.output),
    )


if __name__ == "__main__":
    main()
