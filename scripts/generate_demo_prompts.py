"""generate_demo_prompts.py — Generate demonstration prompts for ArchitectAI.

Produces a structured text file covering 6 architecture patterns:
  1. Layered (traditional web application)
  2. Microservices (API gateway + independent services)
  3. Event-driven (message broker + consumers)
  4. Streaming pipeline (real-time data processing)
  5. Cache-enabled (CDN + Redis caching tier)
  6. Queue-based worker pool (async job processing)

Each entry has:
  • A natural-language prompt suitable for the parse_prompt endpoint.
  • A bullet list of expected components for manual validation.

Output
------
  docs/demo_prompts.txt  — human-readable format
  docs/demo_prompts.json — machine-readable format (for run_demo.py)

Usage
-----
    python scripts/generate_demo_prompts.py [--output docs/demo_prompts]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Prompt definitions
# ---------------------------------------------------------------------------

DEMO_PROMPTS: list[dict] = [
    # ── 1. Layered ────────────────────────────────────────────────────────────
    {
        "id": 1,
        "pattern": "layered",
        "title": "Classic Three-Tier Web Application",
        "prompt": (
            "Design a three-tier web application with a React frontend, "
            "a FastAPI application server, and a PostgreSQL database. "
            "Use Nginx as the reverse proxy and add Redis for session caching."
        ),
        "expected_components": [
            "React (frontend)",
            "Nginx (reverse proxy / load balancer)",
            "FastAPI (API server)",
            "Redis (cache)",
            "PostgreSQL (database)",
        ],
    },
    # ── 2. Microservices ─────────────────────────────────────────────────────
    {
        "id": 2,
        "pattern": "microservices",
        "title": "E-Commerce Microservices Platform",
        "prompt": (
            "Build a microservices e-commerce platform with an API Gateway as the "
            "single entry point. Behind the gateway, run four services: "
            "User Service, Product Catalogue Service, Order Service, and Payment Service. "
            "Each service has its own PostgreSQL database. "
            "Use Kafka for inter-service events and Redis for distributed caching."
        ),
        "expected_components": [
            "API Gateway",
            "User Service",
            "Product Catalogue Service",
            "Order Service",
            "Payment Service",
            "Kafka (message broker)",
            "Redis (cache)",
            "PostgreSQL (×4 dedicated databases)",
        ],
    },
    # ── 3. Event-driven ──────────────────────────────────────────────────────
    {
        "id": 3,
        "pattern": "event-driven",
        "title": "IoT Event Processing System",
        "prompt": (
            "Create an event-driven IoT data pipeline. "
            "Devices publish sensor readings to an MQTT broker. "
            "A Kafka adapter bridges MQTT to Kafka topics. "
            "Three consumer microservices subscribe: "
            "Anomaly Detector, Metrics Aggregator, and Alert Service. "
            "Processed data lands in InfluxDB for time-series storage "
            "and Grafana for real-time monitoring."
        ),
        "expected_components": [
            "IoT Devices / sensors (external)",
            "MQTT Broker",
            "Kafka (event streaming)",
            "Anomaly Detector (consumer)",
            "Metrics Aggregator (consumer)",
            "Alert Service (consumer)",
            "InfluxDB (time-series database)",
            "Grafana (monitoring dashboard)",
        ],
    },
    # ── 4. Streaming pipeline ─────────────────────────────────────────────────
    {
        "id": 4,
        "pattern": "streaming_pipeline",
        "title": "Real-Time Clickstream Analytics",
        "prompt": (
            "Design a real-time clickstream analytics system. "
            "Website events are captured by a JavaScript tracker and sent to "
            "a Kafka cluster. Apache Flink processes the stream for "
            "session stitching and funnel aggregation. "
            "Results flow into Apache Druid for OLAP queries and "
            "a Redis leaderboard for live rankings. "
            "A Superset dashboard provides the analyst UI."
        ),
        "expected_components": [
            "JS Event Tracker (external client)",
            "Kafka (stream ingestion)",
            "Apache Flink (stream processor)",
            "Apache Druid (OLAP store)",
            "Redis (leaderboard cache)",
            "Apache Superset (analytics dashboard)",
        ],
    },
    # ── 5. Cache-enabled CDN ─────────────────────────────────────────────────
    {
        "id": 5,
        "pattern": "cache_enabled",
        "title": "High-Traffic Content Delivery with Multi-Layer Caching",
        "prompt": (
            "Architect a high-traffic media site with multi-layer caching. "
            "CloudFront CDN handles static assets globally. "
            "Requests that miss the CDN reach an Nginx reverse proxy "
            "which checks a Redis L2 cache before forwarding to Django REST API. "
            "The API reads from a read-replica PostgreSQL cluster (primary + 2 replicas). "
            "Background Celery workers refresh cache entries on content updates."
        ),
        "expected_components": [
            "CloudFront CDN (external / edge)",
            "Nginx (reverse proxy + L1 cache)",
            "Redis (L2 application cache)",
            "Django REST API",
            "Celery Workers (background jobs)",
            "PostgreSQL Primary",
            "PostgreSQL Read Replica ×2",
        ],
    },
    # ── 6. Queue-based worker pool ────────────────────────────────────────────
    {
        "id": 6,
        "pattern": "queue_worker",
        "title": "Asynchronous Document Processing Pipeline",
        "prompt": (
            "Build an asynchronous document processing pipeline. "
            "Users upload files through a FastAPI upload endpoint. "
            "Uploaded files are stored in S3. "
            "A job record is pushed to RabbitMQ. "
            "A pool of five Python worker processes consumes the queue: "
            "each worker fetches the file from S3, runs OCR, extracts entities, "
            "and writes results to MongoDB. "
            "A notification service reads a results queue and emails the user."
        ),
        "expected_components": [
            "FastAPI Upload Service",
            "S3 (object storage)",
            "RabbitMQ (job queue)",
            "OCR Worker ×5 (worker pool)",
            "MongoDB (document store)",
            "Notification Service",
            "Email Provider (external)",
        ],
    },
]


# ---------------------------------------------------------------------------
# Serialisers
# ---------------------------------------------------------------------------


def _to_text(prompts: list[dict]) -> str:
    lines: list[str] = [
        "=" * 70,
        "ArchitectAI — Demonstration Prompts",
        "=" * 70,
        "",
        f"Total prompts: {len(prompts)}",
        "Patterns covered: layered, microservices, event-driven,",
        "  streaming_pipeline, cache_enabled, queue_worker",
        "",
    ]
    for entry in prompts:
        lines += [
            "=" * 70,
            f"[{entry['id']}] {entry['title']}",
            f"Pattern: {entry['pattern']}",
            "-" * 70,
            "PROMPT:",
            entry["prompt"],
            "",
            "EXPECTED COMPONENTS:",
        ]
        for component in entry["expected_components"]:
            lines.append(f"  - {component}")
        lines.append("")
    lines.append("=" * 70)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate demo prompts for ArchitectAI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output", default="docs/demo_prompts",
        help="Output file base path (without extension; .txt and .json are written).",
    )
    args = parser.parse_args()

    out_base = Path(args.output)
    out_base.parent.mkdir(parents=True, exist_ok=True)

    txt_path  = out_base.with_suffix(".txt")
    json_path = out_base.with_suffix(".json")

    txt_path.write_text(_to_text(DEMO_PROMPTS), encoding="utf-8")
    json_path.write_text(json.dumps(DEMO_PROMPTS, indent=2), encoding="utf-8")

    print(f"Written {len(DEMO_PROMPTS)} prompts:")
    print(f"  {txt_path}")
    print(f"  {json_path}")


if __name__ == "__main__":
    main()
