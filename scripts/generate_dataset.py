"""
Synthetic training dataset generator for ArchitectAI.

Generates N random architecture graphs, renders each as a PNG diagram, creates
a template-based text explanation, and saves all three artefacts together with
a JSONL manifest.

Output layout::

    data/synthetic/
    ├── json/   sample_0000.json  …
    ├── png/    sample_0000.png   …
    ├── text/   sample_0000.txt   …
    └── dataset.jsonl

Run::

    python -m scripts.generate_dataset --n 500 --out data/synthetic
    # or via the module directly:
    python scripts/generate_dataset.py --n 100
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

# Allow running as a top-level script from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.api.schemas.architecture import Architecture, Edge, Metadata, Node, NodeType
from backend.core.diagram.generator import generate_diagram
from backend.core.prompt_parser.rules import build_default_edges
from backend.core.vlm.explainer import generate_explanation_rule_based

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Generation config
# ---------------------------------------------------------------------------

# Possible node type pools, weighted toward common patterns.
_COMMON_PATTERNS: list[list[NodeType]] = [
    # 3-tier
    [NodeType.Frontend, NodeType.Backend, NodeType.Database],
    # 3-tier + cache
    [NodeType.Frontend, NodeType.Backend, NodeType.Database, NodeType.Cache],
    # Microservices
    [NodeType.Frontend, NodeType.Backend, NodeType.Service, NodeType.Database, NodeType.Queue],
    # Full stack
    [NodeType.Frontend, NodeType.Backend, NodeType.Service, NodeType.Database,
     NodeType.Cache, NodeType.Queue, NodeType.External],
    # API + DB
    [NodeType.Backend, NodeType.Database],
    # API + cache + DB
    [NodeType.Backend, NodeType.Cache, NodeType.Database],
    # Backend microservices
    [NodeType.Backend, NodeType.Service, NodeType.Service, NodeType.Database],
]

_LABEL_MAP: dict[NodeType, list[str]] = {
    NodeType.Frontend:  ["Web App", "React UI", "Next.js", "Mobile Client", "Angular SPA"],
    NodeType.Backend:   ["REST API", "FastAPI", "GraphQL Gateway", "Django Backend", "Express Server"],
    NodeType.Service:   ["Auth Service", "Notification Service", "Email Service", "Search Service", "Payment Service"],
    NodeType.Database:  ["PostgreSQL", "MySQL", "MongoDB", "DynamoDB", "CockroachDB"],
    NodeType.Cache:     ["Redis Cache", "Memcached", "CDN Layer"],
    NodeType.Queue:     ["Kafka", "RabbitMQ", "SQS", "NATS"],
    NodeType.External:  ["Stripe", "Twilio", "SendGrid", "Auth0", "S3"],
}

_LAYER_MAP: dict[NodeType, str] = {
    NodeType.Frontend:  "Presentation",
    NodeType.Backend:   "Application",
    NodeType.Service:   "Application",
    NodeType.Database:  "Data",
    NodeType.Cache:     "Data",
    NodeType.Queue:     "Infrastructure",
    NodeType.External:  "External",
}

_PROTOCOLS: dict[tuple[NodeType, NodeType], list[str]] = {
    (NodeType.Frontend,  NodeType.Backend):  ["HTTPS", "REST", "GraphQL"],
    (NodeType.Backend,   NodeType.Service):  ["gRPC", "HTTP", "Thrift"],
    (NodeType.Backend,   NodeType.Database): ["SQL", "TCP"],
    (NodeType.Backend,   NodeType.Cache):    ["TCP"],
    (NodeType.Backend,   NodeType.Queue):    ["AMQP", "Kafka"],
    (NodeType.Service,   NodeType.Database): ["SQL", "TCP"],
    (NodeType.Service,   NodeType.Queue):    ["AMQP"],
    (NodeType.Backend,   NodeType.External): ["HTTPS"],
}


def _random_label(node_type: NodeType, suffix_counter: int) -> tuple[str, str]:
    """Return ``(label, node_id)`` for a node of *node_type*."""
    base_label = random.choice(_LABEL_MAP.get(node_type, [node_type.value]))
    node_id = (
        base_label.lower()
        .replace(" ", "_")
        .replace(".", "")
        .replace("-", "_")
    )
    if suffix_counter > 0:
        node_id = f"{node_id}_{suffix_counter}"
    return base_label, node_id


def _random_architecture(idx: int) -> Architecture:
    """Generate a random Architecture from a weighted template pool."""
    pattern = random.choice(_COMMON_PATTERNS)

    # deduplicate types while keeping order, tracking count for id uniqueness
    type_counts: dict[NodeType, int] = {}
    nodes: list[Node] = []
    for node_type in pattern:
        count = type_counts.get(node_type, 0)
        label, node_id = _random_label(node_type, count)
        type_counts[node_type] = count + 1
        nodes.append(
            Node(
                id=node_id,
                type=node_type,
                label=label,
                layer=_LAYER_MAP.get(node_type),
            )
        )

    # Build edges with protocol annotations where possible.
    default_edges = build_default_edges(nodes)
    edges: list[Edge] = []
    node_map = {n.id: n for n in nodes}
    for e in default_edges:
        src_type = node_map[e.from_node].type
        dst_type = node_map[e.to_node].type
        protocols = _PROTOCOLS.get((src_type, dst_type), [])
        protocol = random.choice(protocols) if protocols else None
        edges.append(Edge(**{"from": e.from_node, "to": e.to_node, "protocol": protocol}))

    return Architecture(
        nodes=nodes,
        edges=edges,
        metadata=Metadata(version=1, style=random.choice(["light", "dark", None])),
    )


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------


def generate_dataset(n: int, out_dir: Path) -> None:
    """Generate *n* synthetic samples and save to *out_dir*.

    Args:
        n:       Number of samples to generate.
        out_dir: Root output directory (created if necessary).
    """
    json_dir  = out_dir / "json"
    png_dir   = out_dir / "png"
    text_dir  = out_dir / "text"

    for d in (json_dir, png_dir, text_dir):
        d.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "dataset.jsonl"
    manifest_entries: list[dict] = []

    logger.info("Generating %d synthetic samples → %s", n, out_dir)

    for i in range(n):
        stem = f"sample_{i:04d}"
        try:
            arch = _random_architecture(i)
        except Exception as exc:
            logger.warning("Sample %04d: architecture generation failed — %s", i, exc)
            continue

        # ── JSON ─────────────────────────────────────────────────────────────
        json_path = json_dir / f"{stem}.json"
        arch_dict = arch.model_dump(by_alias=True)
        json_path.write_text(json.dumps(arch_dict, indent=2), encoding="utf-8")

        # ── PNG ──────────────────────────────────────────────────────────────
        png_base = str(png_dir / stem)
        try:
            png_path = generate_diagram(arch, output_path=png_base)
        except Exception as exc:
            logger.warning("Sample %04d: diagram generation failed — %s", i, exc)
            png_path = ""

        # ── Text explanation ─────────────────────────────────────────────────
        explanation = generate_explanation_rule_based(arch)
        text_path = text_dir / f"{stem}.txt"
        text_path.write_text(explanation, encoding="utf-8")

        # ── Manifest ─────────────────────────────────────────────────────────
        manifest_entries.append(
            {
                "id":          stem,
                "image":       str(png_path),
                "architecture": arch_dict,
                "explanation": explanation,
            }
        )

        if (i + 1) % 50 == 0 or i == n - 1:
            logger.info("  %d / %d samples generated", i + 1, n)

    # Write JSONL manifest.
    with manifest_path.open("w", encoding="utf-8") as fh:
        for entry in manifest_entries:
            fh.write(json.dumps(entry) + "\n")

    logger.info(
        "Dataset complete: %d samples saved to %s", len(manifest_entries), out_dir
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic ArchitectAI training data."
    )
    parser.add_argument(
        "--n", type=int, default=100,
        help="Number of samples to generate (default: 100)."
    )
    parser.add_argument(
        "--out", type=str, default="data/synthetic",
        help="Output directory (default: data/synthetic)."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    random.seed(args.seed)
    generate_dataset(n=args.n, out_dir=Path(args.out))
