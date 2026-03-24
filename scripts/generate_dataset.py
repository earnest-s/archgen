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

# Possible node type pools — weighted toward common real-world patterns.
# Covers: 3-tier, microservices, event-driven, CQRS, CDN, stream-processing,
# background-job, and pure API/data topologies.
_COMMON_PATTERNS: list[list[NodeType]] = [
    # ── Classic n-tier ────────────────────────────────────────────────────
    [NodeType.Frontend, NodeType.Backend, NodeType.Database],
    [NodeType.Frontend, NodeType.Backend, NodeType.Database, NodeType.Cache],
    [NodeType.Backend, NodeType.Database],
    [NodeType.Backend, NodeType.Cache, NodeType.Database],
    # ── Microservices ────────────────────────────────────────────────
    [NodeType.Frontend, NodeType.Backend, NodeType.Service, NodeType.Database, NodeType.Queue],
    [NodeType.Frontend, NodeType.Backend, NodeType.Service, NodeType.Database,
     NodeType.Cache, NodeType.Queue, NodeType.External],
    [NodeType.Backend, NodeType.Service, NodeType.Service, NodeType.Database],
    [NodeType.Frontend, NodeType.Backend, NodeType.Service, NodeType.Service,
     NodeType.Database, NodeType.Cache],
    # ── Event-driven ───────────────────────────────────────────────
    [NodeType.Frontend, NodeType.Backend, NodeType.Queue, NodeType.Service, NodeType.Database],
    [NodeType.Backend, NodeType.Queue, NodeType.Service, NodeType.Database, NodeType.Cache],
    [NodeType.Backend, NodeType.Queue, NodeType.Queue, NodeType.Service, NodeType.Database],
    # ── CQRS ───────────────────────────────────────────────────────────
    [NodeType.Backend, NodeType.Backend, NodeType.Queue, NodeType.Service,
     NodeType.Database, NodeType.Database],
    # ── CDN / external-integration ──────────────────────────────────
    [NodeType.External, NodeType.Frontend, NodeType.Backend, NodeType.Cache, NodeType.Database],
    [NodeType.Frontend, NodeType.Backend, NodeType.External, NodeType.Database],
    # ── Background-job processing ──────────────────────────────────
    [NodeType.Backend, NodeType.Queue, NodeType.Service, NodeType.Database],
    [NodeType.Backend, NodeType.Service, NodeType.Queue, NodeType.External, NodeType.Database],
    # ── Stream processing ─────────────────────────────────────────
    [NodeType.Backend, NodeType.Queue, NodeType.Queue, NodeType.Service,
     NodeType.Cache, NodeType.Database],
    # ── Layered (all tiers) ───────────────────────────────────────
    [NodeType.Frontend, NodeType.Backend, NodeType.Service, NodeType.Cache,
     NodeType.Database, NodeType.External],
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


def _random_topology_architecture() -> Architecture:
    """Generate a fully random architecture with 2-8 nodes and variable edge density.

    Unlike ``_random_architecture`` which picks a fixed pattern, this function
    samples node count and types freely, then wires up edges with a random
    density factor.  This ensures balanced NodeType coverage across the dataset.
    """
    # Pick 2-8 node types (with repetition allowed for multi-instance types)
    n_nodes = random.randint(2, 8)
    all_types = list(NodeType)

    # Bias selection: ensure rare types (Service, Queue, External) appear often
    weights = [1.5, 1.5, 2.0, 1.5, 1.5, 2.0, 2.0]  # Frontend..External
    node_types_seq = random.choices(all_types, weights=weights, k=n_nodes)

    type_counts: dict[NodeType, int] = {}
    nodes: list[Node] = []
    for node_type in node_types_seq:
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

    # Random edge density: 0.2 (sparse) to 0.7 (dense) of possible edges
    edge_density = random.uniform(0.2, 0.7)
    node_ids = [n.id for n in nodes]
    node_map = {n.id: n for n in nodes}

    # Generate candidate directed edges (no self-loops)
    candidate_pairs = [
        (src, dst)
        for i, src in enumerate(node_ids)
        for j, dst in enumerate(node_ids)
        if i != j
    ]
    random.shuffle(candidate_pairs)
    max_edges = max(1, int(len(candidate_pairs) * edge_density))

    # Ensure at least a spanning path so the graph is connected
    shuffled_ids = node_ids.copy()
    random.shuffle(shuffled_ids)
    spine_edges = list(zip(shuffled_ids, shuffled_ids[1:]))

    edges: list[Edge] = []
    seen_pairs: set[tuple[str, str]] = set()

    for src, dst in spine_edges + candidate_pairs:
        if (src, dst) in seen_pairs:
            continue
        if len(edges) >= max_edges:
            break
        src_type = node_map[src].type
        dst_type = node_map[dst].type
        protocols = _PROTOCOLS.get((src_type, dst_type), [])
        protocol  = random.choice(protocols) if protocols else None
        edges.append(Edge(**{"from": src, "to": dst, "protocol": protocol}))
        seen_pairs.add((src, dst))

    return Architecture(
        nodes=nodes,
        edges=edges,
        metadata=Metadata(version=1, style=random.choice(["light", "dark", None])),
    )


def _balanced_architecture(idx: int, n_total: int, diversity: str = "medium") -> Architecture:
    """Return a pattern-based or random-topology architecture.

    The distribution of pattern vs. random is controlled by *diversity*:
    - "low":    70% patterns, 30% random (less variety, more similar architectures)
    - "medium": 50% patterns, 50% random (balanced)
    - "high":   30% patterns, 70% random (maximum variety)

    Every 7th sample is forced to include a specific NodeType to ensure
    balanced coverage across the dataset.

    Args:
        idx:       Sample index in the dataset.
        n_total:   Total number of samples.
        diversity: One of "low", "medium", "high".
    """
    # Control pattern vs. random split
    diversity_thresholds = {
        "low": 7,      # 70% patterns
        "medium": 5,   # 50% patterns
        "high": 3,     # 30% patterns
    }
    pattern_threshold = diversity_thresholds.get(diversity, 5)

    # Every 7th sample: force-include a specific NodeType for balance
    if idx % 7 == 0:
        forced_type = list(NodeType)[idx // 7 % len(NodeType)]
        for _ in range(10):  # retry up to 10 times to get a valid arch
            arch = _random_topology_architecture()
            types_present = {n.type for n in arch.nodes}
            if forced_type in types_present:
                return arch
        # fallback: inject the forced type
        arch = _random_topology_architecture()
        label, node_id = _random_label(forced_type, 99)
        extra = Node(
            id=node_id,
            type=forced_type,
            label=label,
            layer=_LAYER_MAP.get(forced_type),
        )
        arch.nodes.append(extra)
        return arch

    if idx % 10 < pattern_threshold:  # Pattern-based
        return _random_architecture(idx)
    else:                              # Random topology
        return _random_topology_architecture()




def generate_dataset(n: int, out_dir: Path, diversity: str = "medium") -> None:
    """Generate *n* synthetic samples and save to *out_dir*.

    Uses balanced sampling with configurable diversity:
    - "low":    70% template-based patterns + 30% random (less variety)
    - "medium": 50% template-based patterns + 50% random (balanced)
    - "high":   30% template-based patterns + 70% random (maximum variety)

    Every 7th sample forces inclusion of a specific NodeType to guarantee
    coverage across all 7 types.

    Args:
        n:         Number of samples to generate.
        out_dir:   Root output directory (created if necessary).
        diversity: One of "low", "medium", "high" (default: "medium").
    """
    json_dir  = out_dir / "json"
    png_dir   = out_dir / "png"
    text_dir  = out_dir / "text"

    for d in (json_dir, png_dir, text_dir):
        d.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "dataset.jsonl"
    manifest_entries: list[dict] = []

    logger.info("Generating %d synthetic samples (diversity: %s) → %s", n, diversity, out_dir)

    for i in range(n):
        stem = f"sample_{i:04d}"
        try:
            arch = _balanced_architecture(i, n, diversity=diversity)
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
        "-n", "--num-samples", dest="n", type=int, default=100,
        metavar="N",
        help="Number of samples to generate (default: 100). Supports up to 5000+."
    )
    parser.add_argument(
        "--out", "--output-dir", dest="out", type=str, default="data/synthetic",
        metavar="DIR",
        help="Output directory (default: data/synthetic)."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)."
    )
    parser.add_argument(
        "--diversity", choices=["low", "medium", "high"], default="medium",
        help="Architecture diversity level: low (70%% patterns), medium (50%%), high (30%%). Affects uniqueness."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    random.seed(args.seed)
    generate_dataset(n=args.n, out_dir=Path(args.out))
