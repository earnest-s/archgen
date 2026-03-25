#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

PATTERNS = [
    "layered",
    "microservices",
    "event-driven",
    "streaming",
    "cache-enabled",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic ArchitectAI dataset.")
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="data/synthetic/dataset.jsonl")
    parser.add_argument("--with-png", action="store_true", help="Optional lightweight flag (disabled by default).")
    return parser.parse_args()


def make_nodes(pattern: str, idx: int) -> List[Dict[str, str]]:
    if pattern == "layered":
        labels = ["frontend", "service", "database"]
    elif pattern == "microservices":
        labels = ["api-gateway", "auth-service", "orders-service", "db"]
    elif pattern == "event-driven":
        labels = ["producer", "broker", "consumer", "store"]
    elif pattern == "streaming":
        labels = ["ingest", "stream-processor", "analytics", "warehouse"]
    else:
        labels = ["frontend", "backend", "cache", "database"]

    return [
        {
            "id": f"n{idx}_{i}",
            "type": "Component",
            "label": label,
        }
        for i, label in enumerate(labels)
    ]


def make_edges(pattern: str, nodes: List[Dict[str, str]]) -> List[Dict[str, str]]:
    ids = [n["id"] for n in nodes]
    if pattern == "layered":
        links = [(ids[0], ids[1], "https"), (ids[1], ids[2], "sql")]
    elif pattern == "microservices":
        links = [
            (ids[0], ids[1], "https"),
            (ids[0], ids[2], "https"),
            (ids[1], ids[3], "sql"),
            (ids[2], ids[3], "sql"),
        ]
    elif pattern == "event-driven":
        links = [
            (ids[0], ids[1], "events"),
            (ids[1], ids[2], "events"),
            (ids[2], ids[3], "batch-write"),
        ]
    elif pattern == "streaming":
        links = [
            (ids[0], ids[1], "stream"),
            (ids[1], ids[2], "stream"),
            (ids[2], ids[3], "etl"),
        ]
    else:
        links = [
            (ids[0], ids[1], "https"),
            (ids[1], ids[2], "redis"),
            (ids[1], ids[3], "sql"),
            (ids[2], ids[3], "cache-fill"),
        ]

    return [
        {"from": src, "to": dst, "protocol": protocol}
        for src, dst, protocol in links
    ]


def make_explanation(pattern: str, architecture: Dict[str, object], rng: random.Random) -> str:
    node_labels = [n["label"] for n in architecture["nodes"]]
    components_csv = ", ".join(node_labels)

    component_templates = [
        "The main components are {components}.",
        "This design includes {components} as core building blocks.",
        "Key system parts are {components}.",
    ]

    flow_templates = [
        "Traffic moves through {paths}.",
        "Data exchanges follow {paths}.",
        "Primary communication paths are {paths}.",
    ]

    type_templates = [
        "This follows a {pattern} architecture style optimized for clear service boundaries.",
        "Overall, this is a {pattern} architecture emphasizing separation of responsibilities.",
        "Architecture type: {pattern}; components are organized for predictable request handling.",
    ]

    paths = []
    id_to_label = {n["id"]: n["label"] for n in architecture["nodes"]}
    for edge in architecture["edges"]:
        src = id_to_label.get(edge["from"], edge["from"])
        dst = id_to_label.get(edge["to"], edge["to"])
        protocol = edge["protocol"]
        paths.append(f"{src} -> {dst} via {protocol}")

    components_line = rng.choice(component_templates).format(components=components_csv)
    flow_line = rng.choice(flow_templates).format(paths="; ".join(paths))
    type_line = rng.choice(type_templates).format(pattern=pattern)

    return (
        f"Components: {components_line}\n"
        f"Data flow: {flow_line}\n"
        f"Architecture type: {type_line}"
    )


def architecture_signature(architecture: Dict[str, object]) -> str:
    labels = sorted(n["label"] for n in architecture["nodes"])
    edges = sorted((e["protocol"], e["from"], e["to"]) for e in architecture["edges"])
    return json.dumps({"labels": labels, "edges": edges}, sort_keys=True)


def generate_records(total: int, seed: int) -> List[Dict[str, object]]:
    rng = random.Random(seed)
    records: List[Dict[str, object]] = []
    signatures = set()
    attempts = 0

    while len(records) < total and attempts < total * 20:
        attempts += 1
        pattern = PATTERNS[len(records) % len(PATTERNS)]

        arch_idx = len(records)
        nodes = make_nodes(pattern, arch_idx)

        if rng.random() < 0.5:
            rng.shuffle(nodes)

        edges = make_edges(pattern, nodes)
        architecture = {
            "name": f"{pattern}-arch-{arch_idx}",
            "pattern": pattern,
            "nodes": nodes,
            "edges": edges,
        }

        sig = architecture_signature(architecture)
        if sig in signatures:
            continue
        signatures.add(sig)

        records.append(
            {
                "architecture": architecture,
                "explanation": make_explanation(pattern, architecture, rng),
            }
        )

    return records


def main() -> None:
    args = parse_args()
    count = min(max(1, args.num_samples), 800)

    print("[STEP 1/3] Dataset generation started")
    print(f"[INFO] Requested samples={args.num_samples}, capped_samples={count}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records = generate_records(count, args.seed)

    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=True) + "\n")

    if args.with_png:
        print("PNG export is intentionally skipped to keep disk usage minimal.")

    print(f"[INFO] Generated {len(records)} unique samples at {out_path}")
    print("[STEP 1/3] Dataset generation completed")


if __name__ == "__main__":
    main()
