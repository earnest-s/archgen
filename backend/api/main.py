#!/usr/bin/env python3
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.core.inference import generate_explanation, preload_model

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def preload_inference_model() -> None:
    preload_model()


def _parse_text_to_architecture(text: str) -> dict:
    lowered = text.lower()

    keyword_nodes = [
        (("frontend", "front-end", "ui", "client"), {"id": "frontend", "type": "ui"}),
        (("backend", "back-end", "api", "server"), {"id": "backend", "type": "service"}),
        (("database", "db", "postgres", "mysql", "mongo", "redis"), {"id": "database", "type": "data"}),
        (("docker", "kubernetes", "k8s", "nginx"), {"id": "docker", "type": "service"}),
    ]

    nodes: list[dict[str, str]] = []

    for keywords, node in keyword_nodes:
        if any(keyword in lowered for keyword in keywords):
            nodes.append({"id": node["id"], "type": node["type"]})

    return _normalize_architecture({"nodes": nodes})


def _infer_node_type(node_id: str) -> str:
    lowered = node_id.lower()
    if any(part in lowered for part in ("front", "ui", "client")):
        return "ui"
    if any(part in lowered for part in ("db", "data", "database", "postgres", "mysql", "mongo", "redis")):
        return "data"
    return "service"


def _coerce_nodes(raw_nodes: object) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    seen_ids: set[str] = set()

    def add_node(node_id: str, node_type: str) -> None:
        normalized_type = node_type if node_type in {"ui", "service", "data"} else _infer_node_type(node_id)
        if node_id in seen_ids:
            return
        seen_ids.add(node_id)
        out.append({"id": node_id, "type": normalized_type})

    if isinstance(raw_nodes, list):
        for node in raw_nodes:
            if isinstance(node, str) and node.strip():
                node_id = node.strip().lower().replace(" ", "-")
                add_node(node_id, _infer_node_type(node_id))
            elif isinstance(node, dict):
                node_id = node.get("id") or node.get("label")
                node_type = node.get("type")
                if isinstance(node_id, str) and node_id.strip():
                    normalized_id = node_id.strip().lower().replace(" ", "-")
                    if isinstance(node_type, str):
                        add_node(normalized_id, node_type)
                    else:
                        add_node(normalized_id, _infer_node_type(normalized_id))

    if not out:
        add_node("frontend", "ui")
        add_node("backend", "service")
        add_node("database", "data")

    has_ui = any(node["type"] == "ui" for node in out)
    has_service = any(node["type"] == "service" for node in out)
    has_data = any(node["type"] == "data" for node in out)

    # Ensure deterministic flow layers exist so edges are never empty.
    if has_ui and not has_service:
        add_node("backend", "service")
        has_service = True
    if has_service and not has_data:
        add_node("database", "data")
        has_data = True
    if has_data and not has_service:
        add_node("backend", "service")
        has_service = True
    if not has_ui and has_service:
        add_node("frontend", "ui")

    return out


def _build_edges(nodes: list[dict[str, str]]) -> list[dict[str, str]]:
    ui_nodes = [node["id"] for node in nodes if node["type"] == "ui"]
    service_nodes = [node["id"] for node in nodes if node["type"] == "service"]
    data_nodes = [node["id"] for node in nodes if node["type"] == "data"]

    edges: list[dict[str, str]] = []
    seen_edges: set[tuple[str, str]] = set()

    def add_edge(source: str, target: str) -> None:
        key = (source, target)
        if source == target or key in seen_edges:
            return
        seen_edges.add(key)
        edges.append({"source": source, "target": target})

    # Rule 1: ui -> service
    for source in ui_nodes:
        for target in service_nodes:
            add_edge(source, target)

    # Rule 2: service -> data
    for source in service_nodes:
        for target in data_nodes:
            add_edge(source, target)

    # Rule 3: service -> service when multiple services exist
    if len(service_nodes) > 1:
        for idx in range(len(service_nodes) - 1):
            add_edge(service_nodes[idx], service_nodes[idx + 1])

    if not edges and len(nodes) > 1:
        for idx in range(len(nodes) - 1):
            add_edge(nodes[idx]["id"], nodes[idx + 1]["id"])

    return edges


def _normalize_architecture(raw: object) -> dict:
    raw_nodes = raw.get("nodes") if isinstance(raw, dict) else None
    nodes = _coerce_nodes(raw_nodes)
    edges = _build_edges(nodes)

    return {"nodes": nodes, "edges": edges}


@app.post("/explain")
async def explain(payload: dict):
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Invalid request body")

    architecture = None

    text = payload.get("text")
    if isinstance(text, str) and text.strip():
        architecture = _parse_text_to_architecture(text.strip())
    elif payload.get("architecture") is not None:
        architecture = _normalize_architecture(payload.get("architecture"))

    if architecture is None:
        raise HTTPException(status_code=400, detail="Provide either 'architecture' (JSON) or 'text' (natural language).")

    explanation = generate_explanation(architecture)
    return {"explanation": explanation, "architecture": architecture}
