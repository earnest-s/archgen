#!/usr/bin/env python3
import re

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.core.inference import generate_explanation

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _parse_text_to_architecture(text: str) -> dict:
    lowered = text.lower()

    if any(k in lowered for k in ("event", "broker", "queue")):
        pattern = "event-driven"
    elif any(k in lowered for k in ("microservice", "api gateway")):
        pattern = "microservices"
    elif any(k in lowered for k in ("stream", "pipeline", "etl")):
        pattern = "streaming"
    elif "cache" in lowered:
        pattern = "cache-enabled"
    else:
        pattern = "layered"

    candidates = [
        ("frontend", ["frontend", "ui", "client", "web"]),
        ("backend", ["backend", "api", "service", "server"]),
        ("database", ["database", "db", "postgres", "mysql", "mongodb"]),
        ("cache", ["cache", "redis"]),
        ("broker", ["broker", "queue", "kafka", "rabbitmq"]),
    ]

    labels = []
    for label, keys in candidates:
        if any(re.search(rf"\\b{re.escape(k)}\\b", lowered) for k in keys):
            labels.append(label)

    if not labels:
        labels = ["frontend", "backend", "database"]

    nodes = []
    for i, label in enumerate(labels):
        nodes.append({"id": f"n{i}", "type": "Component", "label": label})

    id_by_label = {n["label"]: n["id"] for n in nodes}
    edges = []

    def add_edge(src: str, dst: str, protocol: str) -> None:
        if src in id_by_label and dst in id_by_label:
            edges.append({"from": id_by_label[src], "to": id_by_label[dst], "protocol": protocol})

    add_edge("frontend", "backend", "https")
    add_edge("backend", "database", "sql")
    add_edge("backend", "cache", "redis")
    add_edge("frontend", "broker", "events")
    add_edge("broker", "backend", "events")

    if not edges and len(nodes) > 1:
        for i in range(len(nodes) - 1):
            edges.append({"from": nodes[i]["id"], "to": nodes[i + 1]["id"], "protocol": "https"})

    return {
        "name": "nl-architecture",
        "pattern": pattern,
        "nodes": nodes,
        "edges": edges,
    }


@app.post("/explain")
async def explain(payload: dict):
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Invalid request body")

    architecture = payload.get("architecture")
    if architecture is None:
        text = payload.get("text")
        if isinstance(text, str) and text.strip():
            architecture = _parse_text_to_architecture(text.strip())

    if architecture is None:
        raise HTTPException(status_code=400, detail="Provide either 'architecture' (JSON) or 'text' (natural language).")

    explanation = generate_explanation(architecture)
    return {"explanation": explanation}
