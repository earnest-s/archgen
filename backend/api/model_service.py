#!/usr/bin/env python3
import os

from fastapi import FastAPI, HTTPException

from backend.core.inference import generate_explanation, preload_model

app = FastAPI()
MODEL_READY = False
MODEL_ERROR = ""
MODEL_ALLOW_FALLBACK = os.getenv("MODEL_ALLOW_FALLBACK", "true").strip().lower() in {"1", "true", "yes", "on"}


def _fallback_explanation(architecture: dict) -> str:
    nodes = architecture.get("nodes", []) if isinstance(architecture, dict) else []
    edges = architecture.get("edges", []) if isinstance(architecture, dict) else []

    labels = []
    for node in nodes:
        if isinstance(node, dict):
            node_id = node.get("id")
            if isinstance(node_id, str) and node_id:
                labels.append(node_id)

    components_line = (
        f"Components: Main components are {', '.join(labels)}."
        if labels
        else "Components: Main components are frontend, backend, and database."
    )

    if edges and isinstance(edges, list):
        flow_steps = []
        for edge in edges:
            if not isinstance(edge, dict):
                continue
            src = edge.get("source")
            dst = edge.get("target")
            if isinstance(src, str) and isinstance(dst, str):
                flow_steps.append(f"{src} -> {dst}")
        if flow_steps:
            data_flow_line = f"Data flow: {'; '.join(flow_steps[:4])}."
        else:
            data_flow_line = "Data flow: Frontend communicates with backend, which interacts with database."
    else:
        data_flow_line = "Data flow: Frontend communicates with backend, which interacts with database."

    architecture_type_line = "Architecture type: Layered architecture."
    return f"{components_line}\n\n{data_flow_line}\n\n{architecture_type_line}"


@app.on_event("startup")
async def preload_inference_model() -> None:
    global MODEL_READY, MODEL_ERROR
    try:
        preload_model()
        MODEL_READY = True
        MODEL_ERROR = ""
    except Exception as exc:  # noqa: BLE001
        MODEL_READY = False
        MODEL_ERROR = str(exc)
        if not MODEL_ALLOW_FALLBACK:
            raise
        # Keep service alive in degraded mode so docker-compose can start without GPU.
        print(f"Model preload failed, running in fallback mode: {MODEL_ERROR}")


@app.get("/healthz")
async def healthz() -> dict:
    if MODEL_READY:
        return {"status": "ok", "mode": "model"}
    return {"status": "ok", "mode": "fallback", "detail": MODEL_ERROR}


@app.post("/infer")
async def infer(payload: dict):
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Invalid request body")

    architecture = payload.get("architecture")
    if architecture is None:
        raise HTTPException(status_code=400, detail="Provide 'architecture' in payload")

    if MODEL_READY:
        explanation = generate_explanation(architecture)
    else:
        explanation = _fallback_explanation(architecture)
    return {"explanation": explanation}
