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

    labels = []
    if "frontend" in lowered:
        labels.append("frontend")
    if "backend" in lowered:
        labels.append("backend")
    if "database" in lowered or " db" in lowered or "db " in lowered:
        labels.append("database")

    if not labels:
        labels = ["frontend", "backend", "database"]

    return {"nodes": labels}


@app.post("/explain")
async def explain(payload: dict):
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Invalid request body")

    architecture = None

    text = payload.get("text")
    if isinstance(text, str) and text.strip():
        architecture = _parse_text_to_architecture(text.strip())
    elif payload.get("architecture") is not None:
        architecture = payload.get("architecture")

    if architecture is None:
        raise HTTPException(status_code=400, detail="Provide either 'architecture' (JSON) or 'text' (natural language).")

    explanation = generate_explanation(architecture)
    return {"explanation": explanation}
