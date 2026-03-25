#!/usr/bin/env python3
from fastapi import FastAPI, HTTPException

from backend.core.inference import generate_explanation, preload_model

app = FastAPI()


@app.on_event("startup")
async def preload_inference_model() -> None:
    preload_model()


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}


@app.post("/infer")
async def infer(payload: dict):
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Invalid request body")

    architecture = payload.get("architecture")
    if architecture is None:
        raise HTTPException(status_code=400, detail="Provide 'architecture' in payload")

    explanation = generate_explanation(architecture)
    return {"explanation": explanation}
