#!/usr/bin/env python3
from fastapi import FastAPI, HTTPException

from backend.core.inference import generate_explanation

app = FastAPI()


@app.post("/explain")
async def explain(payload: dict):
    architecture = payload.get("architecture") if isinstance(payload, dict) else None
    if architecture is None:
        raise HTTPException(status_code=400, detail="Missing 'architecture' in request body")

    explanation = generate_explanation(architecture)
    return {"explanation": explanation}
