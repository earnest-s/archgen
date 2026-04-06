#!/usr/bin/env python3
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.core.inference import generate_architecture, preload_model

app = FastAPI()

FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "").strip()

if FRONTEND_ORIGIN:
    allowed_origins = [origin.strip() for origin in FRONTEND_ORIGIN.split(",") if origin.strip()]
else:
    allowed_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def preload_inference_model() -> None:
    preload_model()
    _ = generate_architecture("A frontend web client calls an API service which writes to a postgres database.", deterministic=True)
    print("MODEL READY")


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}


@app.post("/explain")
async def explain(payload: dict):
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Invalid request body")

    text = payload.get("text")
    if not isinstance(text, str) or not text.strip():
        raise HTTPException(status_code=400, detail="Provide non-empty 'text' in request body")

    try:
        architecture, raw_output = generate_architecture(text)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Model inference failed: {exc}") from exc

    return {"architecture": architecture, "raw_model_output": raw_output}
