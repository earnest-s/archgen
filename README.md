# ArchitectAI

ArchitectAI is a local-first architecture generation and editing app.

- Frontend: React + Vite + React Flow visual editor
- Backend: FastAPI with in-process Qwen + LoRA inference

## Local Setup

Create and activate the Python environment:

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt --no-cache
```

Set cache directories:

```bash
export HF_HOME=./.cache/huggingface
export TRANSFORMERS_CACHE=./.cache/huggingface
export TORCH_HOME=./.cache/torch
```

## Run Backend

From the project root:

```bash
uvicorn backend.api.main:app --host 127.0.0.1 --port 8000
```

Notes:
- The backend is GPU-only and will fail fast if CUDA is unavailable.
- Model weights are loaded at startup.

## Run Frontend

In a separate terminal:

```bash
cd frontend
npm install
npm run dev
```

The frontend is configured to call the backend at `http://localhost:8000` via [frontend/.env.local](frontend/.env.local).

## Open In Browser

- Frontend: http://localhost:5173
- Backend: http://localhost:8000

## Health Check

- Backend: `GET /healthz`

## Notes

- No Docker files are used in this setup.
- If the backend fails on startup, check that the host GPU is visible to PyTorch.
