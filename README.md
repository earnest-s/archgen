# ArchitectAI

ArchitectAI is a full-stack architecture generation and editing app:

- Frontend: React + Vite + React Flow visual editor
- Backend: FastAPI API gateway
- Model service: FastAPI inference service (separate container)

The project is containerized and designed to run with one command.

## Quick Start (Docker Compose)

From project root:

```bash
docker compose up --build
```

Then open:

- Frontend: http://localhost:5173
- Backend API: http://localhost:8000
- Model service: http://localhost:9000

Stop services:

```bash
docker compose down
```

## Flatpak VS Code Note

If your shell is inside Flatpak VS Code and `docker` is not visible, run host Docker with:

```bash
flatpak-spawn --host sh -lc 'cd /home/earnest/Downloads/Ai_Architecture_Generator && docker compose up --build'
```

Validate compose config from the same environment:

```bash
flatpak-spawn --host sh -lc 'cd /home/earnest/Downloads/Ai_Architecture_Generator && docker compose config'
```

## Services

### frontend

- Build context: `frontend/`
- Dockerfile: `frontend/Dockerfile`
- Runtime port: `5173`
- Serves production build using `serve`

### backend

- Build context: project root, Dockerfile `backend/Dockerfile`
- Runtime port: `8000`
- Entry command:
	- `uvicorn backend.api.main:app --host 0.0.0.0 --port 8000`
- Handles parser + orchestration
- Calls model service when `MODEL_URL` is set

### model

- Build context: project root, Dockerfile `backend/Dockerfile`
- Runtime port: `9000`
- Entry command:
	- `uvicorn backend.api.model_service:app --host 0.0.0.0 --port 9000`
- Exposes inference endpoint for backend:
	- `POST /infer`

## Environment Variables

Configured via `.env` (example in `.env.example`):

- `VITE_API_URL`
	- Frontend API base URL at build time
	- Default: `http://localhost:8000`
- `FRONTEND_ORIGIN`
	- Backend CORS allow-list (comma-separated)
	- Default: `http://localhost:5173`
- `MODEL_URL`
	- Backend to model internal URL
	- Default: `http://model:9000`
- `MODEL_ALLOW_FALLBACK`
	- `true`: if model preload fails, model service stays up and returns fallback responses
	- `false`: fail fast when GPU/model load fails (recommended when validating real GPU inference)
	- Default: `true`
- `MODEL_GPUS`
	- GPU request passed to Docker Compose for model service
	- Default: `all`
- `NVIDIA_VISIBLE_DEVICES`
	- NVIDIA runtime visible devices
	- Default: `all`
- `NVIDIA_DRIVER_CAPABILITIES`
	- NVIDIA runtime capabilities
	- Default: `compute,utility`

## GPU Mode (No Fallback)

To force real model inference and fail if GPU is not available, set this in `.env`:

```bash
MODEL_ALLOW_FALLBACK=false
MODEL_GPUS=all
NVIDIA_VISIBLE_DEVICES=all
NVIDIA_DRIVER_CAPABILITIES=compute,utility
```

Then restart:

```bash
docker compose down
docker compose up --build
```

Quick checks:

```bash
docker compose exec model python -c "import torch; print('cuda', torch.cuda.is_available()); print('count', torch.cuda.device_count())"
docker compose logs -f model
```

If CUDA is unavailable inside container, install NVIDIA Container Toolkit on host and restart Docker.

## Health Endpoints

- Backend: `GET /healthz`
- Model: `GET /healthz`

Compose healthchecks use these endpoints for startup ordering.

## Docker Files

- Root compose file: `docker-compose.yml`
- Root ignore: `.dockerignore`
- Frontend ignore: `frontend/.dockerignore`
- Backend ignore: `backend/.dockerignore`

## Local (Non-Docker) Setup

If you want to run locally without containers:

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt --no-cache
```

Suggested cache location:

```bash
export HF_HOME=./.cache/huggingface
```

Run backend:

```bash
uvicorn backend.api.main:app --host 127.0.0.1 --port 8000
```

Run frontend:

```bash
cd frontend
npm install
npm run dev
```
