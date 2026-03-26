# ArchitectAI

ArchitectAI is a full-stack architecture generation and editing app:

- Frontend: React + Vite + React Flow visual editor
- Backend: FastAPI with in-process Qwen + LoRA inference (GPU required)

The project is containerized and designed to run with one command.

## Quick Start (Docker Compose)

From project root:

```bash
docker compose up --build
```

Then open:

- Frontend: http://localhost:5173
- Backend API: http://localhost:8000

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

- Build context: `backend/`, Dockerfile `backend/Dockerfile`
- Runtime port: `8000`
- Entry command:
	- `uvicorn backend.api.main:app --host 0.0.0.0 --port 8000`
- Loads and warms up the real model at startup
- Fails fast if CUDA is unavailable or model is not on GPU

## GPU Requirements

The backend is GPU-only. There is no mock/rule fallback path.

If Docker reports GPU vendor/runtime errors, configure NVIDIA runtime on host and retry:

```bash
nvidia-smi
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
docker run --rm --gpus all nvidia/cuda:12.3.2-runtime-ubuntu22.04 nvidia-smi
```

## Environment Variables

Configured via `.env` (example in `.env.example`):

- `VITE_API_URL`
	- Frontend API base URL at build time
	- Default: `http://backend:8000`
- `FRONTEND_ORIGIN`
	- Backend CORS allow-list (comma-separated)
	- Default: `http://localhost:5173`
- `NVIDIA_VISIBLE_DEVICES`
	- NVIDIA runtime visible devices
	- Default: `all`
- `NVIDIA_DRIVER_CAPABILITIES`
	- NVIDIA runtime capabilities
	- Default: `compute,utility`

## Verify GPU In Container

After startup:

```bash
docker compose exec backend python3 -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
docker compose exec backend nvidia-smi
```

## Health Endpoints

- Backend: `GET /healthz`

Compose healthcheck uses backend endpoint for startup ordering.

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
