# ArchitectAI

ArchitectAI is a local-first architecture generation and editing workspace.

It combines:
- A FastAPI backend that runs local model inference and returns architecture JSON.
- A React + React Flow frontend that renders, edits, styles, and exports diagrams.

## What It Does

- Generates architecture graphs from plain-language prompts.
- Validates and normalizes model output into safe graph structures.
- Opens results in an interactive editor with:
	- node and edge editing
	- drag-and-drop node creation
	- theme toggle (light/dark)
	- PNG export
	- local persistence for prompt + latest architecture

## Tech Stack

- Backend: FastAPI, PyTorch, Transformers, PEFT (LoRA), BitsAndBytes
- Frontend: React 18, TypeScript, Vite 5, React Flow

## Project Structure

```text
backend/
	api/main.py            # FastAPI app and endpoints
	core/inference.py      # Model loading, inference, output validation
frontend/
	src/App.tsx            # Prompt UI, API call, persisted app state
	src/components/        # Diagram editor and interaction logic
	src/index.css          # Theme and editor styling
requirements.txt         # Python dependencies
```

## Prerequisites

- Linux with NVIDIA GPU + CUDA available to PyTorch
- Python 3.10+
- Node.js 18+ and npm

The backend is GPU-only and fails fast if CUDA is unavailable.

## Quick Start

### 1. Set up Python environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional (recommended) local cache paths:

```bash
export HF_HOME=./.cache/huggingface
export TRANSFORMERS_CACHE=./.cache/huggingface
export TORCH_HOME=./.cache/torch
```

### 2. Start backend

From project root:

```bash
uvicorn backend.api.main:app --host 127.0.0.1 --port 8000
```

Model initialization occurs at startup (`preload_model` + smoke test).

### 3. Start frontend

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

## URLs

- Frontend: http://localhost:5173
- Backend: http://127.0.0.1:8000

## Backend API

### `GET /healthz`
Returns service health.

Example response:

```json
{ "status": "ok" }
```

### `POST /explain`
Generates an architecture graph from input text.

Request body:

```json
{ "text": "A frontend calls an API that writes to postgres and publishes jobs." }
```

Successful response shape:

```json
{
	"architecture": {
		"nodes": [{ "id": "frontend", "type": "ui" }],
		"edges": [{ "source": "frontend", "target": "api", "label": "HTTP" }]
	},
	"raw_model_output": "..."
}
```

## Frontend Behavior Notes

- The app calls `http://127.0.0.1:8000/explain`.
- Prompt input and the latest generated architecture are saved in localStorage.
- Theme preference is saved in localStorage.
- PNG export captures the React Flow viewport.

## Troubleshooting

- Backend startup fails with GPU/CUDA errors:
	- Verify CUDA visibility and PyTorch GPU support.
	- Confirm your model checkpoint path exists: `checkpoints/qwen_lora`.

- Frontend cannot reach backend:
	- Ensure backend runs on `127.0.0.1:8000`.
	- Check browser network errors for `/explain`.

- PNG export warns about remote stylesheet access:
	- This is commonly caused by cross-origin CSS/font rules in the browser.
	- Current export logic avoids inlining remote font CSS.

- Chrome console shows message channel/extension runtime errors:
	- These are often browser extension-originated, not app runtime faults.

## Development Commands

From `frontend/`:

```bash
npm run dev
npm run build
npm run preview
```

From project root:

```bash
uvicorn backend.api.main:app --host 127.0.0.1 --port 8000
```
