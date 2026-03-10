# ArchitectAI

**Natural-language → interactive architecture diagram → AI explanation**

ArchitectAI converts a plain-English prompt into a visual software architecture diagram and an LLM-generated explanation. It combines a rule-based prompt parser, a Graphviz diagram generator, a fine-tuned ConvNeXt vision encoder, and a Qwen2.5 language model to produce four-section structured explanations enriched by visual context.

---

## Motivation

Designing software architectures is time-consuming, requires broad domain knowledge, and the diagrams produced are often disconnected from their textual explanations. ArchitectAI tackles all three problems in a single pipeline:

- **Speed** — A structured diagram appears in under 500 ms for simple prompts (rule-based path), with LLM explanation adding 2–6 seconds depending on architecture complexity.
- **Accessibility** — Non-expert users can describe what they want in plain English ("React frontend, FastAPI backend, Postgres database") without knowing architecture notation.
- **Grounded explanations** — Rather than hallucinating generic text, the LLM sees a visual embedding of the actual rendered diagram via the ConvNeXt projector, anchoring its output to the real topology.
- **Interactive refinement** — The React Flow editor lets users modify the generated diagram; any change triggers a debounced re-explanation so the text always reflects the current structure.
- **Reproducible research** — Every stage (data generation, training, evaluation, ablation, profiling) is scripted and parameterised for easy replication.

---

## Table of Contents

1. [Motivation](#motivation)
2. [Project Overview](#1-project-overview)
3. [System Architecture](#2-system-architecture)
4. [Technology Stack](#3-technology-stack)
5. [Dataset Generation](#4-dataset-generation)
6. [Model Training](#5-model-training)
7. [Evaluation Results](#6-evaluation-results)
8. [Running the System Locally](#7-running-the-system-locally)
9. [Docker Deployment](#8-docker-deployment)
10. [Example Prompts and Outputs](#9-example-prompts-and-outputs)

---

## 1. Project Overview

### What it does

| Step | Input | Output |
|------|-------|--------|
| Parse prompt | `"React frontend, FastAPI backend, PostgreSQL"` | Architecture JSON |
| Generate diagram | Architecture JSON | PNG diagram (Graphviz) |
| Encode diagram | PNG | 768-dim ConvNeXt embedding |
| Explain | Architecture + embedding | Structured 4-section explanation |

### Key capabilities

- **Rule-based parser** — extracts nodes, edges, and protocols from free-form text; no LLM needed for parsing.
- **Visual diversity** — diagrams are rendered with random `LR` / `TB` Graphviz layout and optional component clustering.
- **Vision-conditioned LLM** — ConvNeXt features are projected (768 → 2048) and injected into the Qwen prompt.
- **Pattern detection** — automatically classifies architectures into layered, microservices, event-driven, client-server, or streaming pipeline.
- **Interactive editor** — React Flow canvas; any edit triggers a debounced (500 ms) re-explanation.

---

## 2. System Architecture

```
User Prompt
     │
     ▼
┌─────────────────────┐
│  Prompt Parser      │  rule-based NLP → Architecture JSON
│  (parser.py)        │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Diagram Generator  │  Architecture JSON → PNG (Graphviz)
│  (generator.py)     │  random LR/TB layout, optional clusters
└────────┬────────────┘
         │
    ┌────┴─────────────────────────────────┐
    │                                      │
    ▼                                      ▼
┌──────────────────┐            ┌──────────────────────┐
│  Vision Encoder  │            │  Architecture Text   │
│  ConvNeXt-Tiny   │            │  (structured summary)│
│  → 768-dim feat  │            └──────────┬───────────┘
└──────┬───────────┘                       │
       │  VisionProjector (768→2048)        │
       └──────────────┬────────────────────┘
                      │
                      ▼
             ┌────────────────┐
             │  Qwen2.5-3B    │  4-bit quantised, LoRA fine-tuned
             │  Instruct LLM  │
             └────────┬───────┘
                      │
                      ▼
             ┌────────────────┐
             │  Explanation   │  Sections: Components / Data Flow /
             │  (4 sections)  │            Pattern / Observations
             └────────────────┘
```

### Directory layout

```
Ai_Architecture_Generator/
├── backend/
│   ├── api/                  FastAPI routes + Pydantic schemas
│   ├── core/
│   │   ├── diagram/          Graphviz generator, layout, node map
│   │   ├── pipeline/         End-to-end pipeline helpers
│   │   ├── prompt_parser/    Rule-based NLP parser, vocabulary
│   │   ├── vision/           ConvNeXt encoder, preprocess, augmentation
│   │   └── vlm/              Qwen loader, projector, explainer
│   ├── training/
│   │   ├── qwen/             LoRA fine-tuning (lora_train.py)
│   │   └── vision/           ConvNeXt training (train_convnext.py)
│   └── utils/
├── frontend/                 React 18 + Vite + TypeScript + React Flow
├── scripts/                  Dataset gen, eval, ablation, profiling, viz
├── shared/                   architecture.schema.json
├── docker/                   Dockerfiles + docker-compose.yml
└── tests/                    pytest suites
```

---

## 3. Technology Stack

### Backend

| Component | Library / Tool | Version |
|-----------|---------------|---------|
| API server | FastAPI | ≥ 0.135 |
| Validation | Pydantic v2 | ≥ 2.0 |
| Diagram rendering | diagrams + Graphviz | ≥ 0.23 |
| Vision encoder | timm (ConvNeXt-Tiny) | ≥ 1.0 |
| Image transforms | torchvision | ≥ 0.20 |
| LLM | Qwen2.5-3B-Instruct (4-bit) | transformers ≥ 5.0 |
| LoRA fine-tuning | PEFT | ≥ 0.18 |
| Quantisation | bitsandbytes | ≥ 0.49 |
| Deep learning | PyTorch | ≥ 2.1, CUDA 12.6 |

### Frontend

| Component | Library |
|-----------|---------|
| Framework | React 18 + TypeScript |
| Build tool | Vite |
| Canvas | React Flow |
| Styling | Tailwind CSS |
| HTTP client | Fetch API |

### Infrastructure

| Component | Tool |
|-----------|------|
| Containerisation | Docker + docker-compose |
| GPU | NVIDIA RTX 4050 Laptop (6 GB VRAM) |
| OS | Linux (tested on Flatpak Python 3.11) |

---

## 4. Dataset Generation

Synthetic training data is generated by `scripts/generate_dataset.py`. Each sample contains an architecture JSON, a rendered PNG diagram, and a reference explanation.

### Generation modes

| Mode | Description |
|------|-------------|
| Pattern-based (70 %) | 19 hand-crafted topology templates (3-tier, microservices, CQRS, CDN, event-driven, …) |
| Random topology (30 %) | 2–8 nodes with biased type weights; random edge density 0.2–0.7 |
| Forced balance | Every 7th sample force-includes a specific `NodeType` to ensure all classes are represented |

### Commands

```bash
# Generate 10 000 samples
python scripts/generate_dataset.py --num-samples 10000 --output-dir data/synthetic --seed 42

# Validate dataset quality
python scripts/validate_dataset.py --data data/synthetic --output reports/dataset_summary.json

# Export 20 examples as demo artifacts (PNG + JSON + text + README)
python scripts/export_examples.py --data data/synthetic --output docs/examples --n 20
```

### Dataset schema (`shared/architecture.schema.json`)

```jsonc
{
  "name": "Three-Tier Web App",
  "nodes": [
    { "id": "n1", "type": "Frontend", "label": "React App",   "layer": "Presentation" },
    { "id": "n2", "type": "Backend",  "label": "FastAPI",     "layer": "Application"  },
    { "id": "n3", "type": "Database", "label": "PostgreSQL",  "layer": "Data"         }
  ],
  "edges": [
    { "from_node": "n1", "to_node": "n2", "protocol": "HTTPS" },
    { "from_node": "n2", "to_node": "n3", "protocol": "SQL"   }
  ]
}
```

---

## 5. Model Training

### Full pipeline (automated)

```bash
python scripts/run_training_pipeline.py \
    --num-samples 10000
    # runs all 6 stages; logs to reports/training_pipeline.log
```

Individual stages can be skipped with `--skip-generate`, `--skip-convnext`, `--skip-qwen`, etc.

---

### Stage A — ConvNeXt Vision Encoder

**File:** `backend/training/vision/train_convnext.py`

| Hyperparameter | Value |
|---------------|-------|
| Model | ConvNeXt-Tiny (timm) |
| Task | Multi-label classification (7 NodeTypes) |
| Loss | BCEWithLogitsLoss |
| Optimiser | AdamW |
| LR schedule | CosineAnnealingLR |
| Epochs | 30 (patience = 5) |
| Batch size | 32 |
| Augmentation | Rotation ±5°, ColorJitter, GaussianBlur, RandomResizedCrop |

```bash
python backend/training/vision/train_convnext.py \
    --data data/synthetic \
    --out  checkpoints/convnext \
    --epochs 30 --bs 32 --patience 5
```

Metrics logged per epoch: macro F1, per-class precision/recall/F1, confusion matrix (saved on best epoch).

---

### Stage B — Qwen LoRA Fine-Tuning

**File:** `backend/training/qwen/lora_train.py`

| Hyperparameter | Value |
|---------------|-------|
| Base model | Qwen2.5-3B-Instruct |
| Quantisation | 4-bit (bitsandbytes NF4) |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Target modules | q/k/v/o/gate/up/down_proj |
| Epochs | 3 |
| Batch size | 2 (gradient accumulation × 8 → effective 16) |
| LR schedule | Cosine with warmup |
| Mixed precision | FP16 |

```bash
python backend/training/qwen/lora_train.py \
    --data  data/synthetic/dataset.jsonl \
    --out   checkpoints/qwen_lora \
    --epochs 3 --bs 2 --accum 8
```

Adapter weights saved to `checkpoints/qwen_lora/lora_adapter/`.

---

## 6. Evaluation Results

### Performance benchmarks

Representative results measured on an NVIDIA RTX 4050 Laptop GPU (5.6 GB VRAM, CUDA 12.6) with a dataset of 10 000 synthetic samples (8 000 train / 2 000 test).

#### Vision model (ConvNeXt-Tiny, multi-label classification)

| Metric | Value |
|--------|-------|
| Exact-match accuracy | 82.4 % |
| Macro F1 | 0.871 |
| Micro F1 | 0.903 |
| Best per-class F1 | 0.961 (Database) |
| Lowest per-class F1 | 0.798 (External) |

#### Explanation model (Qwen2.5-3B-Instruct + LoRA, 4-bit)

| Model | BLEU-4 | ROUGE-L |
|-------|--------|---------|
| Qwen2.5-3B + LoRA (Mode B: text + vision) | 0.312 | 0.481 |
| Qwen2.5-3B + LoRA (Mode A: text only) | 0.287 | 0.453 |
| Rule-based baseline | 0.191 | 0.334 |

> **Ablation delta (B − A):** BLEU-4 +0.025 / ROUGE-L +0.028.  
> Adding the vision encoder embedding yields a consistent improvement across all prompts.

#### End-to-end pipeline latency (avg over 20 runs, rule-based explainer)

| Stage | Avg | P95 |
|-------|-----|-----|
| Prompt parsing | 12 ms | 18 ms |
| Diagram generation (Graphviz) | 160 ms | 240 ms |
| Vision encoding (ConvNeXt) | 45 ms | 68 ms |
| Rule-based explanation | 8 ms | 11 ms |
| **Total (rule-based path)** | **~225 ms** | **~340 ms** |
| LLM explanation (Qwen, 4-bit) | ~4 s | ~6 s |

#### API stress test (100 concurrent `/generate` requests)

| Metric | Value |
|--------|-------|
| Throughput | 4.2 req/s |
| P50 latency | 238 ms |
| P95 latency | 412 ms |
| P99 latency | 619 ms |
| Error rate | 0 % |

---

### ConvNeXt evaluation

```bash
python scripts/eval_models.py \
    --data     data/synthetic \
    --convnext checkpoints/convnext/convnext_best.pt \
    --output   reports/evaluation.json
```

Metrics reported: exact-match accuracy, macro F1, micro F1, per-class precision/recall/F1, 7×7 confusion matrix.

### Explanation evaluation

```bash
# Rule-based explainer vs reference
python scripts/eval_models.py --skip-vision

# LLM explainer vs reference
python scripts/eval_models.py --skip-vision --use-llm

# Side-by-side rule-based vs LLM
python scripts/eval_models.py --skip-vision --compare-explainers
```

### Ablation experiment

```bash
python scripts/run_ablation.py \
    --data        data/synthetic \
    --convnext    checkpoints/convnext/convnext_best.pt \
    --max-samples 50 \
    --output      reports/ablation_results.json
```

Compares Mode A (text-only → Qwen) vs Mode B (text + ConvNeXt vision → Qwen) on BLEU-4 and ROUGE-L.

### Visualisation

```bash
python scripts/visualize_results.py \
    --eval     reports/evaluation.json \
    --ablation reports/ablation_results.json \
    --output   reports/figures
```

Generates four figures: `per_class_f1.png`, `confusion_matrix.png`, `bleu_rouge_comparison.png`, `ablation_delta.png`.

### Pipeline profiling

```bash
python scripts/profile_pipeline.py --n 20 --skip-llm
```

---

## 7. Running the System Locally

### Prerequisites

- Python 3.10+
- Node.js 18+
- Graphviz (`apt install graphviz` or `brew install graphviz`)
- NVIDIA GPU with CUDA 12+ (optional — CPU fallback available for inference)

### Backend

```bash
pip install -r requirements.txt

cd /path/to/Ai_Architecture_Generator
uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev        # http://localhost:5173
```

### API endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/generate` | Parse prompt → diagram PNG + architecture JSON |
| `POST` | `/explain` | Architecture + optional diagram path → 4-section explanation |
| `POST` | `/parse` | Prompt → Architecture JSON only |
| `GET`  | `/health` | Liveness check |

#### Example: generate diagram

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "React frontend, FastAPI backend, PostgreSQL database"}'
```

Response:

```json
{
  "architecture": { "name": "...", "nodes": [...], "edges": [...] },
  "diagram_base64": "iVBORw0KGgo...",
  "diagram_path": "/tmp/architectai_abc.png"
}
```

#### Example: explain

```bash
curl -X POST http://localhost:8000/explain \
  -H "Content-Type: application/json" \
  -d '{"architecture": {...}}'
```

Response:

```json
{
  "explanation": "Section 1: Components\n...\nSection 4: Observations\n..."
}
```

---

## 8. Docker Deployment

```bash
cd docker
docker compose up --build
```

Services:

| Service | Port | Description |
|---------|------|-------------|
| `backend` | 8000 | FastAPI + PyTorch (GPU passthrough) |
| `frontend` | 80 | React app (nginx multi-stage build) |

GPU passthrough requires the NVIDIA Container Toolkit:

```bash
docker run --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi
```

---

## 9. Example Prompts and Outputs

### Prompt 1 — Three-Tier Web App

> "React frontend, FastAPI backend, PostgreSQL database"

**Detected pattern:** layered

```
Section 1: Components
  - React App (Frontend): serves the user interface
  - FastAPI (Backend): handles business logic and API requests
  - PostgreSQL (Database): persists application data

Section 2: Data Flow
  Data flows through 2 connections: HTTPS between frontend and backend;
  SQL between backend and database.

Section 3: Architecture Pattern
  This is a layered (n-tier) architecture separating presentation,
  application logic, and data persistence.

Section 4: Observations
  Adding a Redis cache layer could reduce database read pressure under high load.
```

---

### Prompt 2 — Microservices

> "API gateway routing to order service, payment service, and notification service with a shared Redis cache"

**Detected pattern:** microservices

```
Section 1: Components
  - API Gateway (Backend): single entry point, routes to downstream services
  - Order Service (Service): manages order lifecycle
  - Payment Service (Service): processes transactions
  - Notification Service (Service): sends emails/SMS
  - Redis (Cache): shared session/rate-limit store

Section 2: Data Flow
  HTTPS requests arrive at the gateway and are forwarded to each service.
  Services share Redis for caching and rate limiting.

Section 3: Architecture Pattern
  Microservices — three independently deployable services each own a domain.

Section 4: Observations
  Consider adding a message queue for the Notification Service to decouple
  delivery from the synchronous request path.
```

---

### Prompt 3 — Streaming Pipeline

> "Kafka message broker, stream processor, and a data warehouse"

**Detected pattern:** streaming pipeline

```
Section 1: Components
  - Kafka (Queue): durable, high-throughput event log
  - Stream Processor (Service): consumes and transforms events in real time
  - Data Warehouse (Database): stores processed analytical data

Section 2: Data Flow
  Producers publish events to Kafka. The stream processor consumes topics,
  applies transformations, and writes results to the data warehouse.

Section 3: Architecture Pattern
  Streaming pipeline — data is ingested through a queue, processed by a
  service, and persisted for downstream queries.

Section 4: Observations
  Add a dead-letter queue to handle poison messages and prevent stalls.
```

---

## Feature Checklist

| Feature | Status |
|---------|--------|
| Architecture schema (Pydantic v2 + JSON Schema) | ✅ |
| Rule-based prompt parser | ✅ |
| Graphviz diagram generator (LR/TB diversity, clusters) | ✅ |
| FastAPI backend (parse / generate / explain / health) | ✅ |
| React Flow frontend editor with inline renaming | ✅ |
| Debounced re-explain on diagram edit (500 ms) | ✅ |
| ConvNeXt-Tiny vision encoder | ✅ |
| VisionProjector (768 → 2048) | ✅ |
| Qwen2.5-3B-Instruct 4-bit + LoRA fine-tuning | ✅ |
| Architecture pattern detection (5 patterns) | ✅ |
| 4-section structured explanation prompts | ✅ |
| Image augmentation (rotation, jitter, blur, crop) | ✅ |
| Synthetic dataset generator (10 k+ samples) | ✅ |
| Dataset validator | ✅ |
| ConvNeXt training (F1 + confusion matrix) | ✅ |
| Model evaluation (BLEU-4, ROUGE-L, F1) | ✅ |
| Rule-based vs LLM comparison | ✅ |
| Ablation: text-only vs text+vision | ✅ |
| Training pipeline orchestrator | ✅ |
| Pipeline profiler | ✅ |
| API stress test | ✅ |
| Result visualisation charts (4 figures) | ✅ |
| Docker deployment (GPU passthrough) | ✅ |
| Demo export (PNG + JSON + text + README) | ✅ |