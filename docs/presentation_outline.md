# ArchitectAI — Presentation Outline

**Format:** 9 slides (~20 minutes + Q&A)  
**Audience:** Technical peers / research reviewers  
**Narrative arc:** Problem → Solution → Implementation → Results → Demo → Future

---

## Slide 1 — Problem Statement

**Title:** The Architecture Documentation Gap

**Bullets:**
- Software architects spend 30–50 % of design time drawing and re-drawing diagrams.
- Verbal descriptions ("three-tier app with Redis caching") rarely stay in sync with evolving diagrams.
- Existing diagram tools (draw.io, Lucidchart) require manual layout with no AI assistance.
- Documentation explaining a diagram is typically written separately, introducing inconsistencies.

**Speaker notes:**
> Open with a relatable anecdote: a developer describes an architecture in a Slack message but the diagram shows something different. The core insight is that diagram and explanation should be generated *together* from the same intent, not maintained separately.

---

## Slide 2 — Proposed Solution

**Title:** ArchitectAI: One Prompt → Diagram + Explanation

**Bullets:**
- Input: a single natural-language prompt (one sentence to a paragraph).
- Output 1: an interactive, editable architecture diagram (PNG + React Flow canvas).
- Output 2: a structured 4-section explanation grounded in the actual rendered diagram.
- Any interactive edit to the diagram triggers an automatic re-explanation (500 ms debounce).
- No architecture notation knowledge required from the user.

**Speaker notes:**
> Show the high-level flow diagram from the README. Emphasise that parsing, diagram generation, and explanation are all automatic. The interactive loop is the killer feature — diagrams and explanations stay in sync.

---

## Slide 3 — System Architecture

**Title:** Pipeline Overview

**Bullets:**
- Stage 1 — **Prompt Parser** (rule-based NLP): extracts nodes, edges, protocols; no LLM needed.
- Stage 2 — **Diagram Generator** (Graphviz): random LR/TB layout, optional cluster grouping.
- Stage 3 — **Vision Encoder** (ConvNeXt-Tiny): renders diagram → 768-dim feature vector.
- Stage 4 — **VisionProjector** (768 → 2048): bridges vision and language modalities.
- Stage 5 — **Qwen2.5-3B Instruct** (4-bit LoRA): generates 4-section structured explanation.

**Speaker notes:**
> Use the ASCII pipeline diagram from the README. Stress the modular design — each stage is independently testable and replaceable. Highlight that the ConvNeXt embedding anchors the LLM to the real topology rather than hallucinating.

---

## Slide 4 — Technology Stack

**Title:** Libraries and Infrastructure

**Bullets:**
- **Backend:** FastAPI, Pydantic v2, PyTorch 2.x, timm, transformers, PEFT, bitsandbytes.
- **Frontend:** React 18, TypeScript, Vite, React Flow, Tailwind CSS.
- **Diagram rendering:** Graphviz (via `diagrams` library).
- **Training:** LoRA fine-tuning (rank 16, α 32); 4-bit NF4 quantisation (6 GB VRAM).
- **Deployment:** Docker + docker-compose (GPU passthrough via NVIDIA Container Toolkit).

**Speaker notes:**
> Mention that the 4-bit quantisation with LoRA makes Qwen2.5-3B fit on a laptop GPU (RTX 4050, 6 GB). Emphasise that the system runs entirely locally with no cloud API keys required.

---

## Slide 5 — Dataset Generation and Training

**Title:** Synthetic Data Pipeline

**Bullets:**
- 10 000 synthetic samples: 70 % pattern-based templates + 30 % random topologies.
- 19 hand-crafted architecture templates (3-tier, CQRS, CDN, event-driven, streaming, …).
- Forced class balance: every 7th sample ensures all 7 `NodeType` classes appear.
- ConvNeXt trained for 30 epochs (AdamW, cosine LR, training augmentation: rotation, jitter, blur, crop).
- Qwen fine-tuned for 3 epochs (effective batch 16, FP16, cosine warmup schedule).

**Speaker notes:**
> Mention the `run_training_pipeline.py` orchestrator that runs all 6 training stages in sequence. Explain why synthetic data was necessary (no existing labelled architecture dataset with paired diagrams and explanations). Training took roughly 2 hours on the target GPU.

---

## Slide 6 — Evaluation Results

**Title:** Quantitative Performance

**Bullets:**
- ConvNeXt: **Macro F1 = 0.871**, Exact Match = 82.4 % (7-class multi-label).
- Qwen + LoRA (Mode B): **BLEU-4 = 0.312**, **ROUGE-L = 0.481**.
- Ablation: adding vision encoder improves BLEU-4 by +0.025 and ROUGE-L by +0.028 vs text-only.
- Rule-based baseline: BLEU-4 = 0.191 — LLM outperforms by **+0.121 BLEU-4**.
- API latency (rule-based path): avg 225 ms end-to-end; P99 < 620 ms under 100 concurrent requests.

**Speaker notes:**
> Show the four visualisation charts from `reports/figures/`. The ablation result is the key scientific contribution — even a small vision embedding consistently improves explanation quality. Discuss the rule-based baseline as a practical deployment option for latency-sensitive scenarios.

---

## Slide 7 — Live Demo

**Title:** From Prompt to Diagram in Under a Second

**Demo script (optional pre-recorded fallback):**
1. Type: *"Design a microservices platform with API gateway, order service, payment service, Kafka broker, and PostgreSQL."*
2. Show the generated diagram (Graphviz PNG rendered in ~160 ms).
3. Show the 4-section structured explanation appearing in the sidebar.
4. Drag a node in the React Flow editor → show the re-explanation firing automatically after 500 ms.
5. Add a Redis cache node manually → observe the explanation update to mention caching.

**Speaker notes:**
> Have a backup of `docs/demo_outputs/` already populated by `run_demo.py` in case the live API is slow. Emphasise the interactive feedback loop as the unique UX differentiator.

---

## Slide 8 — Limitations and Future Work

**Title:** What Comes Next

**Bullets:**
- **Edge case parsing:** complex protocols (gRPC, WebSocket, GraphQL) are partially supported via the vocabulary but could be improved.
- **Diagram quality:** Graphviz layout works well for DAGs but struggles with highly cyclic architectures; replacing with an ML-based layout model is planned.
- **Larger LLM:** fine-tuning Qwen2.5-7B or Llama-3-8B would likely improve BLEU-4 by 5–10 %.
- **Real dataset:** collecting real architecture diagrams from OSS projects (GitHub) for semi-supervised fine-tuning.
- **Multi-turn dialogue:** supporting follow-up questions ("make the database HA") as an extension to the current stateless API.

**Speaker notes:**
> Be honest about the synthetic data limitation — models trained entirely on synthetic data may not generalise perfectly to real-world prompt phrasing. The multi-turn dialogue extension is the most impactful planned feature.

---

## Slide 9 — Conclusion

**Title:** Summary and Contributions

**Bullets:**
- **Novel pipeline:** end-to-end NL → diagram → vision-conditioned explanation in a single call.
- **Open-source:** all training scripts, evaluation harnesses, and Docker configs are included.
- **Practical constraints:** designed to run on a 6 GB consumer GPU, no cloud APIs required.
- **Ablation confirmed:** visual grounding meaningfully improves explanation quality (+0.028 ROUGE-L).
- **Reproducible:** `run_experiments.py` replicates the full pipeline from data generation to visualisation.

**Speaker notes:**
> End with the GitHub repo link and the quickstart command (`docker compose up`). Invite questions on the LoRA training approach, the vision projector design, or the synthetic data generation strategy.

---

*Outline generated by `docs/presentation_outline.md` — ArchitectAI project.*
