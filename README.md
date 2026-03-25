# ArchitectAI (Minimal, Disk-Efficient Setup)

## Required Structure

- backend/
- frontend/
- scripts/
- data/synthetic/
- checkpoints/
- reports/
- docs/
- .gitignore
- requirements.txt
- README.md

## Dependency Installation (uv only)

Use one environment only:

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt --no-cache
```

Rules:
- Do not use pip install
- Do not cache wheels
- Do not duplicate installs

## Minimal Requirements

requirements.txt contains only:
- torch
- torchvision
- timm
- transformers
- peft
- bitsandbytes
- accelerate
- fastapi
- uvicorn
- pydantic
- python-multipart
- pillow
- matplotlib
- scikit-learn
- nltk
- rouge-score

## Model Cache Control

Use one HuggingFace cache location:

```bash
export HF_HOME=./.cache/huggingface
```

After first download, reload with local_files_only=True.

## Checkpoint Rules

Keep only final checkpoints:
- checkpoints/convnext_best.pt
- checkpoints/qwen_lora/

No intermediate checkpoints. Overwrite existing files.

## Data Rules

- Maximum dataset size: 1000 samples
- Avoid large PNG datasets
- Prefer on-the-fly generation where possible

## Validation

1. Verify uv environment works
2. Verify torch detects GPU
3. Verify transformers loads once, then local-only

Example validation commands:

```bash
source .venv/bin/activate
export HF_HOME=./.cache/huggingface
python -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available())"
python -c "from transformers import AutoTokenizer; m='sshleifer/tiny-gpt2'; AutoTokenizer.from_pretrained(m); AutoTokenizer.from_pretrained(m, local_files_only=True); print('ok')"
```

## Goal

Keep the project lightweight and reproducible, targeting total size around 3-5GB.
