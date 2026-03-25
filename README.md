# ArchitectAI (Minimal Setup)

Lightweight, reproducible project scaffold optimized for low disk usage.

## Project layout

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

## 1) Create a single uv environment

```bash
uv venv .venv
source .venv/bin/activate
```

Use only one environment for everything.

## 2) Install dependencies (uv only)

```bash
uv pip install -r requirements.txt --no-cache
```

Rules enforced:
- No pip install
- No wheel caching
- Single environment only

## 3) Model cache control (single location)

Always set:

```bash
export HF_HOME=./.cache/huggingface
```

After first model download, reload using local files only.

## 4) Checkpoint policy

Keep only final artifacts:
- checkpoints/convnext_best.pt
- checkpoints/qwen_lora/

Do not keep intermediate checkpoints. Overwrite instead of creating new files.

## 5) Data policy

- Maximum dataset size: 1000 samples
- Avoid saving large PNG sets by default
- Prefer on-the-fly generation unless explicitly needed

Use:

```bash
python scripts/generate_synthetic.py --num-samples 1000
```

## 6) Validation

```bash
export HF_HOME=./.cache/huggingface
python scripts/validate_setup.py --model-id sshleifer/tiny-gpt2
```

Validation checks:
- uv environment is active
- torch import and GPU availability report
- transformers model is downloaded once, then loaded from local cache only

## Disk target

Keep total project footprint around 3-5GB by avoiding duplicate checkpoints, repeated model downloads, and unnecessary generated image datasets.
