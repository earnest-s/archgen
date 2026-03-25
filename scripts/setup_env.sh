#!/usr/bin/env bash
set -euo pipefail

uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt --no-cache

python -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('cuda_version', torch.version.cuda)"

echo "export HF_HOME=./.cache/huggingface"
echo "Run: source .venv/bin/activate"
