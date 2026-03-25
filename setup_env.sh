#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(pwd)"
mkdir -p "$ROOT_DIR/.tmp" "$ROOT_DIR/.cache/huggingface" "$ROOT_DIR/.cache/torch"

export TMPDIR="$ROOT_DIR/.tmp"
export TEMP="$ROOT_DIR/.tmp"
export TMP="$ROOT_DIR/.tmp"

export HF_HOME="$ROOT_DIR/.cache/huggingface"
export TRANSFORMERS_CACHE="$ROOT_DIR/.cache/huggingface"
export TORCH_HOME="$ROOT_DIR/.cache/torch"

source /home/earnest/.var/app/com.visualstudio.code/bin/env

rm -rf "$ROOT_DIR/.venv"
rm -rf "$ROOT_DIR/.tmp"/*
rm -rf "$ROOT_DIR/.cache"/*
rm -rf "$HOME/.cache/huggingface"
rm -rf "$HOME/.cache/torch"

uv venv
source .venv/bin/activate
uv pip install -r requirements.txt --no-cache

python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'); assert torch.cuda.is_available(), 'Torch is not CUDA-enabled'"

echo "setup complete"
