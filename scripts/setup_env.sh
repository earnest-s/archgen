#!/usr/bin/env bash
set -euo pipefail

uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt --no-cache

echo "export HF_HOME=./.cache/huggingface"
echo "Run: source .venv/bin/activate"
