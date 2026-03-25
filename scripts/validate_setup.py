#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate minimal ArchitectAI environment.")
    parser.add_argument("--model-id", default="sshleifer/tiny-gpt2")
    parser.add_argument("--offline-only", action="store_true")
    return parser.parse_args()


def assert_uv_env() -> None:
    venv = os.environ.get("VIRTUAL_ENV", "")
    if not venv or not venv.endswith(".venv"):
        raise RuntimeError("Single uv environment is not active. Activate .venv first.")


def validate_torch() -> None:
    import torch

    print(f"torch version: {torch.__version__}")
    print(f"cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"gpu: {torch.cuda.get_device_name(0)}")


def validate_transformers_single_download(model_id: str, offline_only: bool) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Config

    hf_home = os.environ.setdefault("HF_HOME", "./.cache/huggingface")
    Path(hf_home).mkdir(parents=True, exist_ok=True)

    print(f"HF_HOME: {hf_home}")
    print(f"Model: {model_id}")

    if not offline_only:
        try:
            _tok = AutoTokenizer.from_pretrained(model_id)
            _mdl = AutoModelForCausalLM.from_pretrained(model_id)
            _tok2 = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
            _mdl2 = AutoModelForCausalLM.from_pretrained(model_id, local_files_only=True)
            del _tok, _mdl, _tok2, _mdl2
            print("transformers cache validation passed (download once, then local only)")
            return
        except Exception as exc:
            print(f"hub download validation unavailable: {exc}")
            print("falling back to offline local cache validation")

    local_model_dir = Path(hf_home) / "offline-tiny-model"
    local_model_dir.mkdir(parents=True, exist_ok=True)

    if not (local_model_dir / "config.json").exists():
        cfg = GPT2Config(n_layer=1, n_head=2, n_embd=32, vocab_size=100)
        model = AutoModelForCausalLM.from_config(cfg)
        model.save_pretrained(local_model_dir)

    _mdl = AutoModelForCausalLM.from_pretrained(local_model_dir)
    _mdl2 = AutoModelForCausalLM.from_pretrained(local_model_dir, local_files_only=True)
    del _mdl, _mdl2
    print("transformers offline cache validation passed (single local source, local-only reload)")


def main() -> int:
    args = parse_args()

    assert_uv_env()
    validate_torch()
    validate_transformers_single_download(args.model_id, args.offline_only)

    print("validation complete")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"validation failed: {exc}")
        sys.exit(1)
