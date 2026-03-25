#!/usr/bin/env python3
import json
import os
import time

import timm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _cuda_stats() -> dict:
    if not torch.cuda.is_available():
        return {
            "gpu": "cpu",
            "memory_allocated_mb": 0.0,
            "memory_reserved_mb": 0.0,
            "max_memory_allocated_mb": 0.0,
        }

    device = torch.cuda.current_device()
    return {
        "gpu": torch.cuda.get_device_name(device),
        "memory_allocated_mb": round(torch.cuda.memory_allocated(device) / (1024**2), 2),
        "memory_reserved_mb": round(torch.cuda.memory_reserved(device) / (1024**2), 2),
        "max_memory_allocated_mb": round(torch.cuda.max_memory_allocated(device) / (1024**2), 2),
    }


def main() -> None:
    os.environ.setdefault("HF_HOME", "./.cache/huggingface")

    t0 = time.perf_counter()
    convnext = timm.create_model("convnext_tiny", pretrained=True)
    convnext.eval()
    convnext_load_s = time.perf_counter() - t0

    q0 = time.perf_counter()
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        load_in_4bit=True,
    )
    qwen_load_s = time.perf_counter() - q0

    architecture = {"nodes": ["frontend", "backend", "database"]}
    prompt = (
        "Explain this architecture briefly.\\n"
        f"Architecture JSON: {json.dumps(architecture)}"
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    model_inputs = {k: v.to(model.device) for k, v in inputs.items()}

    g0 = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(
            **model_inputs,
            max_new_tokens=32,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
        )
    gen_s = time.perf_counter() - g0

    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    stats = _cuda_stats()

    print("=== Sanity Inference ===")
    print(f"ConvNeXt load time (s): {convnext_load_s:.2f}")
    print(f"Qwen load time (s): {qwen_load_s:.2f}")
    print(f"Generation time (s): {gen_s:.2f}")
    print(f"GPU used: {stats['gpu']}")
    print(f"Memory allocated (MB): {stats['memory_allocated_mb']}")
    print(f"Memory reserved (MB): {stats['memory_reserved_mb']}")
    print(f"Max memory allocated (MB): {stats['max_memory_allocated_mb']}")
    print("Output:")
    print(text)


if __name__ == "__main__":
    main()
