#!/usr/bin/env python3
import json
import os
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def clean_structured_output(text: str, max_words: int = 150) -> str:
    cleaned = text.strip()

    for marker in ("Human:", "Assistant:", "User:", "###"):
        if marker in cleaned:
            cleaned = cleaned.split(marker, 1)[0].strip()

    section_names = ["Components:", "Data flow:", "Architecture type:"]
    positions = [cleaned.find(name) for name in section_names]

    if all(pos >= 0 for pos in positions) and positions == sorted(positions):
        comp_start, flow_start, arch_start = positions
        components_block = cleaned[comp_start:flow_start].strip()
        flow_block = cleaned[flow_start:arch_start].strip()
        arch_block = cleaned[arch_start:].strip()

        arch_content = arch_block[len("Architecture type:"):].strip()
        sentence_end = -1
        for punct in (".", "!", "?"):
            idx = arch_content.find(punct)
            if idx != -1 and (sentence_end == -1 or idx < sentence_end):
                sentence_end = idx
        if sentence_end != -1:
            arch_content = arch_content[: sentence_end + 1].strip()

        arch_block = f"Architecture type: {arch_content}" if arch_content else "Architecture type:"
        cleaned = "\n".join([components_block, flow_block, arch_block]).strip()
    else:
        short = " ".join(cleaned.split()[:60]).strip()
        cleaned = (
            f"Components: {short}\n"
            "Data flow: Not explicitly provided.\n"
            "Architecture type: Not explicitly provided."
        )

    words = cleaned.split()
    if len(words) > max_words:
        cleaned = " ".join(words[:max_words]).strip()

    return cleaned


def load_samples(dataset_path: Path, limit: int = 5):
    samples = []
    with dataset_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if "architecture" in row and "explanation" in row:
                samples.append(row)
            if len(samples) >= limit:
                break
    return samples


def main() -> None:
    os.environ.setdefault("HF_HOME", "./.cache/huggingface")

    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required for inference")

    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    adapter_path = Path("checkpoints/qwen_lora")
    dataset_path = Path("data/synthetic/dataset.jsonl")

    if not adapter_path.exists():
        raise FileNotFoundError(f"Missing adapter: {adapter_path}")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing dataset: {dataset_path}")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model in 4-bit...")
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        quantization_config=bnb_cfg,
    )

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    model.eval()

    # LoRA validation signal: a PEFT model with at least one adapter config loaded.
    adapter_names = list(model.peft_config.keys())
    print(f"LoRA adapters loaded: {adapter_names}")

    samples = load_samples(dataset_path, limit=5)
    if not samples:
        raise RuntimeError("No valid samples found in dataset")

    model_device = next(model.parameters()).device

    for idx, sample in enumerate(samples, start=1):
        architecture = sample["architecture"]
        ground_truth = sample["explanation"]
        architecture_json = json.dumps(architecture, ensure_ascii=True, indent=2)

        prompt = f"""
Explain the following software architecture clearly and concisely.

Architecture:
{architecture_json}

Explanation:
"""

        input_ids = tokenizer(prompt, return_tensors="pt").to(model_device)
        input_len = input_ids.input_ids.shape[1]

        start = time.perf_counter()
        with torch.inference_mode():
            outputs = model.generate(
                **input_ids,
                max_new_tokens=150,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                do_sample=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )
        elapsed = time.perf_counter() - start

        generated_ids = outputs[:, input_len:]
        generated = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()

        for prefix in ("Explanation:", "Answer:"):
            if generated.startswith(prefix):
                generated = generated[len(prefix):].strip()

        generated = clean_structured_output(generated, max_words=150)

        mem_mb = torch.cuda.memory_allocated() / (1024 ** 2)

        print("----------------------------------------")
        print(f"Sample {idx}")
        print("Architecture")
        print(json.dumps(architecture, indent=2, ensure_ascii=True))
        print("Generated Explanation")
        print(generated)
        print("Ground Truth")
        print(ground_truth)
        print(f"Generation Time (s): {elapsed:.3f}")
        print(f"GPU Memory Allocated (MB): {mem_mb:.2f}")

    print("----------------------------------------")
    print("Inference completed")


if __name__ == "__main__":
    main()
