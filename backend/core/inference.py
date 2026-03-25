#!/usr/bin/env python3
import json
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

_MODEL = None
_TOKENIZER = None
_MODEL_DEVICE = None


def _clean_structured_output(text: str, max_words: int = 150) -> str:
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

        arch_content = arch_block[len("Architecture type:") :].strip()
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


def _load_model_once() -> None:
    global _MODEL, _TOKENIZER, _MODEL_DEVICE

    if _MODEL is not None and _TOKENIZER is not None and _MODEL_DEVICE is not None:
        return

    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required for inference")

    os.environ.setdefault("HF_HOME", "./.cache/huggingface")

    model_id = "Qwen/Qwen2.5-1.5B-Instruct"

    print("Loading tokenizer...")
    _TOKENIZER = AutoTokenizer.from_pretrained(model_id)
    if _TOKENIZER.pad_token is None:
        _TOKENIZER.pad_token = _TOKENIZER.eos_token

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
    _MODEL = PeftModel.from_pretrained(base_model, "checkpoints/qwen_lora")
    _MODEL.eval()

    adapter_names = list(_MODEL.peft_config.keys())
    print(f"LoRA adapters loaded: {adapter_names}")

    _MODEL_DEVICE = next(_MODEL.parameters()).device


def generate_explanation(architecture: dict) -> str:
    _load_model_once()

    architecture_json = json.dumps(architecture, ensure_ascii=True, indent=2)
    prompt = f"""
Explain the following software architecture clearly and concisely.

Architecture:
{architecture_json}

Explanation:
"""

    input_ids = _TOKENIZER(prompt, return_tensors="pt").to(_MODEL_DEVICE)
    input_len = input_ids.input_ids.shape[1]

    with torch.inference_mode():
        outputs = _MODEL.generate(
            **input_ids,
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            do_sample=True,
            eos_token_id=_TOKENIZER.eos_token_id,
            pad_token_id=_TOKENIZER.eos_token_id,
        )

    generated_ids = outputs[:, input_len:]
    generated = _TOKENIZER.decode(generated_ids[0], skip_special_tokens=True).strip()

    for prefix in ("Explanation:", "Answer:"):
        if generated.startswith(prefix):
            generated = generated[len(prefix):].strip()

    return _clean_structured_output(generated, max_words=150)
