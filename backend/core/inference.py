#!/usr/bin/env python3
import json
import os
import re

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

_MODEL = None
_TOKENIZER = None
_MODEL_DEVICE = None


def _limit_sentences(text: str, max_sentences: int = 2) -> str:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    parts = [p for p in parts if p]
    if not parts:
        return ""
    return " ".join(parts[:max_sentences]).strip()


def _extract_section(text: str, section_name: str, next_sections: tuple[str, ...]) -> str:
    boundary = "|".join(re.escape(s) for s in next_sections)
    pattern = rf"{re.escape(section_name)}\s*(.*?)(?={boundary}|$)"
    matches = re.findall(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    return matches[0].strip() if matches else ""


def _sanitize_section(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"\b(Components:|Data flow:|Architecture type:)\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace("\n", " ")

    for marker in ("Human", "JSON", "example", "```", "Assistant:", "User:", "###"):
        idx = cleaned.find(marker)
        if idx != -1:
            cleaned = cleaned[:idx]

    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -:\n\t")
    cleaned = _limit_sentences(cleaned, max_sentences=2)
    return cleaned


def _clean_structured_output(text: str, architecture: dict, max_words: int = 150) -> str:
    raw = text.strip()

    components = _extract_section(raw, "Components:", ("Data flow:", "Architecture type:"))
    data_flow = _extract_section(raw, "Data flow:", ("Architecture type:",))
    arch_type = _extract_section(raw, "Architecture type:", tuple())

    components_clean = _sanitize_section(components)
    dataflow_clean = _sanitize_section(data_flow)
    archtype_clean = _sanitize_section(arch_type)

    if not components_clean:
        nodes = architecture.get("nodes", []) if isinstance(architecture, dict) else []
        labels = [n.get("label", str(n)) if isinstance(n, dict) else str(n) for n in nodes]
        labels = [lbl for lbl in labels if lbl]
        components_clean = (
            f"Main components are {', '.join(labels)}."
            if labels
            else "Main components are frontend, backend, and database."
        )

    if not dataflow_clean:
        dataflow_clean = "Frontend communicates with backend, which interacts with database."

    if not archtype_clean:
        archtype_clean = "Layered architecture."

    components_clean = _limit_sentences(components_clean, max_sentences=2)
    dataflow_clean = _limit_sentences(dataflow_clean, max_sentences=2)
    archtype_clean = _limit_sentences(archtype_clean, max_sentences=2)

    final_output = (
        f"Components: {components_clean}\n\n"
        f"Data flow: {dataflow_clean}\n\n"
        f"Architecture type: {archtype_clean}"
    )

    words = final_output.split()
    if len(words) > max_words:
        final_output = " ".join(words[:max_words]).strip()

    return final_output


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


def preload_model() -> None:
    _load_model_once()


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

    return _clean_structured_output(generated, architecture=architecture, max_words=150)
