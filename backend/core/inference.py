#!/usr/bin/env python3
import json
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

_MODEL = None
_TOKENIZER = None
_MODEL_DEVICE = None
_ALLOWED_NODE_TYPES = {"ui", "service", "database", "cache", "queue", "container"}
_NODE_TYPE_ALIASES = {
    "data": "database",
    "db": "database",
    "worker": "service",
}


def _extract_json_object(raw_text: str) -> dict:
    decoder = json.JSONDecoder()
    for start in (idx for idx, ch in enumerate(raw_text) if ch == "{"):
        try:
            parsed, _ = decoder.raw_decode(raw_text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("Model output did not contain a valid JSON object")


def _validate_architecture(payload: dict) -> dict:
    nodes = payload.get("nodes")
    edges = payload.get("edges")
    if not isinstance(nodes, list) or not isinstance(edges, list):
        raise ValueError("Architecture JSON must contain 'nodes' and 'edges' arrays")
    if len(nodes) == 0 or len(edges) == 0:
        raise ValueError("Architecture JSON must include at least one node and one edge")

    normalized_nodes: list[dict[str, str]] = []
    node_ids: set[str] = set()
    for node in nodes:
        if not isinstance(node, dict):
            raise ValueError("Each node must be an object")
        node_id = node.get("id")
        node_type = node.get("type")
        if not isinstance(node_id, str) or not node_id.strip():
            raise ValueError("Each node must include a non-empty string 'id'")
        if not isinstance(node_type, str):
            raise ValueError("Each node must include a valid 'type'")
        normalized_type = _NODE_TYPE_ALIASES.get(node_type.strip().lower(), node_type.strip().lower())
        if normalized_type not in _ALLOWED_NODE_TYPES:
            raise ValueError("Each node must include a valid 'type'")
        normalized_id = node_id.strip()
        if normalized_id in node_ids:
            raise ValueError(f"Duplicate node id '{normalized_id}'")
        node_ids.add(normalized_id)
        normalized_nodes.append({"id": normalized_id, "type": normalized_type})

    normalized_edges: list[dict[str, str]] = []
    for edge in edges:
        if not isinstance(edge, dict):
            raise ValueError("Each edge must be an object")
        source = edge.get("source")
        target = edge.get("target")
        label = edge.get("label")
        if not isinstance(source, str) or not isinstance(target, str):
            raise ValueError("Each edge must include string 'source' and 'target'")
        if source not in node_ids or target not in node_ids:
            raise ValueError("Edge source/target must reference known node ids")
        normalized_edge = {"source": source, "target": target}
        if isinstance(label, str) and label.strip():
            normalized_edge["label"] = label.strip()
        normalized_edges.append(normalized_edge)

    return {"nodes": normalized_nodes, "edges": normalized_edges}


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
    print("CUDA AVAILABLE:", torch.cuda.is_available())
    print("MODEL DEVICE:", _MODEL_DEVICE)
    print("GPU NAME:", torch.cuda.get_device_name(0))
    if "cuda" not in str(_MODEL_DEVICE):
        raise RuntimeError("Model is NOT using GPU")


def preload_model() -> None:
    _load_model_once()


def generate_architecture(text: str) -> tuple[dict, str]:
    _load_model_once()
    clean_text = text.strip()
    if not clean_text:
        raise ValueError("Input text is required")
    print("INPUT:", clean_text)
    print("RUNNING REAL MODEL")

    prompt = f"""
You are a system architect.

Convert this description into a JSON graph.

STRICT FORMAT:
{{
  "nodes": [{{"id": "...", "type": "ui|service|database|cache|queue|container"}}],
  "edges": [{{"source": "...", "target": "...", "label": "..."}}]
}}

Description:
{clean_text}

ONLY return JSON.
"""

    inputs = _TOKENIZER(prompt, return_tensors="pt").to(_MODEL_DEVICE)

    with torch.inference_mode():
        outputs = _MODEL.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.3,
            top_p=0.9,
            do_sample=True,
            eos_token_id=_TOKENIZER.eos_token_id,
            pad_token_id=_TOKENIZER.eos_token_id,
        )

    result = _TOKENIZER.decode(outputs[0], skip_special_tokens=True).strip()
    if result.startswith(prompt):
        result = result[len(prompt):].strip()

    print("RAW MODEL OUTPUT:", result[:500])

    architecture = _validate_architecture(_extract_json_object(result))
    print("OUTPUT LENGTH:", len(result))
    print("GPU MEMORY:", torch.cuda.memory_allocated() / 1024**2, "MB")
    return architecture, result
