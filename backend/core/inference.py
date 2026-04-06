#!/usr/bin/env python3
import json
import os
import time

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
_LABEL_ALIASES = {
    "request": "HTTP",
    "http": "HTTP",
    "db": "DB Query",
    "db query": "DB Query",
    "async": "Async",
    "cache": "Cache",
}


def _derive_type_from_id(node_id: str) -> str | None:
    normalized = node_id.strip().lower()
    if normalized in _ALLOWED_NODE_TYPES:
        return normalized
    if "cache" in normalized or "redis" in normalized:
        return "cache"
    if "queue" in normalized or "kafka" in normalized or "rabbit" in normalized:
        return "queue"
    if "db" in normalized or "postgres" in normalized or "database" in normalized or "mysql" in normalized:
        return "database"
    if "front" in normalized or "ui" in normalized or "client" in normalized:
        return "ui"
    if "container" in normalized or "docker" in normalized:
        return "container"
    if "api" in normalized or "service" in normalized or "backend" in normalized:
        return "service"
    return None


def _extract_json_object(raw_text: str) -> dict:
    decoder = json.JSONDecoder()
    first_object: dict | None = None
    for start in (idx for idx, ch in enumerate(raw_text) if ch == "{"):
        try:
            parsed, _ = decoder.raw_decode(raw_text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            if first_object is None:
                first_object = parsed
            if "nodes" in parsed and "edges" in parsed:
                return parsed
    if first_object is not None:
        return first_object
    raise ValueError("Model output did not contain a valid JSON object")


def _normalize_label(label: str | None) -> str | None:
    if not isinstance(label, str) or not label.strip():
        return None
    lowered = label.strip().lower()
    for key, normalized in _LABEL_ALIASES.items():
        if key in lowered:
            return normalized
    return label.strip()


def _infer_edge_label(source_type: str, target_type: str, current: str | None) -> str:
    normalized = _normalize_label(current)
    if normalized:
        return normalized
    if source_type == "ui" and target_type == "service":
        return "HTTP"
    if source_type == "service" and target_type == "database":
        return "DB Query"
    if source_type == "service" and target_type == "queue":
        return "Async"
    if source_type == "service" and target_type == "cache":
        return "Cache"
    return "HTTP"


def _validate_architecture(payload: dict) -> dict:
    nodes = payload.get("nodes")
    edges = payload.get("edges")
    if not isinstance(nodes, list) or not isinstance(edges, list):
        raise ValueError("Architecture JSON must contain 'nodes' and 'edges' arrays")
    if len(nodes) == 0 or len(edges) == 0:
        raise ValueError("Architecture JSON must include at least one node and one edge")

    normalized_nodes: list[dict[str, str]] = []
    node_types_by_id: dict[str, str] = {}
    for node in nodes:
        node_id = None
        node_type = None

        if isinstance(node, dict):
            node_id = node.get("id")
            node_type = node.get("type")
        elif isinstance(node, str):
            compact = node.strip().lower()
            if compact in _ALLOWED_NODE_TYPES:
                node_id = compact
                node_type = compact

        if not isinstance(node_id, str) or not node_id.strip():
            raise ValueError("Each node must include a non-empty string 'id'")
        normalized_id = node_id.strip()

        if isinstance(node_type, str):
            candidate_type = _NODE_TYPE_ALIASES.get(node_type.strip().lower(), node_type.strip().lower())
            if candidate_type == "component":
                inferred = _derive_type_from_id(normalized_id)
                if inferred is None:
                    raise ValueError("Each node must include a valid 'type'")
                candidate_type = inferred
        else:
            inferred = _derive_type_from_id(normalized_id)
            if inferred is None:
                raise ValueError("Each node must include a valid 'type'")
            candidate_type = inferred

        normalized_type = candidate_type
        if normalized_type not in _ALLOWED_NODE_TYPES:
            raise ValueError("Each node must include a valid 'type'")

        # Remove duplicate nodes by id (keep first occurrence).
        if normalized_id in node_types_by_id:
            continue
        node_types_by_id[normalized_id] = normalized_type
        normalized_nodes.append({"id": normalized_id, "type": normalized_type})

    normalized_nodes = normalized_nodes[:8]
    allowed_ids = {node["id"] for node in normalized_nodes}
    node_types_by_id = {node["id"]: node["type"] for node in normalized_nodes}

    normalized_edges: list[dict[str, str]] = []
    seen_edges: set[tuple[str, str]] = set()
    for edge in edges:
        source = None
        target = None
        label = None

        if isinstance(edge, dict):
            source = edge.get("source") if edge.get("source") is not None else edge.get("from")
            target = edge.get("target") if edge.get("target") is not None else edge.get("to")
            label = edge.get("label") if edge.get("label") is not None else edge.get("protocol")
        elif isinstance(edge, list) and len(edge) >= 2:
            source = edge[0]
            target = edge[1]
            if len(edge) >= 3:
                label = edge[2]

        if not isinstance(source, str) or not isinstance(target, str):
            raise ValueError("Each edge must include string 'source' and 'target'")

        source_id = source.strip()
        target_id = target.strip()

        # Remove self-loops.
        if source_id == target_id:
            continue

        # Drop edges that reference missing nodes.
        if source_id not in allowed_ids or target_id not in allowed_ids:
            continue

        # Remove duplicate edges by (source, target).
        edge_key = (source_id, target_id)
        if edge_key in seen_edges:
            continue
        seen_edges.add(edge_key)

        edge_label = _infer_edge_label(
            node_types_by_id[source_id],
            node_types_by_id[target_id],
            label if isinstance(label, str) else None,
        )
        normalized_edges.append({"source": source_id, "target": target_id, "label": edge_label})

        if len(normalized_edges) >= 10:
            break

    if len(normalized_edges) == 0:
        raise ValueError("Architecture JSON must include at least one valid edge")

    # Performance guardrails.
    if len(normalized_nodes) > 10 or len(normalized_edges) > 15:
        raise ValueError("Generated graph exceeds production limits")

    return {"nodes": normalized_nodes, "edges": normalized_edges}


def _tokenize_prompt(prompt: str) -> object:
    try:
        chat_prompt = _TOKENIZER.apply_chat_template(
            [
                {
                    "role": "system",
                    "content": "You are a strict JSON generator for software architecture graphs.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        return _TOKENIZER(chat_prompt, return_tensors="pt").to(_MODEL_DEVICE)
    except Exception:  # noqa: BLE001
        return _TOKENIZER(prompt, return_tensors="pt").to(_MODEL_DEVICE)


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


def run_startup_smoke_test() -> None:
    _load_model_once()
    prompt = "Return exactly this JSON and nothing else: {\"status\":\"ready\"}"
    inputs = _TOKENIZER(prompt, return_tensors="pt").to(_MODEL_DEVICE)
    with torch.inference_mode():
        outputs = _MODEL.generate(
            **inputs,
            max_new_tokens=40,
            do_sample=False,
            eos_token_id=_TOKENIZER.eos_token_id,
            pad_token_id=_TOKENIZER.eos_token_id,
        )
    input_len = inputs.input_ids.shape[1]
    generated_ids = outputs[:, input_len:]
    smoke = _TOKENIZER.decode(generated_ids[0], skip_special_tokens=True).strip()
    if not smoke:
        raise RuntimeError("Startup smoke inference returned empty output")


def _is_structurally_weak_graph(architecture: dict) -> bool:
    nodes = architecture.get("nodes", [])
    edges = architecture.get("edges", [])
    if len(nodes) == 0 or len(edges) == 0:
        return True

    degrees: dict[str, int] = {
        node["id"]: 0
        for node in nodes
        if isinstance(node, dict) and isinstance(node.get("id"), str)
    }
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        src = edge.get("source")
        dst = edge.get("target")
        if isinstance(src, str) and src in degrees:
            degrees[src] += 1
        if isinstance(dst, str) and dst in degrees:
            degrees[dst] += 1

    return any(degree == 0 for degree in degrees.values())


def generate_architecture(text: str, deterministic: bool = False) -> tuple[dict, str]:
    _load_model_once()
    clean_text = text.strip()
    if not clean_text:
        raise ValueError("Input text is required")
    print("INPUT:", clean_text)
    print("RUNNING REAL MODEL")

        prompt = f"""
You are a senior software architect.

Convert the following system description into a CLEAN architecture graph.

RULES:
- Use only meaningful components
- No duplicate edges
- No redundant connections
- Keep architecture minimal and logical

FORMAT:
{{
    "nodes": [
        {{"id": "frontend", "type": "ui"}},
        {{"id": "api", "type": "service"}}
    ],
    "edges": [
        {{"source": "frontend", "target": "api", "label": "HTTP"}}
    ]
}}

CONSTRAINTS:
- Each edge must be unique
- Do NOT repeat same connection
- Use proper labels: HTTP, DB Query, Async, Cache
- Max nodes: 8
- Max edges: 10

Description:
{clean_text}

ONLY return JSON. No explanation.
"""

        stricter_prompt = prompt + "\nGenerate a connected architecture graph. All nodes must be part of a valid flow."

    attempts = 1 if deterministic else 3
    last_error = ""
    last_result = ""
    started_at = time.perf_counter()

    def _run_attempts(prompt_text: str, attempt_count: int) -> tuple[dict, str] | None:
        nonlocal last_error, last_result
        for attempt in range(1, attempt_count + 1):
            inputs = _tokenize_prompt(prompt_text)
            generation_kwargs = {
                "max_new_tokens": 300,
                "do_sample": not deterministic,
                "eos_token_id": _TOKENIZER.eos_token_id,
                "pad_token_id": _TOKENIZER.eos_token_id,
            }
            if not deterministic:
                generation_kwargs.update({"temperature": 0.3, "top_p": 0.9})

            with torch.inference_mode():
                outputs = _MODEL.generate(**inputs, **generation_kwargs)

            input_len = inputs.input_ids.shape[1]
            generated_ids = outputs[:, input_len:]
            result = _TOKENIZER.decode(generated_ids[0], skip_special_tokens=True).strip()
            last_result = result
            print(f"RAW MODEL OUTPUT [attempt {attempt}/{attempt_count}]:", result[:500])

            try:
                architecture = _validate_architecture(_extract_json_object(result))
                if _is_structurally_weak_graph(architecture):
                    raise ValueError("Generated graph is structurally weak")

                elapsed_ms = int((time.perf_counter() - started_at) * 1000)
                print("INFERENCE TIME MS:", elapsed_ms)
                print("NODE COUNT:", len(architecture["nodes"]))
                print("EDGE COUNT:", len(architecture["edges"]))
                print("OUTPUT LENGTH:", len(result))
                print("GPU MEMORY:", torch.cuda.memory_allocated() / 1024**2, "MB")
                return architecture, result
            except ValueError as exc:
                last_error = str(exc)
                continue
        return None

    primary_result = _run_attempts(prompt, attempts)
    if primary_result is not None:
        return primary_result

    connected_result = _run_attempts(stricter_prompt, 1)
    if connected_result is not None:
        return connected_result

    raise ValueError(f"Architecture generation failed after {attempts + 1} attempts: {last_error}. Last output: {last_result[:300]}")
