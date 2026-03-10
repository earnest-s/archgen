"""
Architecture diagram explainer using Qwen2.5-3B-Instruct.

Combines a vision embedding (from ConvNeXt-Tiny) with a structured text prompt
that describes the Architecture JSON, then asks the LLM to produce a
plain-English explanation of the system design.

Public API::

    from backend.core.vlm.explainer import generate_explanation

    explanation = generate_explanation(architecture, vision_features)
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from backend.api.schemas.architecture import Architecture
from backend.core.vlm.qwen_loader import load_qwen

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a software architecture expert. "
    "Given a structured description of a system architecture, produce a clear, "
    "concise explanation suitable for a technical audience. "
    "Describe the overall pattern, component roles, and data-flow direction."
)

_USER_TEMPLATE = """\
Below is a JSON description of a software architecture diagram.

Architecture:
{arch_json}

Write a 2-4 sentence plain-English explanation of this architecture.
Focus on: overall pattern, component roles, and how data flows through the system.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _architecture_to_text(arch: Architecture) -> str:
    """Produce a compact, human-readable text summary of *arch*."""
    node_lines = "\n".join(
        f"  - {n.label} ({n.type.value})" for n in arch.nodes
    )
    edge_lines = "\n".join(
        f"  - {e.from_node} → {e.to_node}"
        + (f" [{e.protocol}]" if e.protocol else "")
        for e in arch.edges
    )
    parts = [f"Nodes:\n{node_lines}"]
    if edge_lines:
        parts.append(f"Connections:\n{edge_lines}")
    return "\n".join(parts)


def _build_messages(arch: Architecture) -> list[dict[str, str]]:
    """Build the chat message list for the Qwen instruct model."""
    arch_text = _architecture_to_text(arch)
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": _USER_TEMPLATE.format(arch_json=arch_text)},
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_explanation(
    architecture: Architecture,
    vision_features: Optional[torch.Tensor] = None,
    *,
    max_new_tokens: int = 256,
    temperature: float = 0.3,
    top_p: float = 0.9,
) -> str:
    """Generate a plain-English explanation of *architecture*.

    The function queries Qwen2.5-3B-Instruct with a structured prompt derived
    from the Architecture JSON.  *vision_features* is reserved for future
    multimodal fusion (currently used as a logging signal only).

    Args:
        architecture:    Validated :class:`Architecture` instance.
        vision_features: Optional 768-dim ConvNeXt embedding (not yet fused
                         into the LLM prompt; reserved for the VLM stage).
        max_new_tokens:  Maximum tokens to generate.  Default: 256.
        temperature:     Sampling temperature.  Lower = more deterministic.
        top_p:           Nucleus sampling probability mass.

    Returns:
        Plain-English explanation string.

    Raises:
        ImportError: If ``transformers`` / ``bitsandbytes`` are not installed.
        RuntimeError: If the model fails to generate a response.
    """
    model, tokenizer = load_qwen()

    messages = _build_messages(architecture)

    if vision_features is not None:
        logger.debug(
            "Vision features provided (shape=%s, norm=%.4f) — "
            "multimodal fusion reserved for next training phase.",
            tuple(vision_features.shape),
            vision_features.norm().item(),
        )

    # Apply the chat template (Qwen uses a specific format).
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt")

    # Move inputs to the same device as the model's first parameter.
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    logger.info("Generating explanation for architecture with %d nodes…", len(architecture.nodes))

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens (skip the prompt).
    new_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
    explanation: str = tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    logger.info("Explanation generated (%d tokens).", len(new_ids))
    return explanation


def generate_explanation_rule_based(architecture: Architecture) -> str:
    """Lightweight, dependency-free fallback explanation (no LLM required).

    Produces a template-filled sentence that describes the architecture pattern.
    Used when Qwen is not installed or during unit-testing.

    Args:
        architecture: Validated :class:`Architecture` instance.

    Returns:
        A single-paragraph plain-English description.
    """
    node_descriptions = ", ".join(
        f"a {n.type.value.lower()} layer ({n.label})" for n in architecture.nodes
    )
    n_components = len(architecture.nodes)
    n_edges = len(architecture.edges)

    has_frontend  = any(n.type.value == "Frontend"  for n in architecture.nodes)
    has_backend   = any(n.type.value == "Backend"   for n in architecture.nodes)
    has_database  = any(n.type.value == "Database"  for n in architecture.nodes)
    has_cache     = any(n.type.value == "Cache"     for n in architecture.nodes)
    has_queue     = any(n.type.value == "Queue"     for n in architecture.nodes)

    parts: list[str] = []

    if has_frontend and has_backend and has_database:
        parts.append(
            "This architecture follows a classic 3-tier pattern, "
            "separating presentation, application logic, and data persistence."
        )
    elif has_frontend and has_backend:
        parts.append(
            "This architecture uses a 2-tier client-server model "
            "with a dedicated frontend and backend layer."
        )
    else:
        parts.append(
            f"This architecture consists of {n_components} component"
            f"{'s' if n_components != 1 else ''}: {node_descriptions}."
        )

    if has_cache:
        parts.append("A caching layer reduces database load and improves response latency.")
    if has_queue:
        parts.append("A message queue decouples producers from consumers, enabling async processing.")
    if n_edges > 0:
        parts.append(
            f"Data flows through {n_edges} directed connection"
            f"{'s' if n_edges != 1 else ''} between components."
        )

    return " ".join(parts)
