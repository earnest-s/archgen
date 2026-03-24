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
from functools import lru_cache
from typing import Optional

import torch

from backend.api.schemas.architecture import Architecture
from backend.core.vlm.projector import VisionProjector
from backend.core.vlm.qwen_loader import load_qwen

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a senior software architect. "
    "Given a structured description of a system architecture, produce a concise, "
    "structured explanation that is useful to a technical audience. "
    "Always organise your response into exactly four sections with these headings:\n"
    "  Section 1: Components\n"
    "  Section 2: Data Flow\n"
    "  Section 3: Architecture Pattern\n"
    "  Section 4: Observations\n"
    "Keep each section to 1-3 sentences. Be specific and avoid filler phrases."
)

_USER_TEMPLATE = """\
Below is a description of a software architecture.  The detected pattern is: {pattern}.

Architecture:
{arch_json}

Respond with exactly four sections:

Section 1: Components
<List each component and its role in the system.>

Section 2: Data Flow
<Describe how data moves between components, including protocols where relevant.>

Section 3: Architecture Pattern
<Name and briefly describe the architectural pattern ({pattern}).>

Section 4: Observations
<Note key design decisions, trade-offs, or potential improvements.>
"""

# Used when ConvNeXt vision features are available.
_USER_TEMPLATE_WITH_VISION = """\
Below is a description of a software architecture along with visual analysis from a
ConvNeXt diagram encoder.  The detected pattern is: {pattern}.

Architecture:
{arch_json}

Diagram Visual Context (ConvNeXt analysis):
{vision_context}

Using both the structural description and the visual context, respond with exactly
four sections:

Section 1: Components
<List each component and its role in the system.>

Section 2: Data Flow
<Describe how data moves between components, including protocols where relevant.>

Section 3: Architecture Pattern
<Name and briefly describe the architectural pattern ({pattern}).>

Section 4: Observations
<Note key design decisions, trade-offs, or potential improvements.>
"""


# ---------------------------------------------------------------------------
# Pattern detection
# ---------------------------------------------------------------------------

# Pattern priority: checked in order; first match wins.
_PATTERNS = [
    "streaming pipeline",
    "event-driven",
    "microservices",
    "layered",
    "client-server",
]


def detect_pattern(arch: Architecture) -> str:
    """Classify the architecture into a named pattern based on graph structure.

    Heuristics (checked in priority order):

    * **Streaming pipeline** — contains a Queue node *and* ≥ 1 Service node.
    * **Event-driven** — contains a Queue node without a dedicated Service.
    * **Microservices** — ≥ 3 Service nodes.
    * **Layered** — nodes span ≥ 3 distinct layer types (Frontend/Backend/DB…).
    * **Client-server** — has at least one Frontend and one Backend node.
    * **Unknown** — none of the above heuristics match.

    Args:
        arch: A validated :class:`~backend.api.schemas.architecture.Architecture`.

    Returns:
        One of the pattern strings listed in :data:`_PATTERNS`, or
        ``"unknown"``.
    """
    types = {n.type.value for n in arch.nodes}
    n_services  = sum(1 for n in arch.nodes if n.type.value == "Service")
    n_layers    = len(types)

    has_queue    = "Queue"    in types
    has_service  = n_services > 0
    has_frontend = "Frontend" in types
    has_backend  = "Backend"  in types

    if has_queue and has_service:
        return "streaming pipeline"
    if has_queue:
        return "event-driven"
    if n_services >= 3:
        return "microservices"
    if n_layers >= 3:
        return "layered"
    if has_frontend and has_backend:
        return "client-server"
    return "unknown"


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
    """Build the chat message list for the Qwen instruct model (text-only)."""
    arch_text = _architecture_to_text(arch)
    pattern = detect_pattern(arch)
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": _USER_TEMPLATE.format(arch_json=arch_text, pattern=pattern)},
    ]


@lru_cache(maxsize=1)
def _get_projector() -> VisionProjector:
    """Return a cached VisionProjector (randomly initialised unless a
    checkpoint is loaded externally before the first call).
    """
    proj = VisionProjector()
    proj.eval()
    logger.info(
        "VisionProjector initialised "
        "(random weights — load a checkpoint for fine-tuned projection)."
    )
    return proj


def _vision_context_str(vision_features: torch.Tensor) -> str:
    """Project ConvNeXt features to LM space and return a compact text summary.

    Runs the :class:`VisionProjector` and computes activation statistics that
    can be embedded in the LLM prompt as a structured visual context block.

    Args:
        vision_features: Float tensor of shape ``(768,)``.

    Returns:
        Multi-line string describing the visual feature statistics.
    """
    projector = _get_projector()
    feat = vision_features.float().cpu().detach()
    if feat.ndim == 1:
        feat = feat.unsqueeze(0)  # (1, 768)

    with torch.inference_mode():
        projected = projector(feat).squeeze(0)  # (2048,)

    mean_val   = projected.mean().item()
    std_val    = projected.std().item()
    norm_val   = projected.norm().item()
    # Count dimensions whose absolute activation exceeds mean + 0.5 * std.
    threshold  = projected.abs().mean() + 0.5 * projected.abs().std()
    n_active   = int((projected.abs() > threshold).sum().item())
    complexity = "high" if n_active > 512 else "medium" if n_active > 128 else "low"

    return (
        f"  Activation statistics : mean={mean_val:.4f}, std={std_val:.4f}, "
        f"L2_norm={norm_val:.2f}\n"
        f"  Visual complexity     : {complexity} "
        f"({n_active} active dims out of {projected.shape[0]})"
    )


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

    # ── Detect architecture pattern ───────────────────────────────────────────
    pattern = detect_pattern(architecture)
    logger.info("Detected architecture pattern: %s", pattern)

    # ── Build prompt (with or without vision context) ─────────────────────────
    arch_text = _architecture_to_text(architecture)
    if vision_features is not None:
        try:
            vision_ctx = _vision_context_str(vision_features)
            user_content = _USER_TEMPLATE_WITH_VISION.format(
                arch_json=arch_text,
                vision_context=vision_ctx,
                pattern=pattern,
            )
            logger.info(
                "Vision features integrated into prompt "
                "(shape=%s, norm=%.4f).",
                tuple(vision_features.shape),
                vision_features.norm().item(),
            )
        except Exception as exc:
            logger.warning(
                "Vision projection failed (%s) — falling back to text-only prompt.",
                exc,
            )
            user_content = _USER_TEMPLATE.format(arch_json=arch_text, pattern=pattern)
    else:
        user_content = _USER_TEMPLATE.format(arch_json=arch_text, pattern=pattern)

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]

    # ── Tokenise & generate ───────────────────────────────────────────────────
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt")

    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    logger.info(
        "Generating explanation for architecture with %d nodes…",
        len(architecture.nodes),
    )

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
    explanation: str = tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    logger.info("Explanation generated (%d tokens).", len(new_ids))
    return explanation


def generate_explanation_rule_based(architecture: Architecture) -> str:
    """Lightweight, dependency-free fallback explanation (no LLM required).

    Produces a structured four-section explanation using only string templates.
    Used when Qwen is not installed or during unit-testing.

    Args:
        architecture: Validated :class:`Architecture` instance.

    Returns:
        A plain-English description organised into four sections.
    """
    pattern = detect_pattern(architecture)

    node_lines = "\n".join(
        f"  - {n.label} ({n.type.value})" for n in architecture.nodes
    )
    n_components = len(architecture.nodes)
    n_edges = len(architecture.edges)

    has_frontend  = any(n.type.value == "Frontend"  for n in architecture.nodes)
    has_backend   = any(n.type.value == "Backend"   for n in architecture.nodes)
    has_database  = any(n.type.value == "Database"  for n in architecture.nodes)
    has_cache     = any(n.type.value == "Cache"     for n in architecture.nodes)
    has_queue     = any(n.type.value == "Queue"     for n in architecture.nodes)

    # ── Section 1: Components ─────────────────────────────────────────────────
    section1 = f"Section 1: Components\n{node_lines}\n"

    # ── Section 2: Data Flow ──────────────────────────────────────────────────
    if n_edges > 0:
        edge_desc = (
            f"Data flows through {n_edges} directed connection"
            f"{'s' if n_edges != 1 else ''} between components."
        )
    else:
        edge_desc = "No explicit connections are defined between components."

    extras: list[str] = []
    if has_cache:
        extras.append("A caching layer reduces database load and improves response latency.")
    if has_queue:
        extras.append("A message queue decouples producers from consumers, enabling asynchronous processing.")

    section2 = "Section 2: Data Flow\n" + " ".join([edge_desc] + extras)

    # ── Section 3: Architecture Pattern ──────────────────────────────────────
    pattern_desc: dict[str, str] = {
        "streaming pipeline": (
            "This is a streaming pipeline: data is ingested through a queue, "
            "processed by one or more services, and written to a data store."
        ),
        "event-driven": (
            "This is an event-driven architecture: a message queue decouples "
            "event producers from consumers for asynchronous communication."
        ),
        "microservices": (
            "This is a microservices architecture: multiple independent services "
            "each own a specific domain and communicate over the network."
        ),
        "layered": (
            "This is a layered (n-tier) architecture: concerns are separated "
            "into distinct tiers such as presentation, business logic, and data."
        ),
        "client-server": (
            "This is a client-server architecture: a frontend client delegates "
            "all business logic and data management to a backend server."
        ),
    }
    section3 = (
        "Section 3: Architecture Pattern\n"
        + pattern_desc.get(
            pattern,
            f"The architecture pattern is '{pattern}': "
            f"{n_components} component{'s' if n_components != 1 else ''} "
            "collaborate to deliver the system's functionality.",
        )
    )

    # ── Section 4: Observations ───────────────────────────────────────────────
    observations: list[str] = []
    if has_frontend and not has_backend:
        observations.append("Consider adding a dedicated backend layer for business logic separation.")
    if has_database and not has_cache:
        observations.append("Adding a cache layer could reduce database read pressure.")
    if n_edges == 0:
        observations.append("No connections are defined; ensure components are wired together.")
    if not observations:
        observations.append(
            f"The architecture uses {n_components} component"
            f"{'s' if n_components != 1 else ''} and appears well-structured."
        )
    section4 = "Section 4: Observations\n" + " ".join(observations)

    return "\n\n".join([section1, section2, section3, section4])
