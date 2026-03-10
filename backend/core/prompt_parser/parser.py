"""
Prompt parser for the ArchitectAI pipeline.

Converts a natural-language architecture description into a validated
:class:`~backend.api.schemas.architecture.Architecture` object using
deterministic, rule-based extraction — no ML models are involved.

Pipeline
--------
1. **Normalise** the raw prompt (lowercase, strip punctuation).
2. **Load vocabulary** from ``vocabulary.json`` (adjacent to this package).
3. **Detect node types** by matching tokens against vocabulary keywords.
4. **Construct Node objects** with slugified, unique ids and inferred labels.
5. **Auto-generate edges** using canonical layer ordering.
6. **Return** a fully validated :class:`Architecture` instance.

Usage
-----
::

    from backend.core.prompt_parser.parser import parse_prompt

    arch = parse_prompt("React frontend with FastAPI backend and PostgreSQL database")
    print(arch.model_dump())
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

from backend.api.schemas.architecture import (
    Architecture,
    Edge,
    Metadata,
    Node,
    NodeType,
)
from backend.core.prompt_parser.rules import (
    build_default_edges,
    detect_node_types,
    normalize_prompt,
    slugify,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Vocabulary loading
# ---------------------------------------------------------------------------

_VOCAB_PATH: Path = Path(__file__).parent / "vocabulary.json"


def _load_vocabulary(path: Path = _VOCAB_PATH) -> Dict[str, List[str]]:
    """Load and return the keyword vocabulary from *path*.

    Args:
        path: Absolute path to a JSON file whose top-level keys are
              :class:`~backend.api.schemas.architecture.NodeType` names and
              whose values are lists of trigger keywords.

    Returns:
        Raw vocabulary mapping as a plain dict.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the file cannot be parsed as JSON.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Vocabulary file not found: {path}. "
            "Ensure vocabulary.json is present in the prompt_parser package directory."
        )
    try:
        with path.open(encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse vocabulary JSON at {path}: {exc}") from exc


# ---------------------------------------------------------------------------
# Node construction helpers
# ---------------------------------------------------------------------------

# Human-readable default label per NodeType, used when the matched keyword is
# too technical or abbreviated to serve as a readable diagram label.
_TYPE_LABELS: Dict[NodeType, str] = {
    NodeType.Frontend:  "Frontend",
    NodeType.Backend:   "Backend",
    NodeType.Service:   "Service",
    NodeType.Database:  "Database",
    NodeType.Cache:     "Cache",
    NodeType.Queue:     "Queue",
    NodeType.External:  "External",
}

# Layer assignment mirrors the canonical pipeline order in rules.LAYER_ORDER.
_TYPE_LAYERS: Dict[NodeType, str] = {
    NodeType.Frontend:  "Presentation",
    NodeType.Backend:   "Application",
    NodeType.Service:   "Application",
    NodeType.Database:  "Data",
    NodeType.Cache:     "Data",
    NodeType.Queue:     "Infrastructure",
    NodeType.External:  "External",
}


def _make_label(keyword: str, node_type: NodeType) -> str:
    """Derive a human-readable display label from a matched *keyword*.

    Short or cryptic keywords (≤ 2 characters) fall back to the type's default
    label.  All other keywords are title-cased and have underscores/hyphens
    replaced with spaces.

    Args:
        keyword:   The raw vocabulary keyword that triggered the match
                   (e.g. ``"postgresql"``, ``"fastapi"``).
        node_type: The :class:`NodeType` inferred for this keyword.

    Returns:
        A clean, title-cased label string (e.g. ``"Postgresql"``,
        ``"Fastapi"``).
    """
    if len(keyword) <= 2:
        return _TYPE_LABELS[node_type]
    return re.sub(r"[-_]+", " ", keyword).title()


def _ensure_unique_id(candidate: str, existing: set[str]) -> str:
    """Return *candidate* if unique, otherwise append a numeric suffix.

    Args:
        candidate: Preferred node id slug.
        existing:  Set of ids already allocated in the current parse run.
                   **Mutated in-place** with the chosen id.

    Returns:
        A unique slug derived from *candidate*.
    """
    if candidate not in existing:
        existing.add(candidate)
        return candidate
    counter = 2
    while True:
        suffixed = f"{candidate}_{counter}"
        if suffixed not in existing:
            existing.add(suffixed)
            return suffixed
        counter += 1


# ---------------------------------------------------------------------------
# Fallback node
# ---------------------------------------------------------------------------

def _fallback_node() -> Node:
    """Return a generic Backend node used when the prompt yields no matches.

    This guarantees that :func:`parse_prompt` always returns a valid
    :class:`Architecture` (which requires at least one node).
    """
    return Node(
        id="backend",
        type=NodeType.Backend,
        label="Backend",
        layer=_TYPE_LAYERS[NodeType.Backend],
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_prompt(
    prompt: str,
    *,
    vocab_path: Optional[Path] = None,
    metadata: Optional[Metadata] = None,
) -> Architecture:
    """Parse a natural-language architecture *prompt* into an :class:`Architecture`.

    The function is fully deterministic — identical inputs produce identical
    outputs.  No external services, LLMs, or random state are consulted.

    Args:
        prompt:     Raw user-supplied description of the desired architecture
                    (e.g. ``"React frontend, FastAPI backend, PostgreSQL DB"``).
        vocab_path: Optional override for the vocabulary JSON path.  Defaults
                    to the ``vocabulary.json`` bundled with this package.
        metadata:   Optional :class:`Metadata` to attach to the result.
                    Defaults to ``Metadata(version=1)``.

    Returns:
        A fully validated :class:`Architecture` with detected nodes and
        auto-generated pipeline edges.

    Raises:
        FileNotFoundError: If the vocabulary file cannot be located.
        ValueError:        If *prompt* is empty or whitespace-only.

    Example::

        arch = parse_prompt(
            "Create a 3-tier app with React frontend, FastAPI backend "
            "and PostgreSQL database"
        )
        # arch.nodes → [Node(id='react', type=Frontend), Node(id='fastapi', …), …]
        # arch.edges → [frontend→backend, backend→database]
    """
    if not prompt or not prompt.strip():
        raise ValueError("prompt must be a non-empty string.")

    # Step 1 – normalise
    normalised = normalize_prompt(prompt)
    logger.debug("Normalised prompt: %r", normalised)

    # Step 2 – load vocabulary
    vocab = _load_vocabulary(vocab_path or _VOCAB_PATH)

    # Step 3 – detect (keyword, NodeType) pairs, one per NodeType
    matches = detect_node_types(normalised, vocab)
    logger.debug("Detected matches: %s", matches)

    # Step 4 – construct Node objects
    allocated_ids: set[str] = set()
    nodes: List[Node] = []

    for keyword, node_type in matches:
        raw_id = slugify(keyword)
        node_id = _ensure_unique_id(raw_id, allocated_ids)
        label = _make_label(keyword, node_type)
        nodes.append(
            Node(
                id=node_id,
                type=node_type,
                label=label,
                layer=_TYPE_LAYERS[node_type],
            )
        )

    if not nodes:
        logger.warning(
            "No known components detected in prompt %r — using fallback node.",
            prompt,
        )
        nodes = [_fallback_node()]

    # Step 5 – auto-generate edges
    edges: List[Edge] = build_default_edges(nodes)
    logger.debug("Generated edges: %s", [(e.from_node, e.to_node) for e in edges])

    # Step 6 – assemble and validate the Architecture
    return Architecture(
        nodes=nodes,
        edges=edges,
        metadata=metadata or Metadata(version=1),
    )


# ---------------------------------------------------------------------------
# CLI / usage example
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json as _json
    import sys

    _prompt = (
        " ".join(sys.argv[1:])
        or "React frontend with FastAPI backend and PostgreSQL database"
    )
    print(f"Prompt : {_prompt!r}\n")
    _arch = parse_prompt(_prompt)
    print(_json.dumps(_arch.model_dump(by_alias=True), indent=2))
