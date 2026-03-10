"""
Rule-based extraction helpers for the ArchitectAI prompt parser.

This module contains purely deterministic, stateless functions that operate on
normalised text and vocabulary mappings.  No ML models or external calls are
made here.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

from backend.api.schemas.architecture import Edge, Node, NodeType

# ---------------------------------------------------------------------------
# Layer ordering used when auto-generating edges.
# Nodes are wired left-to-right following this sequence.
# ---------------------------------------------------------------------------
LAYER_ORDER: List[NodeType] = [
    NodeType.Frontend,
    NodeType.Backend,
    NodeType.Service,
    NodeType.Database,
    NodeType.Cache,
    NodeType.Queue,
    NodeType.External,
]

# Pre-compiled pattern: strip everything that is not a word char or whitespace.
_PUNCT_RE = re.compile(r"[^\w\s-]")
# Collapse runs of whitespace (including resulting gaps after stripping).
_SPACE_RE = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def normalize_prompt(prompt: str) -> str:
    """Return a lowercase, punctuation-stripped version of *prompt*.

    Hyphens are preserved so that compound terms like ``"third-party"`` or
    ``"in-memory"`` remain matchable against vocabulary entries.

    Args:
        prompt: Raw natural-language input from the user.

    Returns:
        Normalised string suitable for tokenisation and keyword matching.

    Example::

        >>> normalize_prompt("FastAPI Backend, PostgreSQL DB!")
        'fastapi backend postgresql db'
    """
    lowered = prompt.lower()
    stripped = _PUNCT_RE.sub(" ", lowered)
    return _SPACE_RE.sub(" ", stripped).strip()


def detect_node_types(
    prompt: str,
    vocab: Dict[str, List[str]],
) -> List[Tuple[str, NodeType]]:
    """Scan *prompt* for architectural keywords and return typed component matches.

    The function tokenises the normalised prompt into individual words and
    sliding bigrams (two consecutive words joined by a hyphen in some vocab
    entries), then matches each token against every entry in *vocab*.

    Deduplication rule: only the **first** keyword match per :class:`NodeType`
    is kept, preserving the left-to-right order in which components appear in
    the prompt.  This means ``"react frontend"`` yields a single
    ``("react", NodeType.Frontend)`` tuple rather than two.

    Args:
        prompt: Already-normalised prompt text (see :func:`normalize_prompt`).
        vocab:  Mapping of ``NodeType`` name → list of trigger keywords.
                Typically loaded from ``vocabulary.json``.

    Returns:
        Ordered list of ``(matched_keyword, NodeType)`` tuples, deduplicated
        per node type.

    Example::

        >>> vocab = {"Frontend": ["react"], "Backend": ["fastapi"]}
        >>> detect_node_types("react frontend and fastapi backend", vocab)
        [('react', <NodeType.Frontend: 'Frontend'>),
         ('fastapi', <NodeType.Backend: 'Backend'>)]
    """
    # Build a flat reverse lookup: keyword → NodeType (validated).
    keyword_to_type: Dict[str, NodeType] = {}
    for type_name, keywords in vocab.items():
        try:
            node_type = NodeType(type_name)
        except ValueError:
            # Ignore vocab entries that don't correspond to a known NodeType.
            continue
        for kw in keywords:
            keyword_to_type[kw.lower()] = node_type

    tokens = prompt.split()

    # Also test bigrams to catch compound keywords like "payment service".
    candidates: List[str] = list(tokens)
    for i in range(len(tokens) - 1):
        candidates.append(f"{tokens[i]}-{tokens[i + 1]}")

    seen_types: set[NodeType] = set()
    results: List[Tuple[str, NodeType]] = []

    for token in candidates:
        if token in keyword_to_type:
            node_type = keyword_to_type[token]
            if node_type not in seen_types:
                seen_types.add(node_type)
                results.append((token, node_type))

    return results


def slugify(text: str) -> str:
    """Convert *text* to a safe, lowercase node id.

    Replaces whitespace and hyphens with underscores and strips any character
    that is not alphanumeric or an underscore.

    Args:
        text: Arbitrary string (e.g. a matched keyword or a label).

    Returns:
        A slug suitable for use as a :class:`~backend.api.schemas.architecture.Node` id.

    Example::

        >>> slugify("PostgreSQL DB")
        'postgresql_db'
        >>> slugify("third-party")
        'third_party'
    """
    lowered = text.lower().strip()
    replaced = re.sub(r"[\s\-]+", "_", lowered)
    return re.sub(r"[^\w]", "", replaced)


def build_default_edges(nodes: List[Node]) -> List[Edge]:
    """Auto-generate a linear pipeline of edges based on canonical layer order.

    Nodes are sorted according to :data:`LAYER_ORDER`.  Consecutive neighbours
    in the sorted list receive a directed edge from the earlier to the later
    layer.  Nodes that share the same layer position are connected in the order
    they appear in *nodes*.

    If zero or one node is provided, an empty list is returned.

    Args:
        nodes: List of :class:`~backend.api.schemas.architecture.Node` objects
               to wire together.

    Returns:
        Ordered list of :class:`~backend.api.schemas.architecture.Edge` objects
        representing the default data-flow pipeline.

    Example::

        Given nodes [Frontend, Backend, Database] the result is:
        [Frontend → Backend, Backend → Database]
    """
    if len(nodes) <= 1:
        return []

    def _layer_rank(node: Node) -> int:
        try:
            return LAYER_ORDER.index(node.type)
        except ValueError:
            # Unknown types sort to the end.
            return len(LAYER_ORDER)

    sorted_nodes = sorted(nodes, key=_layer_rank)

    edges: List[Edge] = []
    for src, dst in zip(sorted_nodes, sorted_nodes[1:]):
        edges.append(Edge(**{"from": src.id, "to": dst.id}))

    return edges
