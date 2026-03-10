"""
Layout utilities for the ArchitectAI diagram generator.

Provides deterministic ordering and direction configuration that is consumed by
:mod:`backend.core.diagram.generator` when constructing a ``diagrams.Diagram``.
"""

from __future__ import annotations

from typing import List, Tuple

from backend.api.schemas.architecture import Node, NodeType

# Canonical layer order matches the prompt-parser's LAYER_ORDER so diagrams
# are drawn left-to-right in the same sequence data flows through the system.
LAYER_ORDER: List[NodeType] = [
    NodeType.Frontend,
    NodeType.Backend,
    NodeType.Service,
    NodeType.Database,
    NodeType.Cache,
    NodeType.Queue,
    NodeType.External,
]

# Graphviz rankdir used for all generated diagrams.
DIAGRAM_DIRECTION: str = "LR"  # left → right


def layer_rank(node: Node) -> int:
    """Return the integer rank for *node* based on canonical layer ordering.

    Lower ranks appear further to the left in an LR diagram.  Nodes whose
    type is not listed in :data:`LAYER_ORDER` receive the highest rank so they
    are placed at the right edge.

    Args:
        node: A validated :class:`~backend.api.schemas.architecture.Node`.

    Returns:
        Non-negative integer rank.
    """
    try:
        return LAYER_ORDER.index(node.type)
    except ValueError:
        return len(LAYER_ORDER)


def sort_nodes(nodes: List[Node]) -> List[Node]:
    """Return *nodes* sorted by layer rank then by label (for determinism).

    Args:
        nodes: Unsorted list of :class:`~backend.api.schemas.architecture.Node`.

    Returns:
        New list sorted by ``(layer_rank, label)``.
    """
    return sorted(nodes, key=lambda n: (layer_rank(n), n.label.lower()))


def group_nodes_by_layer(nodes: List[Node]) -> List[Tuple[str, List[Node]]]:
    """Group *nodes* into ordered ``(layer_name, [nodes])`` tuples.

    Nodes that share the same :attr:`~backend.api.schemas.architecture.Node.layer`
    value are grouped together.  Groups are ordered by the first node's layer
    rank within the group so the sequence matches the canonical pipeline.

    Args:
        nodes: List of :class:`~backend.api.schemas.architecture.Node` objects.

    Returns:
        List of ``(layer_name, node_list)`` tuples, sorted by layer rank.

    Example::

        groups = group_nodes_by_layer(arch.nodes)
        # → [("Presentation", [web_node]), ("Application", [api_node]), ...]
    """
    seen: dict[str, List[Node]] = {}
    order: list[str] = []

    for node in sort_nodes(nodes):
        layer_name = node.layer or node.type.value
        if layer_name not in seen:
            seen[layer_name] = []
            order.append(layer_name)
        seen[layer_name].append(node)

    return [(name, seen[name]) for name in order]
