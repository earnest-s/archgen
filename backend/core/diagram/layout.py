"""
Layout utilities for the ArchitectAI diagram generator.

Provides deterministic ordering, random direction selection, and cluster
grouping configuration consumed by :mod:`backend.core.diagram.generator`
when constructing a ``diagrams.Diagram``.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

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

# Available Graphviz rankdir values.
DIAGRAM_DIRECTIONS: List[str] = ["LR", "TB"]

# Default direction (kept for backwards compatibility).
DIAGRAM_DIRECTION: str = "LR"


def get_random_direction(seed: Optional[int] = None) -> str:
    """Return a randomly chosen Graphviz rankdir string.

    Alternates between ``"LR"`` (left-to-right) and ``"TB"`` (top-to-bottom)
    to create visual diversity across generated training diagrams.

    Args:
        seed: Optional integer seed for reproducibility.  When *None*, the
              global random state is used (non-deterministic).

    Returns:
        ``"LR"`` or ``"TB"``.
    """
    rng = random.Random(seed) if seed is not None else random
    return rng.choice(DIAGRAM_DIRECTIONS)


# ---------------------------------------------------------------------------
# Cluster / group mapping
# ---------------------------------------------------------------------------

#: Maps each NodeType to a named cluster used for visual grouping in diagrams.
CLUSTER_MAP: Dict[NodeType, str] = {
    NodeType.Frontend: "Frontend",
    NodeType.Backend:  "Backend Services",
    NodeType.Service:  "Backend Services",
    NodeType.Database: "Data Stores",
    NodeType.Cache:    "Data Stores",
    NodeType.Queue:    "Messaging",
    NodeType.External: "External",
}


def get_cluster_groups(
    nodes: List[Node],
    enabled: bool = True,
) -> Optional[Dict[str, List[Node]]]:
    """Group *nodes* into named clusters for ``diagrams.Cluster`` contexts.

    Clusters improve visual separation of:
    - **Frontend** — UI / client-side nodes.
    - **Backend Services** — API and service nodes.
    - **Data Stores** — databases and caches.
    - **Messaging** — message queues.
    - **External** — third-party / external system nodes.

    Args:
        nodes:   List of architecture nodes to group.
        enabled: When ``False`` returns ``None`` (no clustering applied).

    Returns:
        Ordered ``{cluster_name: [Node, …]}`` dict, or ``None`` when disabled
        or when every node would end up in the same single cluster.
    """
    if not enabled:
        return None

    groups: Dict[str, List[Node]] = {}
    for node in nodes:
        cluster_name = CLUSTER_MAP.get(node.type, "Other")
        groups.setdefault(cluster_name, []).append(node)

    # Skip clustering when there is only one group — it adds no visual value.
    if len(groups) <= 1:
        return None

    return groups


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
