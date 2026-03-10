"""
Node mapping between ArchitectAI NodeType and the ``diagrams`` library classes.

Each :class:`~backend.api.schemas.architecture.NodeType` is mapped to the most
semantically appropriate ``diagrams`` node class.  All mappings are lazily
resolved at call time so the module can be imported even when optional
diagram-provider sub-packages are not installed (helpful for unit tests).
"""

from __future__ import annotations

from typing import Type

from diagrams import Node as DiagramNode
from diagrams.generic.compute import Rack
from diagrams.generic.network import Firewall
from diagrams.onprem.compute import Server
from diagrams.onprem.container import Docker
from diagrams.onprem.database import PostgreSQL
from diagrams.onprem.inmemory import Redis
from diagrams.onprem.queue import Kafka
from diagrams.saas.cdn import Cloudflare

from backend.api.schemas.architecture import NodeType

# ---------------------------------------------------------------------------
# Canonical mapping: NodeType → diagrams node class
# ---------------------------------------------------------------------------

NODE_CLASS_MAP: dict[NodeType, Type[DiagramNode]] = {
    NodeType.Frontend:  Rack,        # generic "client / browser tier"
    NodeType.Backend:   Server,      # application server
    NodeType.Service:   Docker,      # containerised microservice
    NodeType.Database:  PostgreSQL,  # relational / persistent store
    NodeType.Cache:     Redis,       # in-memory cache
    NodeType.Queue:     Kafka,       # message broker / event stream
    NodeType.External:  Cloudflare,  # external / third-party SaaS
}

# Human-readable icon label suffix used when the node label is generated.
NODE_TYPE_SUFFIXES: dict[NodeType, str] = {
    NodeType.Frontend:  "(UI)",
    NodeType.Backend:   "(API)",
    NodeType.Service:   "(Service)",
    NodeType.Database:  "(DB)",
    NodeType.Cache:     "(Cache)",
    NodeType.Queue:     "(Queue)",
    NodeType.External:  "(External)",
}


def get_diagram_class(node_type: NodeType) -> Type[DiagramNode]:
    """Return the ``diagrams`` node class for a given *node_type*.

    Falls back to :class:`~diagrams.generic.compute.Rack` if the type is not
    in the canonical map (should never happen with a validated schema, but
    guards against future enum additions).

    Args:
        node_type: A :class:`~backend.api.schemas.architecture.NodeType` value.

    Returns:
        A ``diagrams`` node class (not an instance).
    """
    return NODE_CLASS_MAP.get(node_type, Rack)


def get_display_label(label: str, node_type: NodeType) -> str:
    """Format a human-readable display label for a diagram node.

    Appends a type suffix so the rendered icon is self-explanatory even when
    the label alone is ambiguous (e.g. ``"Auth"`` → ``"Auth (Service)"``).

    Args:
        label:     The :attr:`~backend.api.schemas.architecture.Node.label`
                   value from the Architecture model.
        node_type: The node's :class:`~backend.api.schemas.architecture.NodeType`.

    Returns:
        Combined display string, e.g. ``"PostgreSQL (DB)"``.
    """
    suffix = NODE_TYPE_SUFFIXES.get(node_type, "")
    return f"{label}\n{suffix}" if suffix else label
