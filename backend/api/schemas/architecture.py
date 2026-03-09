"""
Pydantic models for the ArchitectAI architecture schema.

These models define the core data contract shared across the entire pipeline:
  - FastAPI request/response serialization
  - React Flow diagram editing (via JSON export)
  - Diagram generation (diagrams + Graphviz)
  - Training dataset generation

Schema source-of-truth: shared/architecture.schema.json
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class NodeType(str, Enum):
    """Architectural role of a node within the system diagram.

    Values are kept as human-readable strings so they serialize cleanly in
    JSON responses and are usable directly as diagram labels.
    """

    Frontend = "Frontend"
    Backend = "Backend"
    Service = "Service"
    Database = "Database"
    Cache = "Cache"
    Queue = "Queue"
    External = "External"


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class Node(BaseModel):
    """A single component in the architecture diagram.

    Represents an architectural element such as a microservice, database,
    frontend application, or external third-party system.

    Attributes:
        id: Unique node identifier. Alphanumeric with hyphens/underscores.
            Referenced by :class:`Edge` ``from`` and ``to`` fields.
        type: Architectural category of the node (see :class:`NodeType`).
        label: Human-readable display name rendered inside the diagram node.
        layer: Optional logical grouping layer (e.g. ``"Presentation"``,
            ``"Application"``, ``"Data"``). Used by React Flow for grouping
            and by the diagram generator for vertical layout bands.
    """

    id: str = Field(
        ...,
        min_length=1,
        pattern=r"^[a-zA-Z0-9_\-]+$",
        description=(
            "Unique node identifier. Must contain only alphanumeric characters, "
            "hyphens, or underscores."
        ),
        examples=["api_gateway", "user-db", "auth_service"],
    )
    type: NodeType = Field(
        ...,
        description="Architectural category of the node.",
    )
    label: str = Field(
        ...,
        min_length=1,
        description="Human-readable display name for the node.",
        examples=["API Gateway", "User Database", "Auth Service"],
    )
    layer: Optional[str] = Field(
        default=None,
        min_length=1,
        description=(
            "Optional logical grouping layer. Used for vertical layout ordering "
            "and React Flow node grouping."
        ),
        examples=["Presentation", "Application", "Data", "Infrastructure"],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "id": "api_gateway",
                    "type": "Backend",
                    "label": "API Gateway",
                    "layer": "Application",
                }
            ]
        }
    }


class Edge(BaseModel):
    """A directed connection between two nodes in the architecture diagram.

    Represents a communication channel, data flow, or dependency between
    architectural components.

    Attributes:
        from_node: ``id`` of the source node. Must reference an existing
            :attr:`Node.id` in the parent :class:`Architecture`.
        to_node: ``id`` of the target node. Must reference an existing
            :attr:`Node.id` in the parent :class:`Architecture`.
        protocol: Optional communication protocol. Common values include
            ``"HTTP"``, ``"gRPC"``, ``"TCP"``, ``"WebSocket"``, ``"AMQP"``,
            ``"SQL"``.
    """

    from_node: str = Field(
        ...,
        alias="from",
        min_length=1,
        description="id of the source node.",
        examples=["api_gateway"],
    )
    to_node: str = Field(
        ...,
        alias="to",
        min_length=1,
        description="id of the target node.",
        examples=["auth_service"],
    )
    protocol: Optional[str] = Field(
        default=None,
        min_length=1,
        description="Optional communication protocol used on this connection.",
        examples=["HTTP", "gRPC", "TCP", "WebSocket", "AMQP", "SQL"],
    )

    model_config = {
        # Allow both 'from'/'to' (canonical JSON) and 'from_node'/'to_node'
        # (Python attribute names) to be used during construction.
        "populate_by_name": True,
        "json_schema_extra": {
            "examples": [
                {"from": "api_gateway", "to": "auth_service", "protocol": "gRPC"}
            ]
        },
    }


class Metadata(BaseModel):
    """Diagram-level metadata controlling rendering style and schema versioning.

    Attributes:
        version: Integer schema version. Starts at ``1`` and increments on
            breaking changes. Used to gate dataset and diagram compatibility
            checks across training runs.
        style: Optional visual style hint passed to the diagram renderer
            (e.g. ``"dark"``, ``"light"``, ``"blueprint"``).
    """

    version: int = Field(
        default=1,
        ge=1,
        description=(
            "Schema version number. Increment on breaking structural changes. "
            "Defaults to 1."
        ),
        examples=[1, 2],
    )
    style: Optional[str] = Field(
        default=None,
        min_length=1,
        description="Optional visual style hint for the diagram renderer.",
        examples=["light", "dark", "blueprint", "minimal"],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [{"version": 1, "style": "light"}]
        }
    }


# ---------------------------------------------------------------------------
# Root model
# ---------------------------------------------------------------------------


class Architecture(BaseModel):
    """Root model representing a complete software architecture diagram.

    Contains a validated collection of nodes and directed edges, together with
    diagram-level metadata.  Two cross-field invariants are enforced:

    1. **Unique node ids** – every :attr:`Node.id` in :attr:`nodes` must be
       distinct.
    2. **Referential integrity** – every ``from`` and ``to`` value in
       :attr:`edges` must correspond to an existing :attr:`Node.id`.

    Attributes:
        nodes: Ordered list of architectural components. Must contain at least
            one node.
        edges: Directed connections between nodes. May be empty (a diagram can
            consist of isolated nodes).
        metadata: Diagram-level metadata (version, style). Defaults to
            ``Metadata(version=1)``.

    Example::

        arch = Architecture(
            nodes=[
                Node(id="web", type=NodeType.Frontend, label="Web App", layer="Presentation"),
                Node(id="api", type=NodeType.Backend, label="REST API", layer="Application"),
                Node(id="db",  type=NodeType.Database, label="PostgreSQL", layer="Data"),
            ],
            edges=[
                Edge(**{"from": "web", "to": "api", "protocol": "HTTPS"}),
                Edge(**{"from": "api", "to": "db",  "protocol": "SQL"}),
            ],
            metadata=Metadata(version=1, style="light"),
        )
    """

    nodes: list[Node] = Field(
        ...,
        min_length=1,
        description=(
            "List of architectural components. Each node must have a unique id. "
            "At least one node is required."
        ),
    )
    edges: list[Edge] = Field(
        default_factory=list,
        description=(
            "Directed connections between nodes. Each edge must reference "
            "existing node ids via 'from' and 'to' fields."
        ),
    )
    metadata: Metadata = Field(
        default_factory=Metadata,
        description="Diagram-level metadata (version, optional style).",
    )

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @model_validator(mode="after")
    def _validate_unique_node_ids(self) -> "Architecture":
        """Ensure every node id is unique within this architecture."""
        ids: list[str] = [node.id for node in self.nodes]
        seen: set[str] = set()
        duplicates: list[str] = []
        for node_id in ids:
            if node_id in seen:
                duplicates.append(node_id)
            seen.add(node_id)
        if duplicates:
            raise ValueError(
                f"Duplicate node id(s) detected: {sorted(set(duplicates))}. "
                "Every node must have a unique id."
            )
        return self

    @model_validator(mode="after")
    def _validate_edge_references(self) -> "Architecture":
        """Ensure every edge's 'from' and 'to' reference an existing node id."""
        valid_ids: set[str] = {node.id for node in self.nodes}
        bad_refs: list[str] = []
        for edge in self.edges:
            if edge.from_node not in valid_ids:
                bad_refs.append(
                    f"edge.from='{edge.from_node}' does not match any node id"
                )
            if edge.to_node not in valid_ids:
                bad_refs.append(
                    f"edge.to='{edge.to_node}' does not match any node id"
                )
        if bad_refs:
            raise ValueError(
                "Edge referential integrity failure:\n"
                + "\n".join(f"  - {r}" for r in bad_refs)
                + f"\nValid node ids: {sorted(valid_ids)}"
            )
        return self

    # ------------------------------------------------------------------
    # Convenience helpers (no side effects; safe for serialization)
    # ------------------------------------------------------------------

    def node_ids(self) -> list[str]:
        """Return an ordered list of all node ids in this architecture."""
        return [node.id for node in self.nodes]

    def adjacency(self) -> dict[str, list[str]]:
        """Return a simple adjacency mapping ``{source_id: [target_id, ...]}``.

        Useful for topological traversal during diagram layout and dataset
        generation without pulling in a graph library dependency.
        """
        adj: dict[str, list[str]] = {node.id: [] for node in self.nodes}
        for edge in self.edges:
            adj[edge.from_node].append(edge.to_node)
        return adj

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "nodes": [
                        {
                            "id": "web",
                            "type": "Frontend",
                            "label": "Web App",
                            "layer": "Presentation",
                        },
                        {
                            "id": "api",
                            "type": "Backend",
                            "label": "REST API",
                            "layer": "Application",
                        },
                        {
                            "id": "db",
                            "type": "Database",
                            "label": "PostgreSQL",
                            "layer": "Data",
                        },
                    ],
                    "edges": [
                        {"from": "web", "to": "api", "protocol": "HTTPS"},
                        {"from": "api", "to": "db", "protocol": "SQL"},
                    ],
                    "metadata": {"version": 1, "style": "light"},
                }
            ]
        }
    }
