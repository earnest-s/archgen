"""
Diagram generator for the ArchitectAI pipeline.

Converts a validated :class:`~backend.api.schemas.architecture.Architecture`
object into a PNG diagram using the ``diagrams`` library (which requires
Graphviz to be installed on the host system).

Public API::

    from backend.core.diagram.generator import generate_diagram

    path = generate_diagram(architecture, output_path="/tmp/arch")
    # → "/tmp/arch.png"
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, Optional

from diagrams import Diagram, Edge as DiagramEdge
from diagrams import Node as DiagramNode

from backend.api.schemas.architecture import Architecture, Edge, Node
from backend.core.diagram.layout import DIAGRAM_DIRECTION, sort_nodes
from backend.core.diagram.node_map import get_diagram_class, get_display_label

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def generate_diagram(
    architecture: Architecture,
    output_path: str,
    *,
    show: bool = False,
) -> str:
    """Render *architecture* as a PNG diagram file.

    The function is deterministic: identical :class:`Architecture` inputs
    always produce the same graph structure (node layout is sorted by layer
    rank; see :mod:`backend.core.diagram.layout`).

    Args:
        architecture: A fully validated :class:`Architecture` instance.
        output_path:  Destination path **without** the ``.png`` extension.
                      The ``diagrams`` library appends ``.png`` automatically.
        show:         If ``True``, open the diagram in the system viewer after
                      generation (useful for local debugging only).

    Returns:
        Absolute path of the generated PNG file (i.e. ``output_path + ".png"``).

    Raises:
        RuntimeError:  If Graphviz is not installed or diagram generation fails.
        FileNotFoundError: If the parent directory of *output_path* does not
                           exist and cannot be created.

    Example::

        arch = parse_prompt("React frontend, FastAPI backend, PostgreSQL DB")
        png_path = generate_diagram(arch, output_path="/tmp/my_arch")
        # → "/tmp/my_arch.png"
    """
    output_path = str(output_path)
    parent = Path(output_path).parent
    parent.mkdir(parents=True, exist_ok=True)

    diagram_name = _resolve_diagram_name(architecture)
    logger.info("Generating diagram %r → %s.png", diagram_name, output_path)

    # Build fast id-lookup for nodes.
    node_index: Dict[str, Node] = {n.id: n for n in architecture.nodes}

    # Sorted order ensures deterministic layout.
    ordered_nodes = sort_nodes(architecture.nodes)

    try:
        with Diagram(
            name=diagram_name,
            filename=output_path,
            direction=DIAGRAM_DIRECTION,
            show=show,
            outformat="png",
        ):
            # Step 1 – instantiate diagram nodes and keep references.
            diagram_nodes: Dict[str, DiagramNode] = {}
            for node in ordered_nodes:
                diagram_class = get_diagram_class(node.type)
                display_label = get_display_label(node.label, node.type)
                diagram_nodes[node.id] = diagram_class(display_label)
                logger.debug("Created diagram node: %s (%s)", node.id, node.type.value)

            # Step 2 – connect edges.
            for edge in architecture.edges:
                src_id, dst_id = edge.from_node, edge.to_node
                if src_id not in diagram_nodes:
                    logger.warning(
                        "Edge source %r not found in diagram nodes; skipping.", src_id
                    )
                    continue
                if dst_id not in diagram_nodes:
                    logger.warning(
                        "Edge target %r not found in diagram nodes; skipping.", dst_id
                    )
                    continue

                edge_kwargs: dict = {}
                if edge.protocol:
                    edge_kwargs["label"] = edge.protocol

                diagram_nodes[src_id] >> DiagramEdge(**edge_kwargs) >> diagram_nodes[dst_id]
                logger.debug("Connected %s → %s (%s)", src_id, dst_id, edge.protocol or "")

    except Exception as exc:
        raise RuntimeError(
            f"Diagram generation failed for output path {output_path!r}: {exc}"
        ) from exc

    final_path = output_path + ".png"
    logger.info("Diagram written to %s", final_path)
    return final_path


def generate_diagram_to_tmpfile(architecture: Architecture) -> str:
    """Convenience wrapper: render to a system temp file and return its path.

    Useful for the FastAPI route that needs to return the PNG without a
    pre-configured output directory.

    Args:
        architecture: A validated :class:`Architecture` instance.

    Returns:
        Absolute path of the generated ``.png`` temp file.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        base = os.path.join(tmpdir, "architecture")
        # We must copy out before tmpdir is cleaned up.
        tmp_path = generate_diagram(architecture, base)
        final_tmp = tempfile.mktemp(suffix=".png", prefix="architectai_")

    import shutil
    shutil.copy(tmp_path, final_tmp)
    return final_tmp


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_diagram_name(architecture: Architecture) -> str:
    """Derive a safe diagram title from architecture metadata.

    Uses ``metadata.style`` if present, otherwise falls back to
    ``"architecture"``.

    Args:
        architecture: The :class:`Architecture` model.

    Returns:
        A non-empty string suitable as a Graphviz diagram name.
    """
    style = architecture.metadata.style
    if style and style.strip():
        return style.strip().title()
    return "Architecture"


# ---------------------------------------------------------------------------
# CLI / usage example
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import sys

    from backend.core.prompt_parser.parser import parse_prompt

    prompt = (
        " ".join(sys.argv[1:])
        or "React frontend, FastAPI backend, Redis cache, PostgreSQL database"
    )
    output = sys.argv[-1] if sys.argv[-1].endswith(".png") else "architecture_output"
    # Strip .png suffix if user accidentally supplied it.
    output = output.removesuffix(".png")

    print(f"Prompt : {prompt!r}")
    arch = parse_prompt(prompt)
    print(f"Nodes  : {[n.id for n in arch.nodes]}")

    png_path = generate_diagram(arch, output_path=output)
    print(f"Output : {png_path}")
