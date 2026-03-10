/**
 * Conversion helpers between the ArchitectAI Architecture JSON schema and
 * the React Flow node/edge format used by the diagram editor.
 *
 * Keeps all coordinate logic and id munging in one place so components
 * stay clean.
 */

import type {
  Edge as RFEdge,
  Node as RFNode,
} from "reactflow";

import type {
  ArchEdge,
  ArchNode,
  Architecture,
  NodeType,
} from "../types/architecture";

// ---------------------------------------------------------------------------
// Layout constants
// ---------------------------------------------------------------------------

/** Horizontal gap (px) between columns of nodes in the same layer. */
const COL_WIDTH = 200;
/** Vertical gap (px) between nodes within the same layer column. */
const ROW_HEIGHT = 100;
/** Left margin for the first column. */
const ORIGIN_X = 50;
/** Top margin for the first row. */
const ORIGIN_Y = 60;

const LAYER_ORDER: NodeType[] = [
  "Frontend",
  "Backend",
  "Service",
  "Database",
  "Cache",
  "Queue",
  "External",
];

function layerRank(type: NodeType): number {
  const idx = LAYER_ORDER.indexOf(type);
  return idx === -1 ? LAYER_ORDER.length : idx;
}

// ---------------------------------------------------------------------------
// Architecture → React Flow
// ---------------------------------------------------------------------------

/**
 * Convert an Architecture model into React Flow nodes and edges.
 *
 * Nodes are positioned deterministically in a left-to-right grid based on
 * their architectural tier (Frontend → Backend → … → External).
 */
export function architectureToFlow(arch: Architecture): {
  nodes: RFNode[];
  edges: RFEdge[];
} {
  // Group nodes by tier for column layout.
  const columns: Map<number, ArchNode[]> = new Map();
  for (const node of arch.nodes) {
    const rank = layerRank(node.type);
    if (!columns.has(rank)) columns.set(rank, []);
    columns.get(rank)!.push(node);
  }

  const rfNodes: RFNode[] = [];
  for (const [rank, group] of columns.entries()) {
    group.forEach((node, rowIdx) => {
      rfNodes.push({
        id: node.id,
        type: node.type,           // matches the nodeTypes registry key
        position: {
          x: ORIGIN_X + rank * COL_WIDTH,
          y: ORIGIN_Y + rowIdx * ROW_HEIGHT,
        },
        data: {
          label: node.label,
          nodeType: node.type,
          layer: node.layer,
        },
      });
    });
  }

  const rfEdges: RFEdge[] = arch.edges.map((e, i) => ({
    id: `e_${e.from}_${e.to}_${i}`,
    source: e.from,
    target: e.to,
    label: e.protocol ?? "",
    type: "default",
    animated: false,
  }));

  return { nodes: rfNodes, edges: rfEdges };
}

// ---------------------------------------------------------------------------
// React Flow → Architecture
// ---------------------------------------------------------------------------

/**
 * Convert React Flow nodes and edges back into an Architecture model.
 *
 * The ``data.nodeType`` and ``data.label`` fields on each RF node must have
 * been kept in sync by the editor (i.e. label renames update ``data.label``).
 */
export function flowToArchitecture(
  rfNodes: RFNode[],
  rfEdges: RFEdge[],
  metadata: Architecture["metadata"] = { version: 1 }
): Architecture {
  const nodes: ArchNode[] = rfNodes.map((n) => ({
    id: n.id,
    type: (n.data.nodeType ?? n.type) as NodeType,
    label: n.data.label ?? n.id,
    layer: n.data.layer as string | undefined,
  }));

  const edges: ArchEdge[] = rfEdges
    .filter((e) => e.source && e.target)
    .map((e) => ({
      from: e.source,
      to: e.target,
      protocol: typeof e.label === "string" && e.label ? e.label : undefined,
    }));

  return { nodes, edges, metadata };
}
