/**
 * DiagramEditor
 *
 * A full React Flow canvas that renders an Architecture as an interactive
 * diagram. Supports:
 *  - drag-to-move nodes
 *  - draw new edges by dragging from a handle
 *  - delete selected nodes/edges (Backspace or Delete key)
 *  - inline label rename (double-click a node)
 *  - add a new node via the toolbar button
 *  - sync changes back via onArchitectureChange callback
 */

import React, { useCallback, useEffect, useRef, useState } from "react";
import ReactFlow, {
  addEdge,
  Background,
  Connection,
  Controls,
  Edge,
  MiniMap,
  Node,
  Panel,
  ReactFlowProvider,
  useEdgesState,
  useNodesState,
  useReactFlow,
} from "reactflow";
import "reactflow/dist/style.css";

import type { Architecture, NodeType } from "../types/architecture";
import { edgeTypes } from "../editor/edgeTypes";
import { nodeTypes } from "../editor/nodeTypes";
import { architectureToFlow, flowToArchitecture } from "../editor/schema";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface Props {
  architecture: Architecture;
  onArchitectureChange?: (updated: Architecture) => void;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const NODE_TYPE_OPTIONS: NodeType[] = [
  "Frontend", "Backend", "Service", "Database", "Cache", "Queue", "External",
];

function generateId(): string {
  return `node_${Math.random().toString(36).slice(2, 8)}`;
}

// ---------------------------------------------------------------------------
// Inner toolbar – must live inside ReactFlow to access useReactFlow()
// ---------------------------------------------------------------------------

interface ToolbarProps {
  onAddNode: (type: NodeType) => void;
  onDeleteSelected: () => void;
}

const InnerToolbar: React.FC<ToolbarProps> = ({ onAddNode, onDeleteSelected }) => {
  const { fitView } = useReactFlow();

  return (
    <Panel position="top-left">
      <div className="flex flex-wrap gap-1 bg-white/90 backdrop-blur-sm
                      rounded-lg shadow p-2 border border-gray-200">
        {/* Node type buttons */}
        {NODE_TYPE_OPTIONS.map((type) => (
          <button
            key={type}
            onClick={() => onAddNode(type)}
            className="rounded px-2 py-1 text-xs font-medium text-gray-700
                       bg-gray-100 hover:bg-gray-200 transition-colors"
            title={`Add ${type} node`}
          >
            + {type}
          </button>
        ))}

        <div className="w-px bg-gray-300 mx-1 self-stretch" />

        {/* Reset layout */}
        <button
          onClick={() => fitView({ padding: 0.15, duration: 300 })}
          className="rounded px-2 py-1 text-xs font-medium text-blue-700
                     bg-blue-50 hover:bg-blue-100 transition-colors"
          title="Reset layout (fit view)"
        >
          ⊟ Fit
        </button>

        {/* Delete selected */}
        <button
          onClick={onDeleteSelected}
          className="rounded px-2 py-1 text-xs font-medium text-red-700
                     bg-red-50 hover:bg-red-100 transition-colors"
          title="Delete selected nodes/edges"
        >
          ✕ Delete
        </button>
      </div>
    </Panel>
  );
};



export const DiagramEditor: React.FC<Props> = ({
  architecture,
  onArchitectureChange,
}) => {
  const { nodes: initNodes, edges: initEdges } = architectureToFlow(architecture);
  const [nodes, setNodes, onNodesChange] = useNodesState(initNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initEdges);

  // Track which node is being renamed.
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [renameValue, setRenameValue] = useState<string>("");
  const renameRef = useRef<HTMLInputElement>(null);

  // Track selected elements so the toolbar "Delete" button knows what to remove.
  const [selectedNodeIds, setSelectedNodeIds] = useState<Set<string>>(new Set());
  const [selectedEdgeIds, setSelectedEdgeIds] = useState<Set<string>>(new Set());

  // Sync inbound architecture prop changes to canvas state.
  useEffect(() => {
    const { nodes: n, edges: e } = architectureToFlow(architecture);
    setNodes(n);
    setEdges(e);
  }, [architecture]);

  // Notify parent whenever canvas state changes.
  const notifyChange = useCallback(
    (updatedNodes: Node[], updatedEdges: Edge[]) => {
      if (!onArchitectureChange) return;
      const updated = flowToArchitecture(
        updatedNodes,
        updatedEdges,
        architecture.metadata
      );
      onArchitectureChange(updated);
    },
    [onArchitectureChange, architecture.metadata]
  );

  // ── Edge connection ────────────────────────────────────────────────────────
  const onConnect = useCallback(
    (connection: Connection) => {
      const newEdges = addEdge({ ...connection, type: "default" }, edges);
      setEdges(newEdges);
      notifyChange(nodes, newEdges);
    },
    [edges, nodes, notifyChange]
  );

  // ── Node double-click → rename mode ────────────────────────────────────────
  const onNodeDoubleClick = useCallback(
    (_: React.MouseEvent, node: Node) => {
      setRenamingId(node.id);
      setRenameValue(node.data.label as string);
      setTimeout(() => renameRef.current?.select(), 50);
    },
    []
  );

  const commitRename = useCallback(() => {
    if (!renamingId) return;
    const updated = nodes.map((n) =>
      n.id === renamingId
        ? { ...n, data: { ...n.data, label: renameValue } }
        : n
    );
    setNodes(updated);
    notifyChange(updated, edges);
    setRenamingId(null);
  }, [renamingId, renameValue, nodes, edges, notifyChange]);

  // ── Add new node ───────────────────────────────────────────────────────────
  const addNode = useCallback(
    (type: NodeType) => {
      const id = generateId();
      const newNode: Node = {
        id,
        type,
        position: { x: 200 + Math.random() * 100, y: 150 + Math.random() * 100 },
        data: { label: type, nodeType: type },
      };
      const updated = [...nodes, newNode];
      setNodes(updated);
      notifyChange(updated, edges);
    },
    [nodes, edges, notifyChange]
  );

  // ── Delete selected elements (handled by ReactFlow internally via keyboard) ─
  const onNodesDelete = useCallback(
    (deleted: Node[]) => {
      const deletedIds = new Set(deleted.map((n) => n.id));
      const remainingEdges = edges.filter(
        (e) => !deletedIds.has(e.source) && !deletedIds.has(e.target)
      );
      notifyChange(
        nodes.filter((n) => !deletedIds.has(n.id)),
        remainingEdges
      );
    },
    [nodes, edges, notifyChange]
  );

  const onEdgesDelete = useCallback(
    (deleted: Edge[]) => {
      const deletedIds = new Set(deleted.map((e) => e.id));
      notifyChange(nodes, edges.filter((e) => !deletedIds.has(e.id)));
    },
    [nodes, edges, notifyChange]
  );

  return (
    <div className="relative w-full h-full" style={{ minHeight: 500 }}>
      {/* Rename overlay */}
      {renamingId && (
        <div
          className="absolute inset-0 z-50 flex items-center justify-center bg-black/20"
          onClick={commitRename}
        >
          <div
            className="bg-white rounded-lg shadow-xl p-4 flex flex-col gap-3 min-w-[260px]"
            onClick={(e) => e.stopPropagation()}
          >
            <label className="text-sm font-semibold text-gray-700">
              Rename node
            </label>
            <input
              ref={renameRef}
              value={renameValue}
              onChange={(e) => setRenameValue(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") commitRename();
                if (e.key === "Escape") setRenamingId(null);
              }}
              className="border border-gray-300 rounded px-3 py-1.5 text-sm
                         focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
            <div className="flex gap-2 justify-end">
              <button
                onClick={() => setRenamingId(null)}
                className="text-sm text-gray-500 hover:text-gray-700"
              >
                Cancel
              </button>
              <button
                onClick={commitRename}
                className="rounded bg-blue-600 px-4 py-1 text-sm font-semibold
                           text-white hover:bg-blue-700"
              >
                Rename
              </button>
            </div>
          </div>
        </div>
      )}

      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onNodeDoubleClick={onNodeDoubleClick}
        onNodesDelete={onNodesDelete}
        onEdgesDelete={onEdgesDelete}
        deleteKeyCode={["Backspace", "Delete"]}
        fitView
      >
        <Controls />
        <MiniMap />
        <Background gap={16} />

        {/* Add-node toolbar */}
        <Panel position="top-left">
          <div className="flex flex-wrap gap-1 bg-white/90 backdrop-blur-sm
                          rounded-lg shadow p-2 border border-gray-200">
            {NODE_TYPE_OPTIONS.map((type) => (
              <button
                key={type}
                onClick={() => addNode(type)}
                className="rounded px-2 py-1 text-xs font-medium text-gray-700
                           bg-gray-100 hover:bg-gray-200 transition-colors"
                title={`Add ${type} node`}
              >
                + {type}
              </button>
            ))}
          </div>
        </Panel>
      </ReactFlow>
    </div>
  );
};

export default DiagramEditor;
