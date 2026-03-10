/**
 * Custom React Flow node types for the ArchitectAI diagram editor.
 *
 * Each NodeType gets its own visual component so the editor canvas
 * clearly communicates architectural role via colour and icon text.
 */

import React from "react";
import { Handle, Position } from "reactflow";
import type { NodeProps } from "reactflow";

import type { NodeType } from "../types/architecture";

// ---------------------------------------------------------------------------
// Colour palette – one colour per architectural tier
// ---------------------------------------------------------------------------
const BG_COLORS: Record<NodeType, string> = {
  Frontend:  "#dbeafe",  // blue-100
  Backend:   "#d1fae5",  // green-100
  Service:   "#ede9fe",  // violet-100
  Database:  "#fef3c7",  // amber-100
  Cache:     "#fee2e2",  // red-100
  Queue:     "#ffedd5",  // orange-100
  External:  "#f1f5f9",  // slate-100
};

const BORDER_COLORS: Record<NodeType, string> = {
  Frontend:  "#3b82f6",
  Backend:   "#10b981",
  Service:   "#8b5cf6",
  Database:  "#f59e0b",
  Cache:     "#ef4444",
  Queue:     "#f97316",
  External:  "#94a3b8",
};

// ---------------------------------------------------------------------------
// Generic node component shared across all NodeTypes
// ---------------------------------------------------------------------------

interface ArchNodeData {
  label: string;
  nodeType: NodeType;
  onLabelChange?: (newLabel: string) => void;
}

const ArchNode: React.FC<NodeProps<ArchNodeData>> = ({ data, selected }) => {
  const bg     = BG_COLORS[data.nodeType]     ?? "#f8fafc";
  const border = BORDER_COLORS[data.nodeType] ?? "#cbd5e1";

  return (
    <div
      style={{
        background: bg,
        border: `2px solid ${selected ? "#1d4ed8" : border}`,
        borderRadius: 10,
        padding: "8px 14px",
        minWidth: 120,
        textAlign: "center",
        boxShadow: selected ? "0 0 0 3px rgba(59,130,246,0.3)" : undefined,
        transition: "box-shadow 0.15s",
      }}
    >
      <Handle type="target" position={Position.Left} />
      <div style={{ fontSize: 10, color: "#6b7280", marginBottom: 2 }}>
        {data.nodeType}
      </div>
      <div style={{ fontWeight: 600, fontSize: 13, color: "#1e293b" }}>
        {data.label}
      </div>
      <Handle type="source" position={Position.Right} />
    </div>
  );
};

// ---------------------------------------------------------------------------
// One named export per NodeType (React Flow requires separate entries)
// ---------------------------------------------------------------------------

export const FrontendNode:  React.FC<NodeProps<ArchNodeData>> = (p) => <ArchNode {...p} />;
export const BackendNode:   React.FC<NodeProps<ArchNodeData>> = (p) => <ArchNode {...p} />;
export const ServiceNode:   React.FC<NodeProps<ArchNodeData>> = (p) => <ArchNode {...p} />;
export const DatabaseNode:  React.FC<NodeProps<ArchNodeData>> = (p) => <ArchNode {...p} />;
export const CacheNode:     React.FC<NodeProps<ArchNodeData>> = (p) => <ArchNode {...p} />;
export const QueueNode:     React.FC<NodeProps<ArchNodeData>> = (p) => <ArchNode {...p} />;
export const ExternalNode:  React.FC<NodeProps<ArchNodeData>> = (p) => <ArchNode {...p} />;

/** Registry passed to ``<ReactFlow nodeTypes={...} />`` */
export const nodeTypes = {
  Frontend:  FrontendNode,
  Backend:   BackendNode,
  Service:   ServiceNode,
  Database:  DatabaseNode,
  Cache:     CacheNode,
  Queue:     QueueNode,
  External:  ExternalNode,
};
