import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ReactFlow, {
  addEdge,
  applyEdgeChanges,
  applyNodeChanges,
  Background,
  Connection,
  ConnectionLineType,
  Controls,
  Edge,
  EdgeChange,
  Handle,
  MarkerType,
  Node,
  NodeChange,
  NodeProps,
  Position,
  ReactFlowProvider,
  useEdgesState,
  useNodesState,
} from "reactflow";
import "reactflow/dist/style.css";

type ArchitectureNode = {
  id: string;
  type?: string;
  [key: string]: unknown;
};

type ArchitectureEdge = {
  source: string;
  target: string;
  [key: string]: unknown;
};

type Architecture = {
  nodes: ArchitectureNode[];
  edges: ArchitectureEdge[];
};

type EditorNodeType = "ui" | "service" | "data" | "cache" | "queue";

type EditorCommand = {
  id: number;
  action: "add" | "reset" | "clear";
  nodeType?: EditorNodeType;
};

type DiagramViewProps = {
  architecture: Architecture;
  command?: EditorCommand | null;
};

type LayerType = EditorNodeType;

const layerY: Record<LayerType, number> = {
  ui: 60,
  service: 220,
  cache: 300,
  queue: 370,
  data: 460,
};

type NodeData = {
  label: string;
};

type GlyphKind = "ui" | "service" | "data" | "cache" | "queue";

function NodeGlyph({ kind }: { kind: GlyphKind }) {
  const common = { width: 14, height: 14, viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: 2 };

  if (kind === "ui") {
    return (
      <svg {...common} className="arch-node-icon">
        <rect x="3" y="4" width="18" height="12" rx="2" />
        <path d="M8 20h8" />
        <path d="M12 16v4" />
      </svg>
    );
  }

  if (kind === "service") {
    return (
      <svg {...common} className="arch-node-icon">
        <rect x="3" y="4" width="18" height="6" rx="1" />
        <rect x="3" y="14" width="18" height="6" rx="1" />
        <path d="M7 7h.01" />
        <path d="M7 17h.01" />
      </svg>
    );
  }

  if (kind === "data") {
    return (
      <svg {...common} className="arch-node-icon">
        <ellipse cx="12" cy="6" rx="8" ry="3" />
        <path d="M4 6v8c0 1.7 3.6 3 8 3s8-1.3 8-3V6" />
      </svg>
    );
  }

  if (kind === "cache") {
    return (
      <svg {...common} className="arch-node-icon">
        <path d="M13 2 3 14h7l-1 8 10-12h-7l1-8Z" />
      </svg>
    );
  }

  return (
    <svg {...common} className="arch-node-icon">
      <rect x="4" y="5" width="16" height="4" rx="1" />
      <rect x="4" y="10" width="16" height="4" rx="1" />
      <rect x="4" y="15" width="16" height="4" rx="1" />
    </svg>
  );
}

function UiNode({ data, selected }: NodeProps<NodeData>) {
  return (
    <div className={`arch-node arch-node-ui ${selected ? "selected" : ""}`}>
      <Handle type="target" position={Position.Top} />
      <NodeGlyph kind="ui" />
      <div className="arch-node-label">{data.label}</div>
      <Handle type="source" position={Position.Bottom} />
    </div>
  );
}

function ServiceNode({ data, selected }: NodeProps<NodeData>) {
  return (
    <div className={`arch-node arch-node-service ${selected ? "selected" : ""}`}>
      <Handle type="target" position={Position.Top} />
      <NodeGlyph kind="service" />
      <div className="arch-node-label">{data.label}</div>
      <Handle type="source" position={Position.Bottom} />
    </div>
  );
}

function DataNode({ data, selected }: NodeProps<NodeData>) {
  return (
    <div className={`arch-node arch-node-data ${selected ? "selected" : ""}`}>
      <Handle type="target" position={Position.Top} />
      <NodeGlyph kind="data" />
      <div className="arch-node-label">{data.label}</div>
      <Handle type="source" position={Position.Bottom} />
    </div>
  );
}

function CacheNode({ data, selected }: NodeProps<NodeData>) {
  return (
    <div className={`arch-node arch-node-cache ${selected ? "selected" : ""}`}>
      <Handle type="target" position={Position.Top} />
      <NodeGlyph kind="cache" />
      <div className="arch-node-label">{data.label}</div>
      <Handle type="source" position={Position.Bottom} />
    </div>
  );
}

function QueueNode({ data, selected }: NodeProps<NodeData>) {
  return (
    <div className={`arch-node arch-node-queue ${selected ? "selected" : ""}`}>
      <Handle type="target" position={Position.Top} />
      <NodeGlyph kind="queue" />
      <div className="arch-node-label">{data.label}</div>
      <Handle type="source" position={Position.Bottom} />
    </div>
  );
}

const nodeTypes = {
  uiNode: UiNode,
  serviceNode: ServiceNode,
  dataNode: DataNode,
  cacheNode: CacheNode,
  queueNode: QueueNode,
};

function normalizeLayerType(input: string | undefined): LayerType {
  if (!input) return "service";
  const lowered = input.toLowerCase();
  if (lowered === "ui") return "ui";
  if (lowered === "cache") return "cache";
  if (lowered === "queue") return "queue";
  if (lowered === "data") return "data";
  return "service";
}

function toFlowNodeType(layerType: LayerType): "uiNode" | "serviceNode" | "dataNode" | "cacheNode" | "queueNode" {
  if (layerType === "ui") return "uiNode";
  if (layerType === "cache") return "cacheNode";
  if (layerType === "queue") return "queueNode";
  if (layerType === "data") return "dataNode";
  return "serviceNode";
}

function buildInitialNodes(architecture: Architecture): Node<NodeData>[] {
  const groups: Record<LayerType, string[]> = { ui: [], service: [], data: [], cache: [], queue: [] };
  architecture.nodes.forEach((node) => {
    if (typeof node.id !== "string" || !node.id) return;
    groups[normalizeLayerType(node.type)].push(node.id);
  });

  const spacing = 220;
  const out: Node<NodeData>[] = [];

  (Object.keys(groups) as LayerType[]).forEach((layer) => {
    const layerNodes = groups[layer];
    const count = layerNodes.length;
    const startX = count > 1 ? -((count - 1) * spacing) / 2 : 0;

    layerNodes.forEach((label, index) => {
      out.push({
        id: label,
        type: toFlowNodeType(layer),
        data: { label },
        position: {
          x: startX + index * spacing,
          y: layerY[layer],
        },
        draggable: true,
      });
    });
  });

  return out;
}

function buildInitialEdges(architecture: Architecture): Edge[] {
  return architecture.edges
    .filter((edge) => typeof edge.source === "string" && typeof edge.target === "string")
    .map((edge, index) => ({
      id: `e${index + 1}`,
      source: edge.source,
      target: edge.target,
      type: "smoothstep",
      style: { stroke: "#64748b", strokeWidth: 1.8 },
      markerEnd: { type: MarkerType.ArrowClosed },
      animated: false,
    }));
}

type GraphState = {
  nodes: Node<NodeData>[];
  edges: Edge[];
};

function cloneGraphState(state: GraphState): GraphState {
  return {
    nodes: state.nodes.map((node) => ({
      ...node,
      data: { ...node.data },
      position: { ...node.position },
    })),
    edges: state.edges.map((edge) => ({
      ...edge,
      markerEnd: edge.markerEnd,
      style: edge.style ? { ...edge.style } : undefined,
    })),
  };
}

function DiagramViewInner({ architecture, command }: DiagramViewProps) {
  const initialNodes = useMemo(() => buildInitialNodes(architecture), [architecture]);
  const initialEdges = useMemo(() => buildInitialEdges(architecture), [architecture]);

  const [graph, setGraph] = useState<GraphState>({ nodes: initialNodes, edges: initialEdges });
  const nodeCounterRef = useRef<number>(1);
  const historyRef = useRef<GraphState[]>([]);
  const futureRef = useRef<GraphState[]>([]);
  const restoringRef = useRef(false);
  const graphRef = useRef<GraphState>(graph);

  useEffect(() => {
    graphRef.current = graph;
  }, [graph]);

  const applyGraphChange = useCallback((updater: (current: GraphState) => GraphState) => {
    setGraph((current) => {
      if (!restoringRef.current) {
        historyRef.current.push(cloneGraphState(current));
        if (historyRef.current.length > 100) {
          historyRef.current.shift();
        }
        futureRef.current = [];
      }
      return updater(current);
    });
  }, []);

  useEffect(() => {
    restoringRef.current = true;
    setGraph({ nodes: initialNodes, edges: initialEdges });
    historyRef.current = [];
    futureRef.current = [];
    nodeCounterRef.current = 1;
    restoringRef.current = false;
  }, [initialNodes, initialEdges]);

  const undo = useCallback(() => {
    const previous = historyRef.current.pop();
    if (!previous) return;

    futureRef.current.push(cloneGraphState(graphRef.current));
    restoringRef.current = true;
    setGraph(cloneGraphState(previous));
    restoringRef.current = false;
  }, []);

  const redo = useCallback(() => {
    const next = futureRef.current.pop();
    if (!next) return;

    historyRef.current.push(cloneGraphState(graphRef.current));
    restoringRef.current = true;
    setGraph(cloneGraphState(next));
    restoringRef.current = false;
  }, []);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      const tagName = target?.tagName?.toLowerCase();
      const isEditingText = !!target && (target.isContentEditable || tagName === "input" || tagName === "textarea");
      if (isEditingText) return;

      const key = event.key.toLowerCase();
      const ctrlOrMeta = event.ctrlKey || event.metaKey;

      if (!ctrlOrMeta) return;

      if (key === "z" && !event.shiftKey) {
        event.preventDefault();
        undo();
        return;
      }

      if ((key === "z" && event.shiftKey) || key === "y") {
        event.preventDefault();
        redo();
      }
    };

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [redo, undo]);

  const onNodesChange = useCallback(
    (changes: NodeChange[]) => {
      applyGraphChange((current) => ({
        ...current,
        nodes: applyNodeChanges(changes, current.nodes),
      }));
    },
    [applyGraphChange]
  );

  const onEdgesChange = useCallback(
    (changes: EdgeChange[]) => {
      applyGraphChange((current) => ({
        ...current,
        edges: applyEdgeChanges(changes, current.edges),
      }));
    },
    [applyGraphChange]
  );

  useEffect(() => {
    if (!command) return;

    if (command.action === "add" && command.nodeType) {
      const index = nodeCounterRef.current;
      nodeCounterRef.current += 1;
      const nodeType = command.nodeType;

      const newNode: Node<NodeData> = {
        id: `${nodeType}-${index}`,
        type: toFlowNodeType(nodeType),
        data: { label: `${nodeType}-${index}` },
        position: { x: 90 + index * 18, y: layerY[nodeType] },
        draggable: true,
      };
      applyGraphChange((current) => ({ ...current, nodes: [...current.nodes, newNode] }));
      return;
    }

    if (command.action === "reset") {
      applyGraphChange(() => ({
        nodes: buildInitialNodes(architecture),
        edges: buildInitialEdges(architecture),
      }));
      return;
    }

    if (command.action === "clear") {
      applyGraphChange(() => ({ nodes: [], edges: [] }));
    }
  }, [architecture, command, applyGraphChange]);

  const onConnect = (connection: Connection) => {
    applyGraphChange((current) => ({
      ...current,
      edges: addEdge(
        {
          ...connection,
          type: "smoothstep",
          style: { stroke: "#64748b", strokeWidth: 1.8 },
          markerEnd: { type: MarkerType.ArrowClosed },
        },
        current.edges
      ),
    }));
  };

  return (
    <div className="diagram-canvas">
      <ReactFlow
        nodes={graph.nodes}
        edges={graph.edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        nodeTypes={nodeTypes}
        fitView
        nodesDraggable
        elementsSelectable
        deleteKeyCode={["Backspace", "Delete"]}
        connectionLineType={ConnectionLineType.SmoothStep}
        snapToGrid
        snapGrid={[24, 24]}
        minZoom={0.35}
        maxZoom={1.7}
        fitViewOptions={{ padding: 0.2 }}
      >
        <Background color="#d1d5db" gap={24} />
        <Controls showInteractive />
      </ReactFlow>
    </div>
  );
}

export default function DiagramView({ architecture, command }: DiagramViewProps) {
  return (
    <ReactFlowProvider>
      <DiagramViewInner architecture={architecture} command={command} />
    </ReactFlowProvider>
  );
}
