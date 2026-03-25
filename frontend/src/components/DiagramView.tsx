import { useEffect, useMemo, useRef } from "react";
import ReactFlow, {
  addEdge,
  Background,
  Connection,
  Controls,
  Edge,
  Handle,
  MarkerType,
  Node,
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

type DiagramViewProps = {
  architecture: Architecture;
};

type LayerType = "ui" | "service" | "data";

const layerY: Record<LayerType, number> = {
  ui: 0,
  service: 150,
  data: 300,
};

const layerColor: Record<LayerType, string> = {
  ui: "#dbeafe",
  service: "#dcfce7",
  data: "#ffedd5",
};

const layerBorder: Record<LayerType, string> = {
  ui: "#60a5fa",
  service: "#4ade80",
  data: "#fb923c",
};

type NodeData = {
  label: string;
};

function UiNode({ data }: NodeProps<NodeData>) {
  return (
    <div className="arch-node arch-node-ui">
      <Handle type="target" position={Position.Top} />
      <div className="arch-node-icon">🖥️</div>
      <div className="arch-node-label">{data.label}</div>
      <Handle type="source" position={Position.Bottom} />
    </div>
  );
}

function ServiceNode({ data }: NodeProps<NodeData>) {
  return (
    <div className="arch-node arch-node-service">
      <Handle type="target" position={Position.Top} />
      <div className="arch-node-icon">⚙️</div>
      <div className="arch-node-label">{data.label}</div>
      <Handle type="source" position={Position.Bottom} />
    </div>
  );
}

function DataNode({ data }: NodeProps<NodeData>) {
  return (
    <div className="arch-node arch-node-data">
      <Handle type="target" position={Position.Top} />
      <div className="arch-node-icon">🛢️</div>
      <div className="arch-node-label">{data.label}</div>
      <Handle type="source" position={Position.Bottom} />
    </div>
  );
}

const nodeTypes = {
  uiNode: UiNode,
  serviceNode: ServiceNode,
  dataNode: DataNode,
};

function normalizeLayerType(input: string | undefined): LayerType {
  if (!input) return "service";
  const lowered = input.toLowerCase();
  if (lowered === "ui") return "ui";
  if (lowered === "data") return "data";
  return "service";
}

function toFlowNodeType(layerType: LayerType): "uiNode" | "serviceNode" | "dataNode" {
  if (layerType === "ui") return "uiNode";
  if (layerType === "data") return "dataNode";
  return "serviceNode";
}

function buildInitialNodes(architecture: Architecture): Node<NodeData>[] {
  const groups: Record<LayerType, string[]> = { ui: [], service: [], data: [] };
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
      markerEnd: { type: MarkerType.ArrowClosed },
      animated: false,
    }));
}

function DiagramViewInner({ architecture }: DiagramViewProps) {
  const initialNodes = useMemo(() => buildInitialNodes(architecture), [architecture]);
  const initialEdges = useMemo(() => buildInitialEdges(architecture), [architecture]);

  const [nodes, setNodes, onNodesChange] = useNodesState<NodeData>(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);
  const nodeCounterRef = useRef<number>(1);

  useEffect(() => {
    setNodes(initialNodes);
    setEdges(initialEdges);
    nodeCounterRef.current = 1;
  }, [initialNodes, initialEdges, setNodes, setEdges]);

  const onConnect = (connection: Connection) => {
    setEdges((currentEdges) =>
      addEdge(
        {
          ...connection,
          type: "smoothstep",
          markerEnd: { type: MarkerType.ArrowClosed },
        },
        currentEdges
      )
    );
  };

  const addNode = () => {
    const index = nodeCounterRef.current;
    nodeCounterRef.current += 1;

    const newNode: Node<NodeData> = {
      id: `service-${index}`,
      type: "serviceNode",
      data: { label: `service-${index}` },
      position: { x: 40 + index * 20, y: 150 + index * 10 },
      draggable: true,
    };

    setNodes((currentNodes) => [...currentNodes, newNode]);
  };

  const resetLayout = () => {
    setNodes(buildInitialNodes(architecture));
    setEdges(buildInitialEdges(architecture));
  };

  const clearDiagram = () => {
    setNodes([]);
    setEdges([]);
  };

  return (
    <div className="diagram-canvas">
      <div className="diagram-toolbar">
        <button type="button" className="diagram-toolbar-btn" onClick={addNode}>
          Add Node
        </button>
        <button type="button" className="diagram-toolbar-btn" onClick={resetLayout}>
          Reset Layout
        </button>
        <button type="button" className="diagram-toolbar-btn danger" onClick={clearDiagram}>
          Clear Diagram
        </button>
      </div>

      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        nodeTypes={nodeTypes}
        fitView
        nodesDraggable
        elementsSelectable
        deleteKeyCode={["Backspace", "Delete"]}
        connectionLineType="smoothstep"
      >
        <Background color="#e5e7eb" gap={18} />
        <Controls showInteractive />
      </ReactFlow>
    </div>
  );
}

export default function DiagramView({ architecture }: DiagramViewProps) {
  return (
    <ReactFlowProvider>
      <DiagramViewInner architecture={architecture} />
    </ReactFlowProvider>
  );
}
