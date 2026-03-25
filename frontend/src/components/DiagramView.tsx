import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ReactFlow, {
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

type FlowNodeKind = "ui" | "service" | "database" | "cache" | "container" | "gateway" | "queue";

type NodeData = {
  label: string;
  kind: FlowNodeKind;
};

type GraphState = {
  nodes: Node<NodeData>[];
  edges: Edge[];
};

type LayerType = "ui" | "service" | "data";

const layerY: Record<LayerType, number> = {
  ui: 80,
  service: 250,
  data: 430,
};

const iconMap: Record<string, string> = {
  docker: "https://cdn.simpleicons.org/docker",
  postgresql: "https://cdn.simpleicons.org/postgresql",
  postgres: "https://cdn.simpleicons.org/postgresql",
  mysql: "https://cdn.simpleicons.org/mysql",
  mongodb: "https://cdn.simpleicons.org/mongodb",
  redis: "https://cdn.simpleicons.org/redis",
  nginx: "https://cdn.simpleicons.org/nginx",
  rabbitmq: "https://cdn.simpleicons.org/rabbitmq",
  kafka: "https://cdn.simpleicons.org/apachekafka",
  kubernetes: "https://cdn.simpleicons.org/kubernetes",
  react: "https://cdn.simpleicons.org/react",
  nodejs: "https://cdn.simpleicons.org/nodedotjs",
  express: "https://cdn.simpleicons.org/express",
  fastapi: "https://cdn.simpleicons.org/fastapi",
  python: "https://cdn.simpleicons.org/python",
  typescript: "https://cdn.simpleicons.org/typescript",
  javascript: "https://cdn.simpleicons.org/javascript",
  aws: "https://cdn.simpleicons.org/amazonwebservices",
  gcp: "https://cdn.simpleicons.org/googlecloud",
  azure: "https://cdn.simpleicons.org/microsoftazure",
};

function normalizeLabel(value: string): string {
  return value.trim().toLowerCase();
}

function detectKindFromLabel(label: string, fallbackType?: string): FlowNodeKind {
  const normalized = normalizeLabel(label);

  if (normalized.includes("docker") || normalized.includes("container")) return "container";
  if (normalized.includes("postgres") || normalized.includes("mysql") || normalized.includes("mongo") || normalized.includes("db")) {
    return "database";
  }
  if (normalized.includes("redis") || normalized.includes("cache")) return "cache";
  if (normalized.includes("nginx") || normalized.includes("gateway")) return "gateway";
  if (normalized.includes("queue") || normalized.includes("rabbitmq") || normalized.includes("kafka")) return "queue";
  if (normalized.includes("frontend") || normalized.includes("ui") || normalized.includes("client")) return "ui";

  if (fallbackType === "ui") return "ui";
  if (fallbackType === "data") return "database";
  return "service";
}

function toLayer(kind: FlowNodeKind): LayerType {
  if (kind === "ui") return "ui";
  if (kind === "database" || kind === "cache") return "data";
  return "service";
}

function toCanonicalCategory(kind: FlowNodeKind): "ui" | "service" | "database" | "cache" {
  if (kind === "ui") return "ui";
  if (kind === "database") return "database";
  if (kind === "cache") return "cache";
  return "service";
}

function nodeThemeClass(kind: FlowNodeKind): string {
  if (kind === "ui") return "arch-node-ui";
  if (kind === "database") return "arch-node-database";
  if (kind === "cache") return "arch-node-cache";
  if (kind === "container") return "arch-node-container";
  if (kind === "gateway") return "arch-node-gateway";
  if (kind === "queue") return "arch-node-queue";
  return "arch-node-service";
}

function getIconUrl(label: string): string | null {
  const normalized = normalizeLabel(label);
  if (iconMap[normalized]) return iconMap[normalized];

  const match = Object.keys(iconMap).find((key) => normalized.includes(key));
  return match ? iconMap[match] : null;
}

function FallbackGlyph({ kind }: { kind: FlowNodeKind }) {
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

  if (kind === "service" || kind === "gateway") {
    return (
      <svg {...common} className="arch-node-icon">
        <rect x="3" y="4" width="18" height="6" rx="1" />
        <rect x="3" y="14" width="18" height="6" rx="1" />
        <path d="M7 7h.01" />
        <path d="M7 17h.01" />
      </svg>
    );
  }

  if (kind === "database") {
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
  const iconUrl = getIconUrl(data.label);
  return (
    <div className={`arch-node ${nodeThemeClass(data.kind)} ${selected ? "selected" : ""}`}>
      <Handle type="target" position={Position.Top} />
      {iconUrl ? <img src={iconUrl} width={18} height={18} className="arch-node-logo" alt={data.label} /> : <FallbackGlyph kind="ui" />}
      <div className="arch-node-label">{data.label}</div>
      <Handle type="source" position={Position.Bottom} />
    </div>
  );
}

function ServiceNode({ data, selected }: NodeProps<NodeData>) {
  const iconUrl = getIconUrl(data.label);
  return (
    <div className={`arch-node ${nodeThemeClass(data.kind)} ${selected ? "selected" : ""}`}>
      <Handle type="target" position={Position.Top} />
      {iconUrl ? <img src={iconUrl} width={18} height={18} className="arch-node-logo" alt={data.label} /> : <FallbackGlyph kind={data.kind} />}
      <div className="arch-node-label">{data.label}</div>
      <Handle type="source" position={Position.Bottom} />
    </div>
  );
}

function DataNode({ data, selected }: NodeProps<NodeData>) {
  const iconUrl = getIconUrl(data.label);
  return (
    <div className={`arch-node ${nodeThemeClass(data.kind)} ${selected ? "selected" : ""}`}>
      <Handle type="target" position={Position.Top} />
      {iconUrl ? <img src={iconUrl} width={18} height={18} className="arch-node-logo" alt={data.label} /> : <FallbackGlyph kind={data.kind} />}
      <div className="arch-node-label">{data.label}</div>
      <Handle type="source" position={Position.Bottom} />
    </div>
  );
}

function CacheNode({ data, selected }: NodeProps<NodeData>) {
  const iconUrl = getIconUrl(data.label);
  return (
    <div className={`arch-node ${nodeThemeClass(data.kind)} ${selected ? "selected" : ""}`}>
      <Handle type="target" position={Position.Top} />
      {iconUrl ? <img src={iconUrl} width={18} height={18} className="arch-node-logo" alt={data.label} /> : <FallbackGlyph kind="cache" />}
      <div className="arch-node-label">{data.label}</div>
      <Handle type="source" position={Position.Bottom} />
    </div>
  );
}

function QueueNode({ data, selected }: NodeProps<NodeData>) {
  const iconUrl = getIconUrl(data.label);
  return (
    <div className={`arch-node ${nodeThemeClass(data.kind)} ${selected ? "selected" : ""}`}>
      <Handle type="target" position={Position.Top} />
      {iconUrl ? <img src={iconUrl} width={18} height={18} className="arch-node-logo" alt={data.label} /> : <FallbackGlyph kind="queue" />}
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

function toFlowNodeType(kind: FlowNodeKind): "uiNode" | "serviceNode" | "dataNode" | "cacheNode" | "queueNode" {
  if (kind === "ui") return "uiNode";
  if (kind === "database") return "dataNode";
  if (kind === "cache") return "cacheNode";
  if (kind === "queue") return "queueNode";
  return "serviceNode";
}

function edgeStyle() {
  return { stroke: "#64748b", strokeWidth: 1.8 };
}

function buildNodesFromArchitecture(architecture: Architecture): Node<NodeData>[] {
  const grouped: Record<LayerType, Node<NodeData>[]> = { ui: [], service: [], data: [] };

  architecture.nodes.forEach((node) => {
    if (typeof node.id !== "string" || !node.id) return;
    const kind = detectKindFromLabel(node.id, typeof node.type === "string" ? node.type : undefined);
    const layer = toLayer(kind);

    grouped[layer].push({
      id: node.id,
      type: toFlowNodeType(kind),
      data: { label: node.id, kind },
      position: { x: 0, y: layerY[layer] },
      draggable: true,
      selectable: true,
    });
  });

  const spacing = 190;
  const output: Node<NodeData>[] = [];
  (Object.keys(grouped) as LayerType[]).forEach((layer) => {
    const nodes = grouped[layer];
    const startX = nodes.length > 1 ? -((nodes.length - 1) * spacing) / 2 : 0;
    nodes.forEach((node, index) => {
      output.push({ ...node, position: { x: startX + index * spacing, y: layerY[layer] } });
    });
  });

  return output;
}

function isAllowedHierarchyEdge(source: Node<NodeData> | undefined, target: Node<NodeData> | undefined): boolean {
  if (!source || !target) return false;

  const sourceCategory = toCanonicalCategory(source.data.kind);
  const targetCategory = toCanonicalCategory(target.data.kind);
  if (sourceCategory === "ui" && targetCategory === "service") return true;
  if (sourceCategory === "service" && (targetCategory === "database" || targetCategory === "cache")) return true;
  return false;
}

function dedupeEdges(edges: Edge[]): Edge[] {
  const seen = new Set<string>();
  const output: Edge[] = [];

  edges.forEach((edge) => {
    const key = `${edge.source}->${edge.target}`;
    if (edge.source === edge.target || seen.has(key)) return;
    seen.add(key);
    output.push({
      ...edge,
      type: "smoothstep",
      style: edgeStyle(),
      markerEnd: { type: MarkerType.ArrowClosed },
      animated: false,
    });
  });

  return output;
}

function buildEdgesFromHierarchy(nodes: Node<NodeData>[]): Edge[] {
  const ui = nodes.filter((node) => toCanonicalCategory(node.data.kind) === "ui");
  const services = nodes.filter((node) => toCanonicalCategory(node.data.kind) === "service");
  const databases = nodes.filter((node) => toCanonicalCategory(node.data.kind) === "database");
  const caches = nodes.filter((node) => toCanonicalCategory(node.data.kind) === "cache");

  const edges: Edge[] = [];
  let id = 1;

  ui.forEach((src) => {
    services.forEach((dst) => {
      edges.push({ id: `e${id++}`, source: src.id, target: dst.id, type: "smoothstep", style: edgeStyle(), markerEnd: { type: MarkerType.ArrowClosed } });
    });
  });

  services.forEach((src) => {
    databases.forEach((dst) => {
      edges.push({ id: `e${id++}`, source: src.id, target: dst.id, type: "smoothstep", style: edgeStyle(), markerEnd: { type: MarkerType.ArrowClosed } });
    });
  });

  services.forEach((src) => {
    caches.forEach((dst) => {
      edges.push({ id: `e${id++}`, source: src.id, target: dst.id, type: "smoothstep", style: edgeStyle(), markerEnd: { type: MarkerType.ArrowClosed } });
    });
  });

  return dedupeEdges(edges);
}

function buildEdgesFromArchitecture(nodes: Node<NodeData>[], architecture: Architecture): Edge[] {
  const byId = new Map(nodes.map((node) => [node.id, node]));

  const candidateEdges: Edge[] = architecture.edges
    .filter((edge) => typeof edge.source === "string" && typeof edge.target === "string")
    .map((edge, index) => ({
      id: `e${index + 1}`,
      source: edge.source,
      target: edge.target,
      type: "smoothstep",
      style: edgeStyle(),
      markerEnd: { type: MarkerType.ArrowClosed },
      animated: false,
    }))
    .filter((edge) => isAllowedHierarchyEdge(byId.get(edge.source), byId.get(edge.target)));

  const deduped = dedupeEdges(candidateEdges);
  if (deduped.length > 0) return deduped;
  return buildEdgesFromHierarchy(nodes);
}

async function applyDagreLayout(nodes: Node<NodeData>[], edges: Edge[]): Promise<Node<NodeData>[]> {
  try {
    const dagreModule = await import(/* @vite-ignore */ "https://esm.sh/dagre@0.8.5");
    const dagreLib = (dagreModule as { default?: unknown; graphlib?: unknown; layout?: unknown }).default as
      | { graphlib: { Graph: new () => { setGraph: (g: object) => void; setDefaultEdgeLabel: (f: () => object) => void; setNode: (id: string, v: object) => void; setEdge: (s: string, t: string) => void; node: (id: string) => { x: number; y: number } } }; layout: (g: unknown) => void }
      | undefined;

    if (!dagreLib?.graphlib?.Graph || !dagreLib.layout) {
      return nodes;
    }

    const g = new dagreLib.graphlib.Graph();
    g.setGraph({ rankdir: "TB", ranksep: 90, nodesep: 50, marginx: 20, marginy: 20 });
    g.setDefaultEdgeLabel(() => ({}));

    nodes.forEach((node) => {
      g.setNode(node.id, { width: 150, height: 42 });
    });
    edges.forEach((edge) => {
      g.setEdge(edge.source, edge.target);
    });

    dagreLib.layout(g);

    return nodes.map((node) => {
      const positioned = g.node(node.id);
      return {
        ...node,
        position: {
          x: positioned.x - 75,
          y: positioned.y - 21,
        },
      };
    });
  } catch {
    return nodes;
  }
}

function buildInitialGraph(architecture: Architecture): GraphState {
  const nodes = buildNodesFromArchitecture(architecture);
  const edges = buildEdgesFromArchitecture(nodes, architecture);
  return { nodes, edges };
}

function editorCommandToKind(nodeType: EditorNodeType | undefined): FlowNodeKind {
  if (nodeType === "ui") return "ui";
  if (nodeType === "cache") return "cache";
  if (nodeType === "queue") return "queue";
  if (nodeType === "data") return "database";
  return "service";
}

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
  const initialGraph = useMemo(() => buildInitialGraph(architecture), [architecture]);

  const [graph, setGraph] = useState<GraphState>(initialGraph);
  const nodeCounterRef = useRef<number>(1);
  const historyRef = useRef<GraphState[]>([]);
  const futureRef = useRef<GraphState[]>([]);
  const restoringRef = useRef(false);
  const graphRef = useRef<GraphState>(graph);
  const importInputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    graphRef.current = graph;
  }, [graph]);

  const applyGraphChange = useCallback((updater: (current: GraphState) => GraphState) => {
    setGraph((current) => {
      if (!restoringRef.current) {
        historyRef.current.push(cloneGraphState(current));
        if (historyRef.current.length > 50) {
          historyRef.current.shift();
        }
        futureRef.current = [];
      }
      return updater(current);
    });
  }, []);

  const replaceGraph = useCallback((next: GraphState) => {
    applyGraphChange(() => cloneGraphState(next));
  }, [applyGraphChange]);

  const applyAutoLayout = useCallback(async (state: GraphState) => {
    const laidOutNodes = await applyDagreLayout(state.nodes, state.edges);
    setGraph((current) => {
      if (current !== state) return current;
      return { ...current, nodes: laidOutNodes };
    });
  }, []);

  useEffect(() => {
    restoringRef.current = true;
    setGraph(initialGraph);
    historyRef.current = [];
    futureRef.current = [];
    nodeCounterRef.current = 1;
    restoringRef.current = false;
    void applyAutoLayout(initialGraph);
  }, [initialGraph, applyAutoLayout]);

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
        return;
      }

      if (key === "backspace" || key === "delete") {
        event.preventDefault();
        applyGraphChange((current) => {
          const selectedNodeIds = new Set(current.nodes.filter((node) => node.selected).map((node) => node.id));
          const remainingNodes = current.nodes.filter((node) => !selectedNodeIds.has(node.id));
          const remainingEdges = current.edges.filter(
            (edge) => !edge.selected && !selectedNodeIds.has(edge.source) && !selectedNodeIds.has(edge.target)
          );
          return { nodes: remainingNodes, edges: remainingEdges };
        });
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
      const nodeKind = editorCommandToKind(command.nodeType);

      const newNode: Node<NodeData> = {
        id: `${nodeKind}-${index}`,
        type: toFlowNodeType(nodeKind),
        data: { label: `${nodeKind}-${index}`, kind: nodeKind },
        position: { x: 90 + index * 18, y: layerY[toLayer(nodeKind)] },
        draggable: true,
      };
      applyGraphChange((current) => ({ ...current, nodes: [...current.nodes, newNode] }));
      return;
    }

    if (command.action === "reset") {
      const next = buildInitialGraph(architecture);
      replaceGraph(next);
      void applyAutoLayout(next);
      return;
    }

    if (command.action === "clear") {
      applyGraphChange(() => ({ nodes: [], edges: [] }));
    }
  }, [architecture, command, applyAutoLayout, applyGraphChange, replaceGraph]);

  const onConnect = (connection: Connection) => {
    applyGraphChange((current) => {
      if (!connection.source || !connection.target) return current;

      const sourceNode = current.nodes.find((node) => node.id === connection.source);
      const targetNode = current.nodes.find((node) => node.id === connection.target);
      if (!isAllowedHierarchyEdge(sourceNode, targetNode)) return current;

      const duplicate = current.edges.some((edge) => edge.source === connection.source && edge.target === connection.target);
      if (duplicate) return current;

      const created: Edge = {
        id: `e${Date.now()}`,
        source: connection.source,
        target: connection.target,
        type: "smoothstep",
        style: edgeStyle(),
        markerEnd: { type: MarkerType.ArrowClosed },
        animated: false,
      };

      return { ...current, edges: [...current.edges, created] };
    });
  };

  const onExportJson = () => {
    const payload = JSON.stringify({ nodes: graph.nodes, edges: graph.edges }, null, 2);
    const blob = new Blob([payload], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "architecture-diagram.json";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const onImportJson = () => {
    importInputRef.current?.click();
  };

  const onImportFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    try {
      const text = await file.text();
      const parsed = JSON.parse(text) as { nodes?: unknown; edges?: unknown };

      if (!Array.isArray(parsed.nodes) || !Array.isArray(parsed.edges)) return;

      const importedNodes: Node<NodeData>[] = parsed.nodes
        .map((node, index) => {
          if (!node || typeof node !== "object") return null;
          const candidate = node as { id?: unknown; data?: { label?: unknown; kind?: unknown }; type?: unknown; position?: { x?: unknown; y?: unknown } };
          if (typeof candidate.id !== "string") return null;

          const label = typeof candidate.data?.label === "string" ? candidate.data.label : candidate.id;
          const kind = detectKindFromLabel(label, typeof candidate.type === "string" ? candidate.type : undefined);
          const x = typeof candidate.position?.x === "number" ? candidate.position.x : index * 50;
          const y = typeof candidate.position?.y === "number" ? candidate.position.y : layerY[toLayer(kind)];

          return {
            id: candidate.id,
            type: toFlowNodeType(kind),
            data: { label, kind },
            position: { x, y },
            draggable: true,
            selectable: true,
          } as Node<NodeData>;
        })
        .filter((node): node is Node<NodeData> => node !== null);

      const nodeIdSet = new Set(importedNodes.map((node) => node.id));

      const importedEdges: Edge[] = dedupeEdges(
        parsed.edges
          .map((edge, index) => {
            if (!edge || typeof edge !== "object") return null;
            const candidate = edge as { source?: unknown; target?: unknown };
            if (typeof candidate.source !== "string" || typeof candidate.target !== "string") return null;
            if (!nodeIdSet.has(candidate.source) || !nodeIdSet.has(candidate.target)) return null;

            return {
              id: `i${index + 1}`,
              source: candidate.source,
              target: candidate.target,
              type: "smoothstep",
              style: edgeStyle(),
              markerEnd: { type: MarkerType.ArrowClosed },
            } as Edge;
          })
          .filter((edge): edge is Edge => edge !== null)
      );

      const nextState: GraphState = {
        nodes: importedNodes,
        edges: importedEdges.length > 0 ? importedEdges : buildEdgesFromHierarchy(importedNodes),
      };

      replaceGraph(nextState);
      void applyAutoLayout(nextState);
    } catch {
      // Intentionally ignore malformed imports to keep editor responsive.
    } finally {
      event.target.value = "";
    }
  };

  return (
    <div className="diagram-canvas">
      <div className="diagram-menubar">
        <div className="menu-group">
          <span className="menu-title">File</span>
          <button type="button" className="menu-btn" onClick={onExportJson}>Export JSON</button>
          <button type="button" className="menu-btn" onClick={onImportJson}>Import JSON</button>
          <button type="button" className="menu-btn danger" onClick={() => applyGraphChange(() => ({ nodes: [], edges: [] }))}>Clear Diagram</button>
        </div>
        <div className="menu-group">
          <span className="menu-title">Edit</span>
          <button type="button" className="menu-btn" onClick={undo}>Undo</button>
          <button type="button" className="menu-btn" onClick={redo}>Redo</button>
        </div>
      </div>
      <input ref={importInputRef} type="file" accept="application/json" className="hidden-input" onChange={onImportFileChange} />

      <ReactFlow
        nodes={graph.nodes}
        edges={graph.edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        nodeTypes={nodeTypes}
        fitView
        nodesDraggable
        panOnDrag
        zoomOnScroll
        zoomOnPinch
        zoomOnDoubleClick
        elementsSelectable
        deleteKeyCode={null}
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
