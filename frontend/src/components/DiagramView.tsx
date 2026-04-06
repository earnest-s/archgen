import { ChangeEvent, DragEvent, KeyboardEvent as ReactKeyboardEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { toPng } from "html-to-image";
import { FiBox, FiMoon, FiMove, FiMousePointer, FiPlusCircle, FiSun, FiTrash2 } from "react-icons/fi";
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
  ReactFlowInstance,
  useReactFlow,
} from "reactflow";
import {
  siApachekafka,
  siDocker,
  siFastapi,
  siMongodb,
  siMysql,
  siNodedotjs,
  siNginx,
  siPostgresql,
  siReact,
  siRedis,
} from "simple-icons";
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

type EditorNodeType = "ui" | "service" | "data" | "cache" | "queue" | "container";

type EditorCommand = {
  id: number;
  action: "reset" | "clear";
};

type DiagramViewProps = {
  architecture: Architecture;
  command?: EditorCommand | null;
  theme: "light" | "dark";
  onToggleTheme: () => void;
};

type FlowNodeKind = "ui" | "service" | "database" | "cache" | "container" | "gateway" | "queue";
type EdgeProtocol = "request" | "HTTP" | "gRPC" | "Async" | "Cache" | "DB Query";
type EdgeLine = "sync" | "async";
type ToolMode = "select" | "connect" | "delete" | "pan";

type NodeData = {
  label: string;
  kind: FlowNodeKind;
  editing?: boolean;
  onStartEdit?: (nodeId: string) => void;
  onCommitLabel?: (nodeId: string, label: string) => void;
};

type EdgeData = {
  edgeType: EdgeProtocol;
  lineStyle: EdgeLine;
};

type GraphState = {
  nodes: Node<NodeData>[];
  edges: Edge<EdgeData>[];
};

type LayerType = "ui" | "service" | "data";

const layerY: Record<LayerType, number> = {
  ui: 80,
  service: 250,
  data: 430,
};

const CONTAINER_WIDTH = 340;
const CONTAINER_HEIGHT = 230;
const DEFAULT_NODE_WIDTH = 150;
const DEFAULT_NODE_HEIGHT = 40;

const iconMap: Record<string, string> = {
  docker: "docker",
  postgresql: "postgres",
  postgres: "postgres",
  mysql: "mysql",
  mongodb: "mongodb",
  redis: "redis",
  nginx: "nginx",
  kafka: "kafka",
  react: "react",
  nodejs: "node",
  node: "node",
  fastapi: "fastapi",
};

const simpleIconMap = {
  docker: siDocker,
  postgres: siPostgresql,
  mysql: siMysql,
  mongodb: siMongodb,
  redis: siRedis,
  kafka: siApachekafka,
  nginx: siNginx,
  react: siReact,
  node: siNodedotjs,
  fastapi: siFastapi,
} as const;

function normalizeLabel(value: string): string {
  return value.trim().toLowerCase();
}

function detectKindFromLabel(label: string, fallbackType?: string): FlowNodeKind {
  const normalized = normalizeLabel(label);
  if (normalized.includes("docker") || normalized.includes("container")) return "container";
  if (
    normalized.includes("postgres") ||
    normalized.includes("mysql") ||
    normalized.includes("mongo") ||
    normalized.includes("db") ||
    normalized.includes("database")
  ) {
    return "database";
  }
  if (normalized.includes("redis") || normalized.includes("cache")) return "cache";
  if (normalized.includes("nginx") || normalized.includes("gateway")) return "gateway";
  if (normalized.includes("queue") || normalized.includes("rabbitmq") || normalized.includes("kafka")) return "queue";
  if (normalized.includes("frontend") || normalized.includes("ui") || normalized.includes("client")) return "ui";

  if (fallbackType === "ui") return "ui";
  if (fallbackType === "database") return "database";
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

function categoryForKind(kind: FlowNodeKind): "ui" | "service" | "database" | "cache" | "queue" {
  if (kind === "queue") return "queue";
  return toCanonicalCategory(kind);
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

function getIconKey(label: string): keyof typeof simpleIconMap | null {
  const normalized = normalizeLabel(label);
  if (iconMap[normalized]) return iconMap[normalized] as keyof typeof simpleIconMap;
  const key = Object.keys(iconMap).find((item) => normalized.includes(item));
  return key ? (iconMap[key] as keyof typeof simpleIconMap) : null;
}

function getProtocolVisual(edgeType: EdgeProtocol, lineStyle: EdgeLine): {
  style: React.CSSProperties;
  labelStyle: React.CSSProperties;
  labelBgStyle: React.CSSProperties;
} {
  const baseColor = edgeType === "HTTP" ? "#2563eb" : edgeType === "DB Query" ? "#ea580c" : edgeType === "Async" ? "#7c3aed" : edgeType === "Cache" ? "#0d9488" : "#64748b";
  return {
    style: {
      stroke: baseColor,
      strokeWidth: 2,
      strokeDasharray: lineStyle === "async" ? "6 4" : undefined,
    },
    labelStyle: {
      fontSize: 10,
      fill: "#0f172a",
      fontWeight: 600,
    },
    labelBgStyle: {
      fill: "#f8fafc",
      fillOpacity: 0.92,
      stroke: "#cbd5e1",
      strokeWidth: 1,
    },
  };
}

function protocolFromKinds(source: Node<NodeData> | undefined, target: Node<NodeData> | undefined): EdgeProtocol {
  if (!source || !target) return "request";
  const targetCat = categoryForKind(target.data.kind);
  if (targetCat === "database") return "DB Query";
  if (targetCat === "queue") return "Async";
  if (targetCat === "cache") return "Cache";
  return "HTTP";
}

function normalizeProtocol(value: string | null | undefined): EdgeProtocol {
  const normalized = (value ?? "").trim().toLowerCase();
  if (normalized === "http") return "HTTP";
  if (normalized === "grpc") return "gRPC";
  if (normalized === "queue" || normalized === "async") return "Async";
  if (normalized === "cache") return "Cache";
  if (normalized === "db query" || normalized === "db") return "DB Query";
  if (normalized === "request") return "request";
  return "request";
}

function toFlowNodeType(kind: FlowNodeKind): "uiNode" | "serviceNode" | "dataNode" | "cacheNode" | "queueNode" | "containerNode" {
  if (kind === "ui") return "uiNode";
  if (kind === "database") return "dataNode";
  if (kind === "cache") return "cacheNode";
  if (kind === "queue") return "queueNode";
  if (kind === "container") return "containerNode";
  return "serviceNode";
}

function templateToKind(template: EditorNodeType): FlowNodeKind {
  if (template === "data") return "database";
  return template;
}

function getNodeSize(node: Node<NodeData>): { width: number; height: number } {
  if (node.type === "containerNode") return { width: CONTAINER_WIDTH, height: CONTAINER_HEIGHT };
  return { width: DEFAULT_NODE_WIDTH, height: DEFAULT_NODE_HEIGHT };
}

function getAbsolutePosition(node: Node<NodeData>, byId: Map<string, Node<NodeData>>): { x: number; y: number } {
  if (!node.parentNode) return node.position;
  const parent = byId.get(node.parentNode);
  if (!parent) return node.position;
  const parentAbs = getAbsolutePosition(parent, byId);
  return { x: parentAbs.x + node.position.x, y: parentAbs.y + node.position.y };
}

function findContainerAtPoint(nodes: Node<NodeData>[], point: { x: number; y: number }, excludeId?: string): Node<NodeData> | undefined {
  const byId = new Map(nodes.map((node) => [node.id, node]));
  return nodes
    .filter((node) => node.type === "containerNode" && node.id !== excludeId)
    .find((container) => {
      const abs = getAbsolutePosition(container, byId);
      const dims = getNodeSize(container);
      return point.x >= abs.x && point.x <= abs.x + dims.width && point.y >= abs.y && point.y <= abs.y + dims.height;
    });
}

function attachNodeCallbacks(
  nodes: Node<NodeData>[],
  onStartEdit: (nodeId: string) => void,
  onCommitLabel: (nodeId: string, label: string) => void
): Node<NodeData>[] {
  return nodes.map((node) => ({
    ...node,
    data: {
      ...node.data,
      onStartEdit,
      onCommitLabel,
    },
  }));
}

function buildNodeFromTemplate(template: EditorNodeType, id: string, position: { x: number; y: number }): Node<NodeData> {
  const kind = templateToKind(template);
  if (kind === "container") {
    return {
      id,
      type: "containerNode",
      data: { label: id, kind },
      position,
      style: { width: CONTAINER_WIDTH, height: CONTAINER_HEIGHT },
      draggable: true,
      selectable: true,
    };
  }

  return {
    id,
    type: toFlowNodeType(kind),
    data: { label: id, kind },
    position,
    draggable: true,
    selectable: true,
  };
}

function isAllowedHierarchyEdge(source: Node<NodeData> | undefined, target: Node<NodeData> | undefined): boolean {
  if (!source || !target) return false;
  const sourceCategory = toCanonicalCategory(source.data.kind);
  const targetCategory = toCanonicalCategory(target.data.kind);
  if (sourceCategory === "ui" && targetCategory === "service") return true;
  if (sourceCategory === "service" && (targetCategory === "database" || targetCategory === "cache")) return true;
  return false;
}

function createEdge(
  id: string,
  source: string,
  target: string,
  edgeType: EdgeProtocol,
  lineStyle: EdgeLine
): Edge<EdgeData> {
  const visual = getProtocolVisual(edgeType, lineStyle);
  return {
    id,
    source,
    target,
    type: "smoothstep",
    markerEnd: { type: MarkerType.ArrowClosed, width: 18, height: 18 },
    data: { edgeType, lineStyle },
    label: edgeType,
    style: visual.style,
    labelStyle: visual.labelStyle,
    labelBgStyle: visual.labelBgStyle,
    labelBgBorderRadius: 4,
    labelBgPadding: [4, 3],
    animated: false,
  };
}

function dedupeEdges(edges: Edge<EdgeData>[]): Edge<EdgeData>[] {
  const seen = new Set<string>();
  const out: Edge<EdgeData>[] = [];
  edges.forEach((edge) => {
    const key = `${edge.source}->${edge.target}`;
    if (edge.source === edge.target || seen.has(key)) return;
    seen.add(key);
    const edgeType = edge.data?.edgeType ?? "request";
    const lineStyle = edge.data?.lineStyle ?? (edgeType === "Queue" ? "async" : "sync");
    out.push(createEdge(edge.id, edge.source, edge.target, edgeType, lineStyle));
  });
  return out;
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
      style: kind === "container" ? { width: CONTAINER_WIDTH, height: CONTAINER_HEIGHT } : undefined,
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

function buildHierarchyEdges(nodes: Node<NodeData>[]): Edge<EdgeData>[] {
  const ui = nodes.filter((node) => toCanonicalCategory(node.data.kind) === "ui");
  const services = nodes.filter((node) => toCanonicalCategory(node.data.kind) === "service");
  const databases = nodes.filter((node) => toCanonicalCategory(node.data.kind) === "database");
  const caches = nodes.filter((node) => toCanonicalCategory(node.data.kind) === "cache");
  const queues = nodes.filter((node) => node.data.kind === "queue");

  const edges: Edge<EdgeData>[] = [];
  let id = 1;

  ui.forEach((src) => {
    services.forEach((dst) => edges.push(createEdge(`e${id++}`, src.id, dst.id, "HTTP", "sync")));
  });
  services.forEach((src) => {
    databases.forEach((dst) => edges.push(createEdge(`e${id++}`, src.id, dst.id, "DB Query", "sync")));
  });
  services.forEach((src) => {
    queues.forEach((dst) => edges.push(createEdge(`e${id++}`, src.id, dst.id, "Async", "async")));
  });
  services.forEach((src) => {
    caches.forEach((dst) => edges.push(createEdge(`e${id++}`, src.id, dst.id, "Cache", "sync")));
  });

  return dedupeEdges(edges);
}

function buildEdgesFromArchitecture(nodes: Node<NodeData>[], architecture: Architecture): Edge<EdgeData>[] {
  const byId = new Map(nodes.map((node) => [node.id, node]));

  return dedupeEdges(
    architecture.edges
    .filter((edge) => typeof edge.source === "string" && typeof edge.target === "string")
    .map((edge, index) => {
      const sourceNode = byId.get(edge.source);
      const targetNode = byId.get(edge.target);
      if (!sourceNode || !targetNode) return null;

      const label = typeof edge.label === "string" ? edge.label : undefined;
      const edgeType = normalizeProtocol(label ?? protocolFromKinds(sourceNode, targetNode));
      const lineStyle: EdgeLine = edgeType === "Async" ? "async" : "sync";
      return createEdge(`e${index + 1}`, edge.source, edge.target, edgeType, lineStyle);
    })
    .filter((edge): edge is Edge<EdgeData> => edge !== null)
  );
}

async function applyDagreLayout(nodes: Node<NodeData>[], edges: Edge<EdgeData>[]): Promise<Node<NodeData>[]> {
  try {
    const dagreModule = await import(/* @vite-ignore */ "https://esm.sh/dagre@0.8.5");
    const dagre = dagreModule.default as {
      graphlib: { Graph: new () => { setGraph: (g: object) => void; setDefaultEdgeLabel: (fn: () => object) => void; setNode: (id: string, data: object) => void; setEdge: (s: string, t: string) => void; node: (id: string) => { x: number; y: number } } };
      layout: (g: unknown) => void;
    };

    const graph = new dagre.graphlib.Graph();
    graph.setGraph({ rankdir: "TB", ranksep: 120, nodesep: 80, marginx: 24, marginy: 24 });
    graph.setDefaultEdgeLabel(() => ({}));

    nodes.forEach((node) => {
      const size = getNodeSize(node);
      graph.setNode(node.id, { width: size.width, height: size.height });
    });
    edges.forEach((edge) => graph.setEdge(edge.source, edge.target));
    dagre.layout(graph);

    return nodes.map((node) => {
      if (node.parentNode) return node;
      const placed = graph.node(node.id);
      const size = getNodeSize(node);
      return {
        ...node,
        position: { x: placed.x - size.width / 2, y: placed.y - size.height / 2 },
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

function cloneGraphState(state: GraphState): GraphState {
  return {
    nodes: state.nodes.map((node) => ({
      ...node,
      data: { ...node.data },
      position: { ...node.position },
      style: node.style ? { ...node.style } : undefined,
    })),
    edges: state.edges.map((edge) => ({
      ...edge,
      data: edge.data ? { ...edge.data } : undefined,
      style: edge.style ? { ...edge.style } : undefined,
      labelStyle: edge.labelStyle ? { ...edge.labelStyle } : undefined,
      labelBgStyle: edge.labelBgStyle ? { ...edge.labelBgStyle } : undefined,
      markerEnd: edge.markerEnd,
    })),
  };
}

function TechnologyIcon({ label }: { label: string }) {
  const key = getIconKey(label);
  if (!key) return <FiBox className="arch-node-icon" size={16} />;

  const icon = simpleIconMap[key];
  return (
    <svg className="arch-node-logo" viewBox="0 0 24 24" role="img" aria-label={label}>
      <path d={icon.path} fill={`#${icon.hex}`} />
    </svg>
  );
}

function NodeShell({ id, data, selected }: NodeProps<NodeData>) {
  const [draft, setDraft] = useState(data.label);
  useEffect(() => setDraft(data.label), [data.label]);

  const commit = () => {
    data.onCommitLabel?.(id, draft);
  };

  return (
    <div className={`arch-node ${nodeThemeClass(data.kind)} ${selected ? "selected" : ""}`} onDoubleClick={() => data.onStartEdit?.(id)}>
      <Handle type="target" position={Position.Top} />
      <TechnologyIcon label={data.label} />
      {data.editing ? (
        <input
          className="arch-node-input nodrag nowheel"
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          onBlur={commit}
          onKeyDown={(event: ReactKeyboardEvent<HTMLInputElement>) => {
            if (event.key === "Enter") {
              commit();
            }
          }}
          autoFocus
        />
      ) : (
        <div className="arch-node-label">{data.label}</div>
      )}
      <Handle type="source" position={Position.Bottom} />
    </div>
  );
}

function ContainerNode({ id, data, selected }: NodeProps<NodeData>) {
  const [draft, setDraft] = useState(data.label);
  useEffect(() => setDraft(data.label), [data.label]);

  const commit = () => data.onCommitLabel?.(id, draft);

  return (
    <div className={`arch-container ${selected ? "selected" : ""}`} onDoubleClick={() => data.onStartEdit?.(id)}>
      <div className="arch-container-header">
        <TechnologyIcon label={data.label} />
        {data.editing ? (
          <input
            className="arch-node-input nodrag nowheel"
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
            onBlur={commit}
            onKeyDown={(event: ReactKeyboardEvent<HTMLInputElement>) => {
              if (event.key === "Enter") commit();
            }}
            autoFocus
          />
        ) : (
          <span className="arch-container-title">{data.label}</span>
        )}
      </div>
    </div>
  );
}

const nodeTypes = {
  uiNode: NodeShell,
  serviceNode: NodeShell,
  dataNode: NodeShell,
  cacheNode: NodeShell,
  queueNode: NodeShell,
  containerNode: ContainerNode,
};

function DiagramViewInner({ architecture, command, theme, onToggleTheme }: DiagramViewProps) {
  const reactFlow = useReactFlow<NodeData, EdgeData>();
  const wrapperRef = useRef<HTMLDivElement | null>(null);
  const importInputRef = useRef<HTMLInputElement | null>(null);

  const initialGraph = useMemo(() => buildInitialGraph(architecture), [architecture]);
  const [graph, setGraph] = useState<GraphState>(initialGraph);
  const [editingNodeId, setEditingNodeId] = useState<string | null>(null);
  const [toolMode, setToolMode] = useState<ToolMode>("select");
  const historyRef = useRef<GraphState[]>([]);
  const futureRef = useRef<GraphState[]>([]);
  const graphRef = useRef(graph);
  const nodeCounterRef = useRef(1);
  const hasInitializedRef = useRef(false);

  useEffect(() => {
    graphRef.current = graph;
  }, [graph]);

  const onStartEdit = useCallback((nodeId: string) => {
    setEditingNodeId(nodeId);
  }, []);

  const onCommitLabel = useCallback((nodeId: string, label: string) => {
    const clean = label.trim() || nodeId;
    setEditingNodeId(null);
    setGraph((current) => {
      historyRef.current.push(cloneGraphState(current));
      if (historyRef.current.length > 50) historyRef.current.shift();
      futureRef.current = [];

      return {
        ...current,
        nodes: current.nodes.map((node) => {
          if (node.id !== nodeId) return node;
          const kind = detectKindFromLabel(clean, node.type);
          return {
            ...node,
            type: toFlowNodeType(kind),
            data: {
              ...node.data,
              label: clean,
              kind,
              editing: false,
              onStartEdit,
              onCommitLabel,
            },
            style: kind === "container" ? { width: CONTAINER_WIDTH, height: CONTAINER_HEIGHT } : node.style,
          };
        }),
      };
    });
  }, [onStartEdit]);

  const withCallbacks = useCallback(
    (state: GraphState): GraphState => ({
      nodes: attachNodeCallbacks(
        state.nodes.map((node) => ({
          ...node,
          data: { ...node.data, editing: node.id === editingNodeId },
        })),
        onStartEdit,
        onCommitLabel
      ),
      edges: state.edges,
    }),
    [editingNodeId, onCommitLabel, onStartEdit]
  );

  const applyGraphChange = useCallback(
    (updater: (current: GraphState) => GraphState, options?: { recordHistory?: boolean }) => {
      const shouldRecord = options?.recordHistory ?? true;
      setGraph((current) => {
        if (shouldRecord) {
          historyRef.current.push(cloneGraphState(current));
          if (historyRef.current.length > 50) historyRef.current.shift();
          futureRef.current = [];
        }
        return updater(current);
      });
    },
    []
  );

  const undo = useCallback(() => {
    const prev = historyRef.current.pop();
    if (!prev) return;
    futureRef.current.push(cloneGraphState(graphRef.current));
    setGraph(prev);
    setEditingNodeId(null);
  }, []);

  const redo = useCallback(() => {
    const next = futureRef.current.pop();
    if (!next) return;
    historyRef.current.push(cloneGraphState(graphRef.current));
    setGraph(next);
    setEditingNodeId(null);
  }, []);

  const applyAutoLayout = useCallback(async (state: GraphState) => {
    const laidOut = await applyDagreLayout(state.nodes, state.edges);
    setGraph((current) => {
      if (current !== state) return current;
      return { ...current, nodes: laidOut };
    });
  }, []);

  useEffect(() => {
    if (hasInitializedRef.current) {
      return;
    }

    hasInitializedRef.current = true;
    historyRef.current = [];
    futureRef.current = [];
    nodeCounterRef.current = 1;
    setGraph(initialGraph);
    setEditingNodeId(null);
    void applyAutoLayout(initialGraph);
  }, [initialGraph, applyAutoLayout]);

  useEffect(() => {
    if (!command) return;
    if (command.action === "clear") {
      applyGraphChange(() => ({ nodes: [], edges: [] }));
      return;
    }
    if (command.action === "reset") {
      const next = buildInitialGraph(architecture);
      // Keep this as a history-recorded change so prompt-generated resets are undoable.
      applyGraphChange(() => next);
      void applyAutoLayout(next);
    }
  }, [architecture, command, applyAutoLayout, applyGraphChange]);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      reactFlow.fitView({ padding: 0.2, duration: 250 });
    }, 80);
    return () => window.clearTimeout(timer);
  }, [graph.nodes, graph.edges, reactFlow]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      const tag = target?.tagName?.toLowerCase();
      const editingText = !!target && (target.isContentEditable || tag === "input" || tag === "textarea");
      if (editingText) return;

      const key = event.key.toLowerCase();
      const ctrlMeta = event.ctrlKey || event.metaKey;

      if (ctrlMeta && key === "z" && !event.shiftKey) {
        event.preventDefault();
        undo();
        return;
      }
      if (ctrlMeta && (key === "y" || (key === "z" && event.shiftKey))) {
        event.preventDefault();
        redo();
        return;
      }
      if (key === "delete" || key === "backspace") {
        event.preventDefault();
        applyGraphChange((current) => {
          const selectedNodeIds = new Set(current.nodes.filter((n) => n.selected).map((n) => n.id));
          const nodes = current.nodes.filter((n) => !selectedNodeIds.has(n.id));
          const edges = current.edges.filter((e) => !e.selected && !selectedNodeIds.has(e.source) && !selectedNodeIds.has(e.target));
          return { nodes, edges };
        });
      }
    };

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [applyGraphChange, redo, undo]);

  const onNodesChange = useCallback(
    (changes: NodeChange[]) => {
      const hasMeaningfulChange = changes.some((change) => {
        if (change.type === "select" || change.type === "dimensions") return false;

        // Drag emits many transient position updates; history is committed on drag stop.
        if (change.type === "position") {
          const dragging = "dragging" in change ? Boolean(change.dragging) : false;
          return !dragging;
        }

        return true;
      });
      applyGraphChange((current) => ({
        ...current,
        nodes: applyNodeChanges(changes, current.nodes),
      }), { recordHistory: hasMeaningfulChange });
    },
    [applyGraphChange]
  );

  const onEdgesChange = useCallback(
    (changes: EdgeChange[]) => {
      const hasMeaningfulChange = changes.some((change) => change.type !== "select");
      applyGraphChange((current) => ({
        ...current,
        edges: applyEdgeChanges(changes, current.edges),
      }), { recordHistory: hasMeaningfulChange });
    },
    [applyGraphChange]
  );

  const onConnect = useCallback(
    (connection: Connection) => {
      applyGraphChange((current) => {
        if (!connection.source || !connection.target) return current;
        const sourceNode = current.nodes.find((n) => n.id === connection.source);
        const targetNode = current.nodes.find((n) => n.id === connection.target);
        if (!isAllowedHierarchyEdge(sourceNode, targetNode)) return current;

        const exists = current.edges.some((e) => e.source === connection.source && e.target === connection.target);
        if (exists) return current;

        const suggested = protocolFromKinds(sourceNode, targetNode);
        const chosen = normalizeProtocol(window.prompt("Edge type: HTTP, gRPC, Queue, DB Query, request", suggested));
        const lineStyle: EdgeLine = chosen === "Queue" ? "async" : "sync";

        const edge = createEdge(`e${Date.now()}`, connection.source, connection.target, chosen, lineStyle);
        return { ...current, edges: [...current.edges, edge] };
      });
    },
    [applyGraphChange]
  );

  const onDragOver = useCallback((event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = "move";
  }, []);

  const onDrop = useCallback(
    (event: DragEvent<HTMLDivElement>) => {
      event.preventDefault();
      const raw = event.dataTransfer.getData("application/x-arch-node") as EditorNodeType;
      if (!raw) return;

      const position = reactFlow.screenToFlowPosition({ x: event.clientX, y: event.clientY });
      const id = `${raw}-${Date.now()}`;
      const node = buildNodeFromTemplate(raw, id, position);

      applyGraphChange((current) => {
        const container = raw !== "container" ? findContainerAtPoint(current.nodes, position) : undefined;
        if (container) {
          const byId = new Map(current.nodes.map((n) => [n.id, n]));
          const containerAbs = getAbsolutePosition(container, byId);
          node.parentNode = container.id;
          node.extent = "parent";
          node.position = { x: position.x - containerAbs.x - 20, y: position.y - containerAbs.y - 28 };
        }
        return { ...current, nodes: [...current.nodes, node] };
      });
    },
    [applyGraphChange, reactFlow]
  );

  const onNodeDragStop = useCallback(
    (_: React.MouseEvent, draggedNode: Node<NodeData>) => {
      if (draggedNode.type === "containerNode") return;

      applyGraphChange((current) => {
        const byId = new Map(current.nodes.map((n) => [n.id, n]));
        const currentNode = byId.get(draggedNode.id);
        if (!currentNode) return current;

        const abs = (draggedNode.positionAbsolute ?? draggedNode.position) as { x: number; y: number };
        const container = findContainerAtPoint(current.nodes, abs, draggedNode.id);

        const updatedNodes = current.nodes.map((node) => {
          if (node.id !== draggedNode.id) return node;

          if (container) {
            const containerAbs = getAbsolutePosition(container, byId);
            return {
              ...node,
              parentNode: container.id,
              extent: "parent" as const,
              position: {
                x: abs.x - containerAbs.x,
                y: abs.y - containerAbs.y,
              },
            };
          }

          if (node.parentNode) {
            return {
              ...node,
              parentNode: undefined,
              extent: undefined,
              position: abs,
            };
          }

          return {
            ...node,
            position: {
              x: abs.x,
              y: abs.y,
            },
          };
        });

        return { ...current, nodes: updatedNodes };
      });
    },
    [applyGraphChange]
  );

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

  const onImportFileChange = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      const text = await file.text();
      const parsed = JSON.parse(text) as { nodes?: unknown; edges?: unknown };
      if (!Array.isArray(parsed.nodes) || !Array.isArray(parsed.edges)) return;

      const nodes = parsed.nodes
        .map((item, index) => {
          if (!item || typeof item !== "object") return null;
          const node = item as { id?: unknown; type?: unknown; data?: { label?: unknown }; position?: { x?: unknown; y?: unknown } };
          if (typeof node.id !== "string") return null;
          const label = typeof node.data?.label === "string" ? node.data.label : node.id;
          const kind = detectKindFromLabel(label, typeof node.type === "string" ? node.type : undefined);
          const built = buildNodeFromTemplate(kind === "database" ? "data" : (kind as EditorNodeType), node.id, {
            x: typeof node.position?.x === "number" ? node.position.x : index * 40,
            y: typeof node.position?.y === "number" ? node.position.y : layerY[toLayer(kind)],
          });
          built.data = { ...built.data, label, kind };
          return built;
        })
        .filter((n): n is Node<NodeData> => n !== null);

      const nodeSet = new Set(nodes.map((n) => n.id));
      const edges = dedupeEdges(
        parsed.edges
          .map((item, index) => {
            if (!item || typeof item !== "object") return null;
            const edge = item as { source?: unknown; target?: unknown; data?: { edgeType?: unknown; lineStyle?: unknown } };
            if (typeof edge.source !== "string" || typeof edge.target !== "string") return null;
            if (!nodeSet.has(edge.source) || !nodeSet.has(edge.target)) return null;
            const edgeType = normalizeProtocol(typeof edge.data?.edgeType === "string" ? edge.data.edgeType : "request");
            const lineStyle: EdgeLine = edgeType === "Queue" ? "async" : "sync";
            return createEdge(`i${index + 1}`, edge.source, edge.target, edgeType, lineStyle);
          })
          .filter((e): e is Edge<EdgeData> => e !== null)
      );

      const next = { nodes, edges };
      applyGraphChange(() => next);
      void applyAutoLayout(next);
    } catch {
      // Ignore malformed imports.
    } finally {
      event.target.value = "";
    }
  };

  const rendered = withCallbacks(graph);

  return (
    <div className="diagram-editor">
      <div className="diagram-workspace" ref={wrapperRef} onDrop={onDrop} onDragOver={onDragOver}>
        <div className="diagram-menubar">
          <div className="menu-group">
            <span className="menu-title">File</span>
            <button type="button" className="menu-btn" onClick={onExportJson}>Export JSON</button>
            <button type="button" className="menu-btn" onClick={() => importInputRef.current?.click()}>Import JSON</button>
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
          nodes={rendered.nodes}
          edges={rendered.edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          onNodeDragStop={onNodeDragStop}
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
