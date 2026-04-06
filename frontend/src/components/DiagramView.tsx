import { ChangeEvent, DragEvent, KeyboardEvent as ReactKeyboardEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { toPng } from "html-to-image";
import { FiMoon, FiMousePointer, FiPlusCircle, FiSun } from "react-icons/fi";
import { FaAws } from "react-icons/fa";
import { Box, Database, GitBranch, Monitor, Server, Zap } from "lucide-react";
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
  useEdgesState,
  useNodesState,
  useReactFlow,
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
type ToolMode = "select" | "connect";

type NodeData = {
  label: string;
  kind: FlowNodeKind;
  type: string;
  icon?: string;
  style?: {
    background?: string;
    borderColor?: string;
    textColor?: string;
  };
  editing?: boolean;
  onStartEdit?: (nodeId: string) => void;
  onCommitLabel?: (nodeId: string, label: string) => void;
  onCancelEdit?: () => void;
};

type EdgeData = {
  label?: string;
  edgeType: EdgeProtocol;
  lineStyle: EdgeLine;
  style?: {
    stroke?: string;
    width?: number;
    dashed?: boolean;
  };
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

const ICON_OPTIONS = ["auto", "postgres", "redis", "kafka", "docker", "nginx", "react", "node", "aws", "generic"] as const;
type IconOption = (typeof ICON_OPTIONS)[number];

const ICON_MAP: Record<string, string> = {
  postgres: "postgresql",
  aws: "amazonaws",
  gcp: "googlecloud",
  azure: "microsoftazure",
  node: "nodedotjs",
  react: "react",
  docker: "docker",
  redis: "redis",
  kafka: "apachekafka",
  nginx: "nginx",
};

function normalizeLabel(value: string): string {
  return value.trim().toLowerCase();
}

function getIconUrl(iconName: string | undefined): string | null {
  if (!iconName || iconName === "auto" || iconName === "generic") return null;
  const slug = ICON_MAP[iconName.toLowerCase()];
  if (!slug) return null;
  return `https://cdn.simpleicons.org/${slug}`;
}

function inferIconFromLabel(label: string): IconOption | null {
  const normalized = normalizeLabel(label);
  if (normalized.includes("postgres")) return "postgres";
  if (normalized.includes("redis")) return "redis";
  if (normalized.includes("kafka") || normalized.includes("queue") || normalized.includes("broker")) return "kafka";
  if (normalized.includes("docker") || normalized.includes("container")) return "docker";
  if (normalized.includes("nginx") || normalized.includes("gateway") || normalized.includes("proxy")) return "nginx";
  if (normalized.includes("react") || normalized.includes("frontend") || normalized.includes("ui")) return "react";
  if (normalized.includes("node") || normalized.includes("api") || normalized.includes("service")) return "node";
  if (normalized.includes("aws")) return "aws";
  return null;
}

function detectKindFromLabel(label: string, fallbackType?: string): FlowNodeKind {
  const normalized = normalizeLabel(label);
  const normalizedType = normalizeLabel(fallbackType ?? "");
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

  if (
    normalizedType === "ui" ||
    normalizedType.includes("frontend") ||
    normalizedType.includes("client") ||
    normalizedType.includes("web")
  ) {
    return "ui";
  }
  if (
    normalizedType === "database" ||
    normalizedType === "data" ||
    normalizedType.includes("db") ||
    normalizedType.includes("database") ||
    normalizedType.includes("postgres") ||
    normalizedType.includes("mysql") ||
    normalizedType.includes("mongo")
  ) {
    return "database";
  }
  if (normalizedType.includes("cache") || normalizedType.includes("redis") || normalizedType.includes("memcached")) {
    return "cache";
  }
  if (
    normalizedType.includes("queue") ||
    normalizedType.includes("broker") ||
    normalizedType.includes("kafka") ||
    normalizedType.includes("rabbit") ||
    normalizedType.includes("sqs")
  ) {
    return "queue";
  }
  if (
    normalizedType.includes("gateway") ||
    normalizedType.includes("proxy") ||
    normalizedType.includes("ingress") ||
    normalizedType.includes("nginx")
  ) {
    return "gateway";
  }
  if (
    normalizedType.includes("container") ||
    normalizedType.includes("docker") ||
    normalizedType.includes("k8s") ||
    normalizedType.includes("kubernetes") ||
    normalizedType.includes("pod")
  ) {
    return "container";
  }
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

function getProtocolVisual(_edgeType: EdgeProtocol, lineStyle: EdgeLine): {
  style: React.CSSProperties;
  labelStyle: React.CSSProperties;
  labelBgStyle: React.CSSProperties;
} {
  const baseColor = "var(--edge-color)";
  return {
    style: {
      stroke: baseColor,
      strokeWidth: 2,
      strokeDasharray: lineStyle === "async" ? "6 4" : undefined,
    },
    labelStyle: {
      fontSize: 11,
      fill: "var(--edge-label-text)",
      fontWeight: 600,
    },
    labelBgStyle: {
      fill: "var(--edge-label-bg)",
      fillOpacity: 0.92,
      stroke: "var(--edge-label-border)",
      strokeWidth: 1,
    },
  };
}

function buildEdgeStyle(style: EdgeData["style"] | undefined, lineStyle: EdgeLine): React.CSSProperties {
  return {
    stroke: style?.stroke || "var(--edge-color)",
    strokeWidth: style?.width ?? 2,
    strokeDasharray: style?.dashed || lineStyle === "async" ? "5 5" : undefined,
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
  onCommitLabel: (nodeId: string, label: string) => void,
  onCancelEdit: () => void
): Node<NodeData>[] {
  return nodes.map((node) => ({
    ...node,
    data: {
      ...node.data,
      onStartEdit,
      onCommitLabel,
      onCancelEdit,
    },
  }));
}

function buildNodeFromTemplate(template: EditorNodeType, id: string, position: { x: number; y: number }): Node<NodeData> {
  const kind = templateToKind(template);
  if (kind === "container") {
    return {
      id,
      type: "containerNode",
      data: { label: id, kind, type: kind, style: {} },
      position,
      style: { width: CONTAINER_WIDTH, height: CONTAINER_HEIGHT },
      draggable: true,
      selectable: true,
    };
  }

  return {
    id,
    type: toFlowNodeType(kind),
    data: { label: id, kind, type: kind, style: {} },
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
  lineStyle: EdgeLine,
  customStyle?: EdgeData["style"]
): Edge<EdgeData> {
  const visual = getProtocolVisual(edgeType, lineStyle);
  return {
    id,
    source,
    target,
    type: "smoothstep",
    className: "arch-edge",
    markerEnd: { type: MarkerType.ArrowClosed, width: 18, height: 18, color: "var(--edge-color)" },
    data: {
      edgeType,
      lineStyle,
      label: edgeType,
      style: customStyle,
    },
    label: edgeType,
    style: { ...visual.style, ...buildEdgeStyle(customStyle, lineStyle) },
    labelStyle: visual.labelStyle,
    labelBgStyle: visual.labelBgStyle,
    labelBgBorderRadius: 4,
    labelBgPadding: [8, 4],
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
    const lineStyle = edge.data?.lineStyle ?? (edgeType === "Async" ? "async" : "sync");
    out.push(createEdge(edge.id, edge.source, edge.target, edgeType, lineStyle, edge.data?.style));
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
      data: { label: node.id, kind, type: kind, style: {} },
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

function FallbackIcon({ type }: { type: string }) {
  const normalizedType = normalizeLabel(type);
  if (normalizedType === "ui") return <Monitor className="arch-node-icon" size={16} />;
  if (normalizedType === "database" || normalizedType === "db") return <Database className="arch-node-icon" size={16} />;
  if (normalizedType === "queue") return <GitBranch className="arch-node-icon" size={16} />;
  if (normalizedType === "cache") return <Zap className="arch-node-icon" size={16} />;
  if (normalizedType === "container") return <Box className="arch-node-icon" size={16} />;
  return <Server className="arch-node-icon" size={16} />;
}

function TechnologyIcon({ label, kind, type, icon }: { label: string; kind: FlowNodeKind; type: string; icon?: string }) {
  const [hasImageError, setHasImageError] = useState(false);
  const hasExplicitIcon = Boolean(icon && icon !== "auto");
  const explicitIconName = hasExplicitIcon ? normalizeLabel(icon ?? "") : null;
  const explicitUrl = explicitIconName ? getIconUrl(explicitIconName) : null;
  const inferred = hasExplicitIcon ? null : inferIconFromLabel(label);
  const resolvedIconName = explicitIconName || inferred || undefined;
  const iconUrl = explicitUrl || getIconUrl(inferred ?? undefined);

  useEffect(() => {
    setHasImageError(false);
  }, [icon, label]);

  if (resolvedIconName === "aws") {
    return <FaAws className="arch-node-icon" size={16} />;
  }

  if (iconUrl && !hasImageError) {
    return (
      <img
        className="arch-node-logo"
        src={iconUrl}
        alt={label}
        width={16}
        height={16}
        loading="lazy"
        decoding="async"
        onError={(event) => {
          event.currentTarget.style.display = "none";
          setHasImageError(true);
        }}
      />
    );
  }

  return <FallbackIcon type={type || kind} />;
}

function nodeInlineStyle(data: NodeData): React.CSSProperties {
  return {
    background: data.style?.background || "var(--node-default-bg)",
    borderColor: data.style?.borderColor || "var(--node-default-border)",
    color: data.style?.textColor || "var(--node-default-text)",
  };
}

function NodeShell({ id, data, selected }: NodeProps<NodeData>) {
  const [draft, setDraft] = useState(data.label);
  const skipBlurCommitRef = useRef(false);
  useEffect(() => setDraft(data.label), [data.label]);

  const commit = () => {
    data.onCommitLabel?.(id, draft);
  };

  return (
    <div className={`arch-node ${selected ? "selected" : ""}`} style={nodeInlineStyle(data)} onDoubleClick={() => data.onStartEdit?.(id)}>
      <Handle type="target" position={Position.Top} />
      <TechnologyIcon label={data.label} kind={data.kind} type={data.type} icon={data.icon} />
      {data.editing ? (
        <input
          className="arch-node-input nodrag nowheel"
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          onBlur={() => {
            if (skipBlurCommitRef.current) {
              skipBlurCommitRef.current = false;
              return;
            }
            commit();
          }}
          onKeyDown={(event: ReactKeyboardEvent<HTMLInputElement>) => {
            if (event.key === "Enter") {
              commit();
              return;
            }
            if (event.key === "Escape") {
              event.preventDefault();
              skipBlurCommitRef.current = true;
              setDraft(data.label);
              data.onCancelEdit?.();
              (event.currentTarget as HTMLInputElement).blur();
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
  const skipBlurCommitRef = useRef(false);
  useEffect(() => setDraft(data.label), [data.label]);

  const commit = () => data.onCommitLabel?.(id, draft);

  return (
    <div className={`arch-container ${selected ? "selected" : ""}`} style={nodeInlineStyle(data)} onDoubleClick={() => data.onStartEdit?.(id)}>
      <div className="arch-container-header">
        <TechnologyIcon label={data.label} kind={data.kind} type={data.type} icon={data.icon} />
        {data.editing ? (
          <input
            className="arch-node-input nodrag nowheel"
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
            onBlur={() => {
              if (skipBlurCommitRef.current) {
                skipBlurCommitRef.current = false;
                return;
              }
              commit();
            }}
            onKeyDown={(event: ReactKeyboardEvent<HTMLInputElement>) => {
              if (event.key === "Enter") {
                commit();
                return;
              }
              if (event.key === "Escape") {
                event.preventDefault();
                skipBlurCommitRef.current = true;
                setDraft(data.label);
                data.onCancelEdit?.();
                (event.currentTarget as HTMLInputElement).blur();
              }
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
  const [nodes, setNodes] = useNodesState<NodeData>(initialGraph.nodes);
  const [edges, setEdges] = useEdgesState<EdgeData>(initialGraph.edges);
  const [editingNodeId, setEditingNodeId] = useState<string | null>(null);
  const [toolMode, setToolMode] = useState<ToolMode>("select");
  const [edgeEditor, setEdgeEditor] = useState<{ edgeId: string; x: number; y: number; value: string } | null>(null);
  const [isDraggingNode, setIsDraggingNode] = useState(false);
  const historyRef = useRef<GraphState[]>([]);
  const futureRef = useRef<GraphState[]>([]);
  const graphRef = useRef<GraphState>(initialGraph);
  const hasInitializedRef = useRef(false);

  useEffect(() => {
    graphRef.current = { nodes, edges };
  }, [nodes, edges]);

  const onStartEdit = useCallback((nodeId: string) => {
    setEditingNodeId(nodeId);
  }, []);

  const onCancelEdit = useCallback(() => {
    setEditingNodeId(null);
  }, []);

  const onCommitLabel = useCallback((nodeId: string, label: string) => {
    const clean = label.trim() || nodeId;
    setEditingNodeId(null);
    const current = graphRef.current;
    historyRef.current.push(cloneGraphState(current));
    if (historyRef.current.length > 50) historyRef.current.shift();
    futureRef.current = [];

    const next: GraphState = {
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
            type: kind,
            style: node.data.style,
            editing: false,
            onStartEdit,
            onCommitLabel,
          },
          style: kind === "container" ? { width: CONTAINER_WIDTH, height: CONTAINER_HEIGHT } : node.style,
        };
      }),
    };

    graphRef.current = next;
    setNodes(next.nodes);
    setEdges(next.edges);
  }, [onStartEdit, setEdges, setNodes]);

  const withCallbacks = useCallback(
    (state: GraphState): GraphState => ({
      nodes: attachNodeCallbacks(
        state.nodes.map((node) => ({
          ...node,
          data: { ...node.data, editing: node.id === editingNodeId },
        })),
        onStartEdit,
        onCommitLabel,
        onCancelEdit
      ),
      edges: state.edges,
    }),
    [editingNodeId, onCancelEdit, onCommitLabel, onStartEdit]
  );

  const applyGraphChange = useCallback(
    (updater: (current: GraphState) => GraphState, options?: { recordHistory?: boolean }) => {
      const shouldRecord = options?.recordHistory ?? true;
      const current = graphRef.current;
      if (shouldRecord) {
        historyRef.current.push(cloneGraphState(current));
        if (historyRef.current.length > 50) historyRef.current.shift();
        futureRef.current = [];
      }
      const next = updater(current);
      graphRef.current = next;
      setNodes(next.nodes);
      setEdges(next.edges);
    },
    [setEdges, setNodes]
  );

  const undo = useCallback(() => {
    const prev = historyRef.current.pop();
    if (!prev) return;
    futureRef.current.push(cloneGraphState(graphRef.current));
    graphRef.current = prev;
    setNodes(prev.nodes);
    setEdges(prev.edges);
    setEditingNodeId(null);
  }, [setEdges, setNodes]);

  const redo = useCallback(() => {
    const next = futureRef.current.pop();
    if (!next) return;
    historyRef.current.push(cloneGraphState(graphRef.current));
    graphRef.current = next;
    setNodes(next.nodes);
    setEdges(next.edges);
    setEditingNodeId(null);
  }, [setEdges, setNodes]);

  const applyAutoLayout = useCallback(async (state: GraphState) => {
    const laidOut = await applyDagreLayout(state.nodes, state.edges);
    const next = { ...state, nodes: laidOut };
    graphRef.current = next;
    setNodes(next.nodes);
    setEdges(next.edges);
  }, [setEdges, setNodes]);

  useEffect(() => {
    if (hasInitializedRef.current) {
      return;
    }

    hasInitializedRef.current = true;
    historyRef.current = [];
    futureRef.current = [];
    graphRef.current = initialGraph;
    setNodes(initialGraph.nodes);
    setEdges(initialGraph.edges);
    setEditingNodeId(null);
    void applyAutoLayout(initialGraph);
  }, [initialGraph, applyAutoLayout, setEdges, setNodes]);

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
    if (nodes.length === 0) return;
    const timer = window.setTimeout(() => {
      reactFlow.fitView({ padding: 0.2, duration: 250 });
    }, 50);
    return () => window.clearTimeout(timer);
  }, [nodes, reactFlow]);

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

      if (ctrlMeta && key === "a") {
        event.preventDefault();
        applyGraphChange((current) => ({
          ...current,
          nodes: current.nodes.map((node) => ({ ...node, selected: true })),
        }), { recordHistory: false });
      }

      if (ctrlMeta && key === "d") {
        event.preventDefault();
        applyGraphChange((current) => {
          const selected = current.nodes.filter((node) => node.selected);
          if (selected.length === 0) return current;

          const clones = selected.map((node, index) => ({
            ...node,
            id: `${node.id}-copy-${Date.now()}-${index}`,
            position: { x: node.position.x + 40, y: node.position.y + 40 },
            selected: false,
            data: { ...node.data },
          }));

          return { ...current, nodes: [...current.nodes, ...clones] };
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
      if (toolMode !== "connect") return;
      applyGraphChange((current) => {
        if (!connection.source || !connection.target) return current;
        const sourceNode = current.nodes.find((n) => n.id === connection.source);
        const targetNode = current.nodes.find((n) => n.id === connection.target);
        if (!isAllowedHierarchyEdge(sourceNode, targetNode)) return current;

        const exists = current.edges.some((e) => e.source === connection.source && e.target === connection.target);
        if (exists) return current;

        const chosen = protocolFromKinds(sourceNode, targetNode);
        const lineStyle: EdgeLine = chosen === "Async" ? "async" : "sync";

        const edge = createEdge(`e${Date.now()}`, connection.source, connection.target, chosen, lineStyle);
        return { ...current, edges: [...current.edges, edge] };
      });
    },
    [applyGraphChange, toolMode]
  );

  const onEdgeClick = useCallback(
    (event: React.MouseEvent, edge: Edge<EdgeData>) => {
      const bounds = wrapperRef.current?.getBoundingClientRect();
      const x = bounds ? event.clientX - bounds.left : 16;
      const y = bounds ? event.clientY - bounds.top : 16;
      setEdgeEditor({
        edgeId: edge.id,
        x,
        y,
        value: String(edge.label ?? edge.data?.edgeType ?? "HTTP"),
      });
    },
    [applyGraphChange, toolMode]
  );

  const onEdgeContextMenu = useCallback(
    (event: React.MouseEvent, edge: Edge<EdgeData>) => {
      event.preventDefault();
      applyGraphChange((current) => ({
        ...current,
        edges: current.edges.filter((item) => item.id !== edge.id),
      }));
      setEdgeEditor(null);
    },
    [applyGraphChange]
  );

  const onNodeClick = useCallback(
    () => {
      setEdgeEditor(null);
    },
    []
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
      setIsDraggingNode(false);
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

  const onNodeDragStart = useCallback(() => {
    setIsDraggingNode(true);
  }, []);

  const onExportJson = () => {
    const payload = JSON.stringify({ nodes, edges }, null, 2);
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

  const onExportPng = useCallback(async () => {
    if (!wrapperRef.current) return;
    const viewport = wrapperRef.current.querySelector(".react-flow__viewport") as HTMLElement | null;
    if (!viewport) return;
    try {
      const dataUrl = await toPng(viewport, {
        cacheBust: true,
        backgroundColor: theme === "dark" ? "#0f172a" : "#ffffff",
        filter: (node) => !node.classList?.contains("toolbar"),
        fontEmbedCSS: "",
      });
      const link = document.createElement("a");
      link.href = dataUrl;
      link.download = "architecture.png";
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    } catch {
      // Ignore export errors in UI.
    }
  }, [theme]);

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
          built.data = {
            ...built.data,
            label,
            kind,
            type: typeof (node as { data?: { type?: unknown } }).data?.type === "string"
              ? String((node as { data?: { type?: unknown } }).data?.type)
              : kind,
            icon: typeof (node as { data?: { icon?: unknown } }).data?.icon === "string"
              ? String((node as { data?: { icon?: unknown } }).data?.icon)
              : undefined,
            style: typeof (node as { data?: { style?: unknown } }).data?.style === "object" && (node as { data?: { style?: unknown } }).data?.style !== null
              ? (node as { data?: { style?: { background?: string; borderColor?: string; textColor?: string } } }).data?.style
              : {},
          };
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
            const lineStyle: EdgeLine = edgeType === "Async" ? "async" : "sync";
            const style = typeof (edge as { data?: { style?: unknown } }).data?.style === "object" && (edge as { data?: { style?: unknown } }).data?.style !== null
              ? (edge as { data?: { style?: { stroke?: string; width?: number; dashed?: boolean } } }).data?.style
              : undefined;
            return createEdge(`i${index + 1}`, edge.source, edge.target, edgeType, lineStyle, style);
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

  const selectedNode = nodes.find((node) => node.selected);
  const selectedEdge = edges.find((edge) => edge.selected);
  const defaultNodeStyle = theme === "dark"
    ? { background: "#1e293b", borderColor: "#334155", textColor: "#f8fafc" }
    : { background: "#ffffff", borderColor: "#e2e8f0", textColor: "#0f172a" };

  const updateSelectedNodeLabel = (label: string) => {
    if (!selectedNode) return;
    onCommitLabel(selectedNode.id, label);
  };

  const updateSelectedNodeKind = (kind: FlowNodeKind) => {
    if (!selectedNode) return;
    applyGraphChange((current) => ({
      ...current,
      nodes: current.nodes.map((node) => {
        if (node.id !== selectedNode.id) return node;
        return {
          ...node,
          type: toFlowNodeType(kind),
          data: {
            ...node.data,
            kind,
            type: kind,
          },
          style: kind === "container" ? { width: CONTAINER_WIDTH, height: CONTAINER_HEIGHT } : undefined,
        };
      }),
    }));
  };

  const updateSelectedNodeIcon = (icon: string) => {
    if (!selectedNode) return;
    applyGraphChange((current) => ({
      ...current,
      nodes: current.nodes.map((node) => {
        if (node.id !== selectedNode.id) return node;
        return {
          ...node,
          data: {
            ...node.data,
            icon: icon === "auto" ? undefined : icon,
          },
        };
      }),
    }));
  };

  const updateSelectedNodeStyle = (key: "background" | "borderColor" | "textColor", value: string) => {
    if (!selectedNode) return;
    applyGraphChange((current) => ({
      ...current,
      nodes: current.nodes.map((node) => {
        if (node.id !== selectedNode.id) return node;
        return {
          ...node,
          data: {
            ...node.data,
            style: {
              ...(node.data.style || {}),
              [key]: value,
            },
          },
        };
      }),
    }));
  };

  const updateSelectedEdgeType = (value: string) => {
    if (!selectedEdge) return;
    const edgeType = normalizeProtocol(value);
    const lineStyle: EdgeLine = edgeType === "Async" ? "async" : "sync";
    applyGraphChange((current) => ({
      ...current,
      edges: current.edges.map((edge) =>
        edge.id === selectedEdge.id ? createEdge(edge.id, edge.source, edge.target, edgeType, lineStyle, edge.data?.style) : edge
      ),
    }));
  };

  const updateSelectedEdgeStyle = (key: "stroke" | "width" | "dashed", value: string | number | boolean) => {
    if (!selectedEdge) return;
    applyGraphChange((current) => ({
      ...current,
      edges: current.edges.map((edge) => {
        if (edge.id !== selectedEdge.id) return edge;
        const nextStyle = {
          ...(edge.data?.style || {}),
          [key]: value,
        } as EdgeData["style"];
        const nextType = edge.data?.edgeType ?? "request";
        const nextLineStyle = edge.data?.lineStyle ?? (nextType === "Async" ? "async" : "sync");
        return createEdge(edge.id, edge.source, edge.target, nextType, nextLineStyle, nextStyle);
      }),
    }));
  };

  const commitEdgeEditor = useCallback(() => {
    if (!edgeEditor) return;
    const clean = edgeEditor.value.trim();
    if (!clean) {
      setEdgeEditor(null);
      return;
    }
    const edgeType = normalizeProtocol(clean);
    const lineStyle: EdgeLine = edgeType === "Async" ? "async" : "sync";
    applyGraphChange((current) => ({
      ...current,
      edges: current.edges.map((edge) =>
        edge.id === edgeEditor.edgeId ? createEdge(edge.id, edge.source, edge.target, edgeType, lineStyle, edge.data?.style) : edge
      ),
    }));
    setEdgeEditor(null);
  }, [applyGraphChange, edgeEditor]);

  const deleteEdgeEditorTarget = useCallback(() => {
    if (!edgeEditor) return;
    applyGraphChange((current) => ({
      ...current,
      edges: current.edges.filter((edge) => edge.id !== edgeEditor.edgeId),
    }));
    setEdgeEditor(null);
  }, [applyGraphChange, edgeEditor]);

  const rendered = withCallbacks({ nodes, edges });

  return (
    <div className="diagram-editor">
      <div className="diagram-workspace" ref={wrapperRef} onDrop={onDrop} onDragOver={onDragOver}>
        <div className="diagram-menubar toolbar">
          <div className="menu-group">
            <span className="menu-title">File</span>
            <button type="button" className="menu-btn" onClick={onExportJson}>Export JSON</button>
            <button type="button" className="menu-btn" onClick={() => importInputRef.current?.click()}>Import JSON</button>
            <button type="button" className="menu-btn" onClick={onExportPng}>Export PNG</button>
            <button type="button" className="menu-btn danger" onClick={() => applyGraphChange(() => ({ nodes: [], edges: [] }))}>Clear Diagram</button>
          </div>
          <div className="menu-group">
            <span className="menu-title">Edit</span>
            <button type="button" className="menu-btn" onClick={undo}>Undo</button>
            <button type="button" className="menu-btn" onClick={redo}>Redo</button>
            <button type="button" className="menu-btn" onClick={() => void applyAutoLayout(graphRef.current)}>Auto Layout</button>
            <button type="button" className="menu-btn icon-btn" title="Light / Dark" onClick={onToggleTheme}>{theme === "dark" ? <FiSun /> : <FiMoon />}</button>
          </div>
          <div className="menu-group">
            <button type="button" className={`menu-btn icon-btn ${toolMode === "select" ? "active" : ""}`} title="Select" onClick={() => setToolMode("select")}><FiMousePointer /></button>
            <button type="button" className={`menu-btn icon-btn ${toolMode === "connect" ? "active" : ""}`} title="Connect" onClick={() => setToolMode("connect")}><FiPlusCircle /></button>
          </div>
        </div>

        <input ref={importInputRef} type="file" accept="application/json" className="hidden-input" onChange={onImportFileChange} />

        <ReactFlow
          className={`diagram-flow mode-${toolMode} ${isDraggingNode ? "is-dragging" : ""}`}
          nodes={rendered.nodes}
          edges={rendered.edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          onNodeClick={onNodeClick}
          onEdgeClick={onEdgeClick}
          onEdgeContextMenu={onEdgeContextMenu}
          onNodeDragStart={onNodeDragStart}
          onNodeDragStop={onNodeDragStop}
          onPaneClick={() => setEdgeEditor(null)}
          nodeTypes={nodeTypes}
          fitView
          nodesDraggable={toolMode === "select"}
          nodesConnectable={toolMode === "connect"}
          panOnDrag={toolMode === "select"}
          zoomOnScroll
          zoomOnPinch
          zoomOnDoubleClick
          elementsSelectable
          deleteKeyCode={null}
          connectionLineType={ConnectionLineType.SmoothStep}
          snapToGrid
          snapGrid={[20, 20]}
          minZoom={0.35}
          maxZoom={1.7}
          fitViewOptions={{ padding: 0.2 }}
        >
          <Background color="var(--grid-color)" gap={20} />
          <Controls showInteractive />
        </ReactFlow>

        {nodes.length === 0 || edges.length === 0 ? (
          <div className="canvas-empty-state">Run generation to render a model-produced architecture graph.</div>
        ) : null}

        {edgeEditor ? (
          <div className="edge-floating-editor" style={{ left: edgeEditor.x, top: edgeEditor.y }}>
            <input
              className="edge-floating-input"
              value={edgeEditor.value}
              onChange={(event) => setEdgeEditor((current) => (current ? { ...current, value: event.target.value } : current))}
              onKeyDown={(event) => {
                if (event.key === "Enter") {
                  commitEdgeEditor();
                }
                if (event.key === "Escape") {
                  setEdgeEditor(null);
                }
              }}
              onBlur={commitEdgeEditor}
              autoFocus
            />
            <button type="button" className="edge-floating-delete" onMouseDown={(event) => event.preventDefault()} onClick={deleteEdgeEditorTarget}>Delete</button>
          </div>
        ) : null}
      </div>

      <aside className="property-panel">
        <h3>Properties</h3>
        {selectedNode ? (
          <div className="prop-block">
            <h4>Node</h4>
            <label>Label</label>
            <input className="prop-input" value={selectedNode.data.label} onChange={(event) => updateSelectedNodeLabel(event.target.value)} />
            <label>Type</label>
            <select className="prop-input" value={selectedNode.data.kind} onChange={(event) => updateSelectedNodeKind(event.target.value as FlowNodeKind)}>
              <option value="ui">ui</option>
              <option value="service">service</option>
              <option value="database">db</option>
              <option value="cache">cache</option>
              <option value="queue">queue</option>
              <option value="container">container</option>
            </select>
            <label>Icon</label>
            <select className="prop-input" value={selectedNode.data.icon ?? "auto"} onChange={(event) => updateSelectedNodeIcon(event.target.value)}>
              {ICON_OPTIONS.map((iconName) => (
                <option key={iconName} value={iconName}>{iconName}</option>
              ))}
            </select>
            <label>Background</label>
            <input
              className="prop-input prop-color-input"
              type="color"
              value={selectedNode.data.style?.background || defaultNodeStyle.background}
              onChange={(event) => updateSelectedNodeStyle("background", event.target.value)}
            />
            <label>Border</label>
            <input
              className="prop-input prop-color-input"
              type="color"
              value={selectedNode.data.style?.borderColor || defaultNodeStyle.borderColor}
              onChange={(event) => updateSelectedNodeStyle("borderColor", event.target.value)}
            />
            <label>Text</label>
            <input
              className="prop-input prop-color-input"
              type="color"
              value={selectedNode.data.style?.textColor || defaultNodeStyle.textColor}
              onChange={(event) => updateSelectedNodeStyle("textColor", event.target.value)}
            />
            <label>Color Preview</label>
            <div
              className="prop-color-preview"
              style={{
                background: selectedNode.data.style?.background || defaultNodeStyle.background,
                borderColor: selectedNode.data.style?.borderColor || defaultNodeStyle.borderColor,
                color: selectedNode.data.style?.textColor || defaultNodeStyle.textColor,
              }}
            />
          </div>
        ) : null}

        {selectedEdge ? (
          <div className="prop-block">
            <h4>Edge</h4>
            <label>Label / Type</label>
            <input className="prop-input" value={String(selectedEdge.label ?? selectedEdge.data?.edgeType ?? "HTTP")} onChange={(event) => updateSelectedEdgeType(event.target.value)} />
            <label>Stroke</label>
            <input
              className="prop-input prop-color-input"
              type="color"
              value={selectedEdge.data?.style?.stroke || "#64748b"}
              onChange={(event) => updateSelectedEdgeStyle("stroke", event.target.value)}
            />
            <label>Width</label>
            <input
              className="prop-input"
              type="range"
              min={1}
              max={5}
              step={1}
              value={selectedEdge.data?.style?.width ?? 2}
              onChange={(event) => updateSelectedEdgeStyle("width", Number(event.target.value))}
            />
            <label className="prop-checkbox-row">
              <input
                type="checkbox"
                checked={Boolean(selectedEdge.data?.style?.dashed)}
                onChange={(event) => updateSelectedEdgeStyle("dashed", event.target.checked)}
              />
              Dashed
            </label>
            <label>Path</label>
            <div className="prop-path">{selectedEdge.source} -&gt; {selectedEdge.target}</div>
          </div>
        ) : null}

        {!selectedNode && !selectedEdge ? <p className="prop-empty">Select a node or edge to edit properties.</p> : null}
      </aside>
    </div>
  );
}

export default function DiagramView({ architecture, command, theme, onToggleTheme }: DiagramViewProps) {
  return (
    <ReactFlowProvider>
      <DiagramViewInner architecture={architecture} command={command} theme={theme} onToggleTheme={onToggleTheme} />
    </ReactFlowProvider>
  );
}
