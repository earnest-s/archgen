import { useMemo } from "react";
import ReactFlow, { Background, Controls, Edge, Node, ReactFlowProvider } from "reactflow";
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

function normalizeLayerType(input: string | undefined): LayerType {
  if (!input) return "service";
  const lowered = input.toLowerCase();
  if (lowered === "ui") return "ui";
  if (lowered === "data") return "data";
  return "service";
}

function DiagramViewInner({ architecture }: DiagramViewProps) {
  const nodes: Node[] = useMemo(() => {
    const groups: Record<LayerType, string[]> = { ui: [], service: [], data: [] };
    architecture.nodes.forEach((node) => {
      if (typeof node.id !== "string" || !node.id) return;
      groups[normalizeLayerType(node.type)].push(node.id);
    });

    const spacing = 220;
    const out: Node[] = [];

    (Object.keys(groups) as LayerType[]).forEach((layer) => {
      const layerNodes = groups[layer];
      const count = layerNodes.length;
      const startX = count > 1 ? -((count - 1) * spacing) / 2 : 0;

      layerNodes.forEach((label, index) => {
        out.push({
          id: label,
          data: { label },
          position: {
            x: startX + index * spacing,
            y: layerY[layer],
          },
          style: {
            borderRadius: 10,
            border: `1px solid ${layerBorder[layer]}`,
            padding: 8,
            background: layerColor[layer],
            width: 150,
            textAlign: "center",
            fontSize: 13,
            fontWeight: 600,
          },
        });
      });
    });

    return out;
  }, [architecture.nodes]);

  const edges: Edge[] = useMemo(() => {
    return architecture.edges.map((edge, index) => ({
      id: `e${index + 1}`,
      source: edge.source,
      target: edge.target,
      animated: false,
    }));
  }, [architecture.edges]);

  return (
    <div className="diagram-canvas">
      <ReactFlow nodes={nodes} edges={edges} fitView>
        <Background color="#e5e7eb" gap={18} />
        <Controls showInteractive={false} />
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
