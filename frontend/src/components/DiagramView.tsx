import { useMemo } from "react";
import ReactFlow, { Background, Controls, Edge, Node, ReactFlowProvider } from "reactflow";
import "reactflow/dist/style.css";

type Architecture = {
  nodes: string[];
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

function getNodeType(label: string): LayerType {
  const lowered = label.toLowerCase();
  if (lowered.includes("front") || lowered.includes("ui") || lowered.includes("client")) {
    return "ui";
  }
  if (lowered.includes("db") || lowered.includes("data") || lowered.includes("database")) {
    return "data";
  }
  return "service";
}

function DiagramViewInner({ architecture }: DiagramViewProps) {
  const nodes: Node[] = useMemo(() => {
    const groups: Record<LayerType, string[]> = { ui: [], service: [], data: [] };
    architecture.nodes.forEach((label) => {
      groups[getNodeType(label)].push(label);
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
    const groups: Record<LayerType, string[]> = { ui: [], service: [], data: [] };
    architecture.nodes.forEach((label) => {
      groups[getNodeType(label)].push(label);
    });

    const out: Edge[] = [];
    let edgeId = 1;

    groups.ui.forEach((uiNode) => {
      groups.service.forEach((serviceNode) => {
        out.push({ id: `e${edgeId++}`, source: uiNode, target: serviceNode, animated: false });
      });
    });

    groups.service.forEach((serviceNode) => {
      groups.data.forEach((dataNode) => {
        out.push({ id: `e${edgeId++}`, source: serviceNode, target: dataNode, animated: false });
      });
    });

    return out;
  }, [architecture.nodes]);

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
