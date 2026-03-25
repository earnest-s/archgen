import { useMemo } from "react";
import ReactFlow, { Background, Controls, Edge, Node, ReactFlowProvider } from "reactflow";
import "reactflow/dist/style.css";

type Architecture = {
  nodes: string[];
};

type DiagramViewProps = {
  architecture: Architecture;
};

function DiagramViewInner({ architecture }: DiagramViewProps) {
  const nodes: Node[] = useMemo(
    () =>
      architecture.nodes.map((node, index) => ({
        id: node,
        data: { label: node },
        position: { x: index * 200, y: 100 },
        style: {
          borderRadius: 10,
          border: "1px solid #d1d5db",
          padding: 8,
          background: "#ffffff",
          width: 140,
          textAlign: "center",
          fontSize: 13,
        },
      })),
    [architecture.nodes]
  );

  const edges: Edge[] = useMemo(() => {
    const out: Edge[] = [];
    for (let i = 0; i < architecture.nodes.length - 1; i += 1) {
      out.push({
        id: `e${i + 1}`,
        source: architecture.nodes[i],
        target: architecture.nodes[i + 1],
      });
    }
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
