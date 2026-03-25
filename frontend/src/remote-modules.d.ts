declare module "https://esm.sh/dagre@0.8.5" {
  const dagre: {
    graphlib: {
      Graph: new () => {
        setGraph: (config: Record<string, unknown>) => void;
        setDefaultEdgeLabel: (label: () => Record<string, unknown>) => void;
        setNode: (id: string, data: Record<string, unknown>) => void;
        setEdge: (source: string, target: string) => void;
        node: (id: string) => { x: number; y: number };
      };
    };
    layout: (graph: unknown) => void;
  };

  export default dagre;
}
