/**
 * Custom React Flow edge types for the ArchitectAI diagram editor.
 *
 * Currently exports the default ``smoothstep`` style with a protocol label.
 * Add custom edge components here when richer styling is needed.
 */

import { SmoothStepEdge } from "reactflow";

/** Registry passed to ``<ReactFlow edgeTypes={...} />`` */
export const edgeTypes = {
  default: SmoothStepEdge,
};
