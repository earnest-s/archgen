/**
 * Shared TypeScript types that mirror the Pydantic models in
 * backend/api/schemas/architecture.py. Keep in sync with the JSON schema at
 * shared/architecture.schema.json.
 */

export type NodeType =
  | "Frontend"
  | "Backend"
  | "Service"
  | "Database"
  | "Cache"
  | "Queue"
  | "External";

export interface ArchNode {
  id: string;
  type: NodeType;
  label: string;
  layer?: string;
}

export interface ArchEdge {
  from: string;
  to: string;
  protocol?: string;
}

export interface Metadata {
  version: number;
  style?: string;
}

export interface Architecture {
  nodes: ArchNode[];
  edges: ArchEdge[];
  metadata: Metadata;
}

/** Response from POST /parse-prompt */
export interface ParsePromptResponse {
  architecture: Architecture;
}

/** Response from POST /generate-diagram */
export interface GenerateDiagramResponse {
  diagram_path: string;
  diagram_base64?: string;
}

/** Response from POST /generate (full pipeline) */
export interface FullPipelineResponse {
  architecture: Architecture;
  diagram_path: string;
  diagram_base64?: string;
}
