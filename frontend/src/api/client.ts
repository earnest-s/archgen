/**
 * Low-level HTTP client for the ArchitectAI backend.
 *
 * Reads the base URL from the Vite environment variable
 * VITE_API_BASE_URL (defaults to http://localhost:8000).
 *
 * All functions throw on non-2xx responses.
 */

import type {
  Architecture,
  ExplainResponse,
  FullPipelineResponse,
  GenerateDiagramResponse,
  ParsePromptResponse,
} from "../types/architecture";

const BASE_URL =
  (import.meta as Record<string, any>).env?.VITE_API_BASE_URL ??
  "http://localhost:8000";

async function post<T>(endpoint: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE_URL}${endpoint}?inline=true`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    const detail = await res.text();
    throw new Error(
      `ArchitectAI API error ${res.status} on ${endpoint}: ${detail}`
    );
  }

  return res.json() as Promise<T>;
}

// ---------------------------------------------------------------------------
// Public API functions
// ---------------------------------------------------------------------------

/**
 * Call POST /parse-prompt.
 * Returns the parsed Architecture without generating a diagram.
 */
export async function parsePrompt(
  prompt: string
): Promise<ParsePromptResponse> {
  return post<ParsePromptResponse>("/parse-prompt", { prompt });
}

/**
 * Call POST /generate-diagram.
 * Returns diagram path and optional base64 PNG.
 */
export async function generateDiagram(
  architecture: Architecture
): Promise<GenerateDiagramResponse> {
  return post<GenerateDiagramResponse>("/generate-diagram", { architecture });
}

/**
 * Call POST /generate (full pipeline).
 * Returns architecture JSON + diagram path/base64.
 */
export async function generateArchitecture(
  prompt: string
): Promise<FullPipelineResponse> {
  return post<FullPipelineResponse>("/generate", { prompt });
}

/**
 * Call POST /explain.
 * Returns AI-generated plain-English explanation of the architecture.
 */
export async function explainArchitecture(
  architecture: Architecture,
  diagram_path?: string
): Promise<ExplainResponse> {
  return post<ExplainResponse>("/explain", { architecture, diagram_path });
}
