/**
 * Named re-exports for all backend API functions.
 * Import from here rather than directly from client.ts.
 */
export {
  generateArchitecture,
  generateDiagram,
  parsePrompt,
} from "./client";
