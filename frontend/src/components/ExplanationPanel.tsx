import React from "react";
import type { Architecture } from "../types/architecture";

interface Props {
  architecture: Architecture;
}

/**
 * ExplanationPanel
 *
 * Displays a structured summary of the parsed Architecture.
 * Will be augmented with VLM-generated text in a later module.
 */
export const ExplanationPanel: React.FC<Props> = ({ architecture }) => {
  const { nodes, edges, metadata } = architecture;

  return (
    <aside
      className="w-full rounded-xl border border-gray-200 bg-gray-50 p-4 text-sm
                   shadow-sm space-y-3"
      aria-label="Architecture explanation"
    >
      <h2 className="font-semibold text-gray-800">Architecture Details</h2>

      {/* Metadata */}
      <div className="text-xs text-gray-500">
        Schema v{metadata.version}
        {metadata.style ? ` · style: ${metadata.style}` : ""}
      </div>

      {/* Nodes */}
      <section>
        <h3 className="font-medium text-gray-700 mb-1">
          Components ({nodes.length})
        </h3>
        <ul className="space-y-1">
          {nodes.map((n) => (
            <li
              key={n.id}
              className="flex items-center gap-2 rounded-md bg-white px-3 py-1.5
                         border border-gray-100 shadow-xs"
            >
              <span
                className="inline-block w-2 h-2 rounded-full"
                style={{ background: nodeTypeColor(n.type) }}
              />
              <span className="font-mono text-xs text-gray-500 w-24 truncate">
                {n.id}
              </span>
              <span className="text-gray-700">{n.label}</span>
              <span className="ml-auto text-xs text-gray-400">{n.type}</span>
            </li>
          ))}
        </ul>
      </section>

      {/* Edges */}
      <section>
        <h3 className="font-medium text-gray-700 mb-1">
          Connections ({edges.length})
        </h3>
        <ul className="space-y-1">
          {edges.map((e, i) => (
            <li
              key={i}
              className="flex items-center gap-2 text-xs text-gray-600"
            >
              <span className="font-mono">{e.from}</span>
              <span className="text-gray-400">→</span>
              <span className="font-mono">{e.to}</span>
              {e.protocol && (
                <span className="ml-auto rounded bg-blue-100 px-1.5 py-0.5 text-blue-700 text-xs">
                  {e.protocol}
                </span>
              )}
            </li>
          ))}
        </ul>
      </section>
    </aside>
  );
};

function nodeTypeColor(type: string): string {
  const colors: Record<string, string> = {
    Frontend: "#3b82f6",
    Backend: "#10b981",
    Service: "#8b5cf6",
    Database: "#f59e0b",
    Cache: "#ef4444",
    Queue: "#f97316",
    External: "#6b7280",
  };
  return colors[type] ?? "#94a3b8";
}

export default ExplanationPanel;
