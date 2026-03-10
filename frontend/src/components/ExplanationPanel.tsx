import React from "react";
import type { Architecture } from "../types/architecture";

interface Props {
  architecture: Architecture;
  /** AI-generated explanation text from POST /explain. */
  explanation?: string | null;
  /** True while the /explain request is in flight. */
  isLoadingExplanation?: boolean;
}

/**
 * ExplanationPanel
 *
 * Displays:
 *   1. AI-generated explanation (from Qwen2.5-3B-Instruct + ConvNeXt vision)
 *   2. Structured architecture details (components + connections)
 */
export const ExplanationPanel: React.FC<Props> = ({
  architecture,
  explanation,
  isLoadingExplanation = false,
}) => {
  const { nodes, edges, metadata } = architecture;

  return (
    <aside
      className="w-full rounded-xl border border-gray-200 bg-gray-50 p-4 text-sm
                   shadow-sm space-y-4"
      aria-label="Architecture explanation"
    >
      {/* ── AI Explanation ─────────────────────────────────────────────── */}
      <section>
        <h2 className="font-semibold text-gray-800 mb-2 flex items-center gap-2">
          <span className="text-blue-600">✦</span> AI Explanation
        </h2>

        {isLoadingExplanation && (
          <div className="flex items-center gap-2 text-blue-500 text-xs animate-pulse">
            <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
              <circle
                className="opacity-25" cx="12" cy="12" r="10"
                stroke="currentColor" strokeWidth="4"
              />
              <path
                className="opacity-75" fill="currentColor"
                d="M4 12a8 8 0 018-8v8H4z"
              />
            </svg>
            Generating explanation…
          </div>
        )}

        {!isLoadingExplanation && explanation && (
          <p className="text-gray-700 leading-relaxed bg-white rounded-lg
                        border border-blue-100 p-3 shadow-xs">
            {explanation}
          </p>
        )}

        {!isLoadingExplanation && !explanation && (
          <p className="text-gray-400 italic text-xs">
            Explanation will appear after diagram generation.
          </p>
        )}
      </section>

      <hr className="border-gray-200" />

      {/* ── Metadata ────────────────────────────────────────────────────── */}
      <div className="text-xs text-gray-500">
        Schema v{metadata.version}
        {metadata.style ? ` · style: ${metadata.style}` : ""}
      </div>

      {/* ── Components ──────────────────────────────────────────────────── */}
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
                className="inline-block w-2 h-2 rounded-full flex-shrink-0"
                style={{ background: nodeTypeColor(n.type) }}
              />
              <span className="font-mono text-xs text-gray-500 w-24 truncate">
                {n.id}
              </span>
              <span className="text-gray-700 truncate">{n.label}</span>
              <span className="ml-auto text-xs text-gray-400 flex-shrink-0">{n.type}</span>
            </li>
          ))}
        </ul>
      </section>

      {/* ── Connections ─────────────────────────────────────────────────── */}
      {edges.length > 0 && (
        <section>
          <h3 className="font-medium text-gray-700 mb-1">
            Data Flow ({edges.length} connection{edges.length !== 1 ? "s" : ""})
          </h3>
          <ul className="space-y-1">
            {edges.map((e, i) => (
              <li
                key={i}
                className="flex items-center gap-2 text-xs text-gray-600"
              >
                <span
                  className="inline-block w-1.5 h-1.5 rounded-full flex-shrink-0"
                  style={{ background: nodeTypeColor(
                    nodes.find((n) => n.id === e.from)?.type ?? "Backend"
                  )}}
                />
                <span className="font-mono truncate max-w-[80px]">{e.from}</span>
                <span className="text-gray-400">→</span>
                <span className="font-mono truncate max-w-[80px]">{e.to}</span>
                {e.protocol && (
                  <span className="ml-auto flex-shrink-0 rounded bg-blue-100
                                   px-1.5 py-0.5 text-blue-700">
                    {e.protocol}
                  </span>
                )}
              </li>
            ))}
          </ul>
        </section>
      )}
    </aside>
  );
};

function nodeTypeColor(type: string): string {
  const colors: Record<string, string> = {
    Frontend: "#3b82f6",
    Backend:  "#10b981",
    Service:  "#8b5cf6",
    Database: "#f59e0b",
    Cache:    "#ef4444",
    Queue:    "#f97316",
    External: "#6b7280",
  };
  return colors[type] ?? "#94a3b8";
}

export default ExplanationPanel;


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
