/**
 * App
 *
 * Root component.  Manages global state:
 *  - prompt (string)
 *  - architecture (Architecture | null)
 *  - diagramBase64 (string | null)
 *  - viewMode ("viewer" | "editor")
 *
 * Flow:
 *  User enters prompt in PromptInput
 *  → calls POST /generate via generateArchitecture()
 *  → stores architecture + base64 diagram in state
 *  → DiagramViewer renders the PNG
 *  → user can switch to DiagramEditor for interactive editing
 *  → onArchitectureChange keeps architecture state in sync with editor
 */

import React, { useState } from "react";

import { generateArchitecture } from "./api/client";
import { DiagramEditor } from "./components/DiagramEditor";
import { DiagramViewer } from "./components/DiagramViewer";
import { ExplanationPanel } from "./components/ExplanationPanel";
import { PromptInput } from "./components/PromptInput";
import { Toolbar, ViewMode } from "./components/Toolbar";
import type { Architecture } from "./types/architecture";

const App: React.FC = () => {
  const [isLoading, setIsLoading]         = useState(false);
  const [error, setError]                 = useState<string | null>(null);
  const [architecture, setArchitecture]   = useState<Architecture | null>(null);
  const [diagramBase64, setDiagramBase64] = useState<string | null>(null);
  const [viewMode, setViewMode]           = useState<ViewMode>("viewer");

  // ── Generate full pipeline ─────────────────────────────────────────────────
  const handleGenerate = async (prompt: string) => {
    setIsLoading(true);
    setError(null);
    try {
      const resp = await generateArchitecture(prompt);
      setArchitecture(resp.architecture);
      setDiagramBase64(resp.diagram_base64 ?? null);
      setViewMode("viewer");
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setIsLoading(false);
    }
  };

  // ── Editor sync ────────────────────────────────────────────────────────────
  const handleArchitectureChange = (updated: Architecture) => {
    setArchitecture(updated);
    // Clear stale diagram PNG — user must re-render after editing.
    setDiagramBase64(null);
  };

  return (
    <div className="flex flex-col min-h-screen bg-gray-50">
      {/* Top bar */}
      <Toolbar
        mode={viewMode}
        onModeChange={setViewMode}
        hasArchitecture={!!architecture}
      />

      <main className="flex flex-col gap-6 p-6 max-w-6xl mx-auto w-full">
        {/* Input */}
        <PromptInput onGenerate={handleGenerate} isLoading={isLoading} />

        {/* Error banner */}
        {error && (
          <div className="rounded-lg bg-red-50 border border-red-200 p-4 text-sm text-red-700">
            <strong>Error:</strong> {error}
          </div>
        )}

        {/* Loading skeleton */}
        {isLoading && (
          <div className="flex h-64 w-full items-center justify-center rounded-xl
                          border-2 border-dashed border-blue-200 text-blue-400 text-sm
                          animate-pulse">
            Generating diagram…
          </div>
        )}

        {/* Results */}
        {!isLoading && architecture && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Main panel: viewer or editor */}
            <div className="lg:col-span-2">
              {viewMode === "viewer" ? (
                <DiagramViewer
                  diagramBase64={diagramBase64 ?? undefined}
                  alt="Generated architecture diagram"
                />
              ) : (
                <div className="w-full rounded-xl border border-gray-200 bg-white
                                shadow-sm overflow-hidden" style={{ height: 520 }}>
                  <DiagramEditor
                    architecture={architecture}
                    onArchitectureChange={handleArchitectureChange}
                  />
                </div>
              )}
            </div>

            {/* Sidebar */}
            <div className="lg:col-span-1">
              <ExplanationPanel architecture={architecture} />
            </div>
          </div>
        )}
      </main>
    </div>
  );
};

export default App;
