/**
 * Toolbar
 *
 * Top application bar with the app name and a simple tab switcher
 * (Viewer vs Editor mode).
 */

import React from "react";

export type ViewMode = "viewer" | "editor";

interface Props {
  mode: ViewMode;
  onModeChange: (mode: ViewMode) => void;
  hasArchitecture: boolean;
}

export const Toolbar: React.FC<Props> = ({ mode, onModeChange, hasArchitecture }) => (
  <header className="flex items-center justify-between px-6 py-3 bg-white
                     border-b border-gray-200 shadow-sm">
    <div className="flex items-center gap-2">
      <span className="text-xl font-bold text-blue-700">ArchitectAI</span>
      <span className="text-xs text-gray-400 ml-1">by ArchitectAI</span>
    </div>

    {hasArchitecture && (
      <nav className="flex gap-1 rounded-lg overflow-hidden border border-gray-200">
        {(["viewer", "editor"] as ViewMode[]).map((m) => (
          <button
            key={m}
            onClick={() => onModeChange(m)}
            className={`px-4 py-1.5 text-sm font-medium capitalize transition-colors ${
              mode === m
                ? "bg-blue-600 text-white"
                : "bg-white text-gray-600 hover:bg-gray-50"
            }`}
          >
            {m}
          </button>
        ))}
      </nav>
    )}
  </header>
);

export default Toolbar;
