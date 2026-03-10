import React, { useState } from "react";

interface Props {
  onGenerate: (prompt: string) => void;
  isLoading: boolean;
}

/**
 * PromptInput
 *
 * A textarea + submit button that captures the user's architecture description
 * and fires the onGenerate callback.
 */
export const PromptInput: React.FC<Props> = ({ onGenerate, isLoading }) => {
  const [prompt, setPrompt] = useState<string>("");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = prompt.trim();
    if (!trimmed) return;
    onGenerate(trimmed);
  };

  return (
    <form
      onSubmit={handleSubmit}
      className="flex flex-col gap-3 w-full max-w-2xl"
      aria-label="Architecture prompt form"
    >
      <label
        htmlFor="arch-prompt"
        className="text-sm font-semibold text-gray-700"
      >
        Describe your architecture
      </label>
      <textarea
        id="arch-prompt"
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
        rows={4}
        placeholder="e.g. React frontend, FastAPI backend, Redis cache, PostgreSQL database"
        className="w-full rounded-lg border border-gray-300 p-3 text-sm shadow-sm
                   focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none"
        disabled={isLoading}
      />
      <button
        type="submit"
        disabled={isLoading || !prompt.trim()}
        className="self-end rounded-lg bg-blue-600 px-6 py-2 text-sm font-semibold
                   text-white shadow hover:bg-blue-700 disabled:opacity-50
                   disabled:cursor-not-allowed transition-colors"
      >
        {isLoading ? "Generating…" : "Generate Diagram"}
      </button>
    </form>
  );
};

export default PromptInput;
