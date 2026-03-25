import { ChangeEvent, useState } from "react";
import DiagramView from "./components/DiagramView";

const defaultJson = {
  nodes: ["frontend", "backend", "database"],
};

const defaultText = "3-tier app with frontend, backend, database";

function App() {
  const [mode, setMode] = useState<"text" | "json">("text");
  const [input, setInput] = useState(defaultText);
  const [output, setOutput] = useState("");
  const [architecture, setArchitecture] = useState<{ nodes: string[] } | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const architectureFromText = (text: string): { nodes: string[] } => {
    const lowered = text.toLowerCase();
    const nodes: string[] = [];
    if (lowered.includes("frontend")) nodes.push("frontend");
    if (lowered.includes("backend")) nodes.push("backend");
    if (lowered.includes("database") || lowered.includes(" db ") || lowered.startsWith("db ") || lowered.endsWith(" db")) nodes.push("database");
    if (nodes.length === 0) return { nodes: ["frontend", "backend", "database"] };
    return { nodes };
  };

  const architectureFromJson = (inputJson: unknown): { nodes: string[] } | null => {
    if (!inputJson || typeof inputJson !== "object") return null;
    const value = inputJson as { nodes?: unknown[] };
    if (!Array.isArray(value.nodes)) return null;

    const nodes = value.nodes
      .map((n) => {
        if (typeof n === "string") return n;
        if (n && typeof n === "object" && "label" in n && typeof (n as { label?: unknown }).label === "string") {
          return (n as { label: string }).label;
        }
        return "";
      })
      .filter((n) => n.length > 0);

    return nodes.length > 0 ? { nodes } : null;
  };

  const onGenerate = async () => {
    setError("");
    setOutput("");
    setArchitecture(null);

    let body: Record<string, unknown>;
    if (mode === "json") {
      let parsedInput: unknown;
      try {
        parsedInput = JSON.parse(input);
      } catch {
        setError("Invalid JSON. Please fix the input and try again.");
        return;
      }
      body = { architecture: parsedInput };
      setArchitecture(architectureFromJson(parsedInput));
    } else {
      const text = input.trim();
      if (!text) {
        setError("Please enter architecture text.");
        return;
      }
      body = { text };
      setArchitecture(architectureFromText(text));
    }

    setLoading(true);
    try {
      const response = await fetch("http://127.0.0.1:8000/explain", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });

      if (!response.ok) {
        let backendMessage = "Request failed.";
        try {
          const errorJson = (await response.json()) as { detail?: string };
          backendMessage = errorJson.detail || backendMessage;
        } catch {
          const text = await response.text();
          backendMessage = text || backendMessage;
        }
        throw new Error(backendMessage);
      }

      const data = (await response.json()) as { explanation?: string };
      setOutput(data.explanation ?? "No explanation returned.");
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unknown error";
      setError(`Request failed: ${message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="page">
      <section className="card">
        <h1>ArchitectAI</h1>
        <p className="sub">Enter architecture as plain text or JSON and generate an explanation.</p>

        <div className="mode-toggle">
          <button
            className={`mode-btn ${mode === "text" ? "active" : ""}`}
            type="button"
            onClick={() => {
              setMode("text");
              setInput(defaultText);
            }}
          >
            Text Mode
          </button>
          <button
            className={`mode-btn ${mode === "json" ? "active" : ""}`}
            type="button"
            onClick={() => {
              setMode("json");
              setInput(JSON.stringify(defaultJson, null, 2));
            }}
          >
            JSON Mode
          </button>
        </div>

        <label htmlFor="architecture-input">{mode === "json" ? "Architecture JSON" : "Architecture Description"}</label>
        <textarea
          id="architecture-input"
          value={input}
          onChange={(e: ChangeEvent<HTMLTextAreaElement>) => setInput(e.target.value)}
          rows={12}
        />

        <button onClick={onGenerate} disabled={loading}>
          {loading ? "Generating..." : "Generate Explanation"}
        </button>

        {error ? <p className="error">{error}</p> : null}

        <div className="output">
          <h2>Explanation</h2>
          <pre>{output || "No output yet."}</pre>
        </div>

        {architecture ? (
          <div className="diagram-panel">
            <h2>Architecture Diagram</h2>
            <DiagramView architecture={architecture} />
          </div>
        ) : null}
      </section>
    </main>
  );
}

export default App;
