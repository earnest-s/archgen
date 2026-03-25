import { ChangeEvent, useState } from "react";
import DiagramView from "./components/DiagramView";

const defaultJson = {
  nodes: [
    { id: "frontend", type: "ui" },
    { id: "backend", type: "service" },
    { id: "database", type: "data" },
  ],
  edges: [
    { source: "frontend", target: "backend" },
    { source: "backend", target: "database" },
  ],
};

const defaultText = "3-tier app with frontend, backend, database";

type ArchitectureNode = {
  id: string;
  type?: string;
  [key: string]: unknown;
};

type ArchitectureEdge = {
  source: string;
  target: string;
  [key: string]: unknown;
};

type Architecture = {
  nodes: ArchitectureNode[];
  edges: ArchitectureEdge[];
  [key: string]: unknown;
};

function App() {
  const [mode, setMode] = useState<"text" | "json">("text");
  const [input, setInput] = useState(defaultText);
  const [output, setOutput] = useState("");
  const [architecture, setArchitecture] = useState<Architecture | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

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
    } else {
      const text = input.trim();
      if (!text) {
        setError("Please enter architecture text.");
        return;
      }
      body = { text };
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

      const data = (await response.json()) as { explanation?: string; architecture?: unknown };
      console.log("API RESPONSE:", data);
      setOutput(data.explanation ?? "No explanation returned.");

      const arch = data.architecture as { nodes?: unknown; edges?: unknown } | undefined;
      const isValid =
        !!arch &&
        Array.isArray(arch.nodes) &&
        Array.isArray(arch.edges) &&
        arch.nodes.length > 0 &&
        arch.edges.length > 0;

      if (isValid) {
        setArchitecture(data.architecture as Architecture);
      } else {
        console.error("INVALID ARCH:", data);
        setError("Backend response did not include a valid architecture graph.");
      }
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
