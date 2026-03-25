import { ChangeEvent, useRef, useState } from "react";
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

type EditorNodeType = "ui" | "service" | "data" | "cache" | "queue" | "container";

type EditorCommand = {
  id: number;
  action: "reset" | "clear";
};

function App() {
  const [mode, setMode] = useState<"text" | "json">("text");
  const [input, setInput] = useState(defaultText);
  const [output, setOutput] = useState("");
  const [architecture, setArchitecture] = useState<Architecture>(defaultJson);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [editorCommand, setEditorCommand] = useState<EditorCommand | null>(null);
  const commandIdRef = useRef(0);

  const sendEditorCommand = (action: EditorCommand["action"]) => {
    commandIdRef.current += 1;
    setEditorCommand({ id: commandIdRef.current, action });
  };

  const onDragNodeTemplate = (event: React.DragEvent<HTMLButtonElement>, nodeType: EditorNodeType) => {
    event.dataTransfer.setData("application/x-arch-node", nodeType);
    event.dataTransfer.effectAllowed = "move";
  };

  const onGenerate = async () => {
    setError("");
    setOutput("");

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
        sendEditorCommand("reset");
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
    <main className="editor-shell">
      <aside className="editor-sidebar">
        <h1>ArchitectAI</h1>
        <p className="sub">Generate an initial architecture and then edit it directly on the canvas.</p>

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
          rows={8}
        />

        <button onClick={onGenerate} disabled={loading}>
          {loading ? "Generating..." : "Generate From Prompt"}
        </button>

        <div className="tool-section">
          <h2>Node Tools</h2>
          <div className="tool-grid">
            <button type="button" draggable className="tool-btn draggable" onDragStart={(event) => onDragNodeTemplate(event, "ui")}>UI</button>
            <button type="button" draggable className="tool-btn draggable" onDragStart={(event) => onDragNodeTemplate(event, "service")}>Service</button>
            <button type="button" draggable className="tool-btn draggable" onDragStart={(event) => onDragNodeTemplate(event, "data")}>Database</button>
            <button type="button" draggable className="tool-btn draggable" onDragStart={(event) => onDragNodeTemplate(event, "cache")}>Cache</button>
            <button type="button" draggable className="tool-btn draggable" onDragStart={(event) => onDragNodeTemplate(event, "queue")}>Queue</button>
            <button type="button" draggable className="tool-btn draggable" onDragStart={(event) => onDragNodeTemplate(event, "container")}>Container</button>
          </div>
          <div className="tool-row">
            <button type="button" className="tool-btn" onClick={() => sendEditorCommand("reset")}>Reset Layout</button>
            <button type="button" className="tool-btn danger" onClick={() => sendEditorCommand("clear")}>Clear</button>
          </div>
        </div>

        {error ? <p className="error">{error}</p> : null}

        <div className="output">
          <h2>Explanation</h2>
          <pre>{output || "No explanation yet."}</pre>
        </div>
      </aside>

      <section className="editor-canvas-panel">
        <DiagramView architecture={architecture} command={editorCommand} />
      </section>
    </main>
  );
}

export default App;
