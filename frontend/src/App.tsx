import { ChangeEvent, DragEvent, useEffect, useRef, useState } from "react";
import DiagramView from "./components/DiagramView";

const API_URL = "http://127.0.0.1:8000/explain";

const defaultText = "A frontend app calls an API service, which writes to postgres and publishes jobs to a queue.";

type ArchitectureNode = {
  id: string;
  type: "ui" | "service" | "database" | "cache" | "queue" | "container";
  [key: string]: unknown;
};

type ArchitectureEdge = {
  source: string;
  target: string;
  label?: string;
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

type ThemeMode = "light" | "dark";

function App() {
  const [input, setInput] = useState(defaultText);
  const [architecture, setArchitecture] = useState<Architecture | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [editorCommand, setEditorCommand] = useState<EditorCommand | null>(null);
  const [theme, setTheme] = useState<ThemeMode>(() => {
    const stored = localStorage.getItem("architectai-theme");
    return stored === "dark" ? "dark" : "light";
  });
  const commandIdRef = useRef(0);

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("architectai-theme", theme);
  }, [theme]);

  const sendEditorCommand = (action: EditorCommand["action"]) => {
    commandIdRef.current += 1;
    setEditorCommand({ id: commandIdRef.current, action });
  };

  const onDragNodeTemplate = (event: DragEvent<HTMLButtonElement>, nodeType: EditorNodeType) => {
    event.dataTransfer.setData("application/x-arch-node", nodeType);
    event.dataTransfer.effectAllowed = "move";
  };

  const onGenerate = async () => {
    setError("");
    setArchitecture(null);

    const text = input.trim();
    if (!text) {
      setError("Please enter architecture text.");
      return;
    }

    setLoading(true);
    try {
      const requestInit: RequestInit = {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      };

      let response: Response;
      try {
        response = await fetch(API_URL, requestInit);
      } catch {
        response = await fetch(API_URL, requestInit);
      }

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

      const data = (await response.json()) as { architecture?: unknown; raw_model_output?: string };
      console.log("API RESPONSE:", data);

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
        throw new Error("Backend response did not include a valid architecture graph.");
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
        <p className="sub">Enter text. Backend runs the model and returns a graph JSON.</p>

        <label htmlFor="architecture-input">Architecture Description</label>
        <textarea
          id="architecture-input"
          value={input}
          onChange={(e: ChangeEvent<HTMLTextAreaElement>) => setInput(e.target.value)}
          rows={8}
        />

        <button onClick={onGenerate} disabled={loading}>
          {loading ? "Generating..." : "Generate From Prompt"}
        </button>

          <p className="status-line">
            {loading ? "Generating architecture..." : architecture ? "Architecture generated successfully." : "Ready to generate."}
          </p>

        <div className="tool-section">
          <h2>Node Tools</h2>
          <p className="sub">Drag an item onto the canvas to create a node.</p>
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
      </aside>

      <section className="editor-canvas-panel">
        {architecture ? (
          <DiagramView
            architecture={architecture}
            command={editorCommand}
            theme={theme}
            onToggleTheme={() => setTheme((current) => (current === "light" ? "dark" : "light"))}
          />
        ) : (
          <div className="empty-state">Run generation to render a model-produced architecture graph.</div>
        )}
      </section>
    </main>
  );
}

export default App;
