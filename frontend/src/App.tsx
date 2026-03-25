import { ChangeEvent, useState } from "react";

const defaultJson = {
  nodes: ["frontend", "backend", "database"],
};

const defaultText = "3-tier app with frontend, backend, database";

function App() {
  const [mode, setMode] = useState<"text" | "json">("text");
  const [input, setInput] = useState(defaultText);
  const [output, setOutput] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

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
      </section>
    </main>
  );
}

export default App;
