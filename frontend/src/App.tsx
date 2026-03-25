import { ChangeEvent, useState } from "react";

const defaultJson = {
  nodes: ["frontend", "backend", "database"],
};

function App() {
  const [input, setInput] = useState(JSON.stringify(defaultJson, null, 2));
  const [output, setOutput] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const onGenerate = async () => {
    setError("");
    setOutput("");

    let parsedInput: unknown;
    try {
      parsedInput = JSON.parse(input);
    } catch {
      setError("Invalid JSON. Please fix the input and try again.");
      return;
    }

    setLoading(true);
    try {
      const response = await fetch("http://127.0.0.1:8000/explain", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ architecture: parsedInput }),
      });

      if (!response.ok) {
        const text = await response.text();
        throw new Error(text || "Request failed.");
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
        <p className="sub">Paste architecture JSON and generate an explanation.</p>

        <label htmlFor="architecture-input">Architecture JSON</label>
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
