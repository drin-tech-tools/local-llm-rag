"use client";
import { useCallback, useEffect, useState } from "react";

type Health = {
  ok: boolean;
  baseUrl: string;
  chatModel: { name: string; available: boolean };
  embedModel: { name: string; available: boolean };
  error?: string;
};
type Doc = { id: string; name: string; pages: number; chunks: number; embedModel: string; stale: boolean };
type Source = { n: number; document: string; page: number; score: number; snippet: string };
type Msg = {
  role: "user" | "assistant";
  text: string;
  sources?: Source[];
  timing?: string;
};

/** Decode base64(UTF-8 JSON) from X-Rag-Sources (atob alone is Latin-1, not UTF-8-safe). */
function decodeSources(header: string): Source[] {
  const binary = atob(header);
  const bytes = Uint8Array.from(binary, (c) => c.charCodeAt(0));
  return JSON.parse(new TextDecoder().decode(bytes)) as Source[];
}

function seconds(ms: number) {
  return `${(ms / 1000).toFixed(1)}s`;
}

function StatusBar({ health, onRefresh }: { health: Health | null; onRefresh: () => void }) {
  if (!health) {
    return (
      <div className="status">
        <span className="dot wait" /> Checking model server…
      </div>
    );
  }
  const missing = [health.chatModel, health.embedModel].filter((m) => !m.available);
  return (
    <div className="status">
      <span className={`dot ${health.ok ? "ok" : "bad"}`} />
      <div style={{ flex: 1 }}>
        {health.ok ? (
          <>
            Chat <code>{health.chatModel.name}</code> · Embeddings <code>{health.embedModel.name}</code>
            <div className="muted">{health.baseUrl}</div>
          </>
        ) : health.error ? (
          <>
            {health.error}
            <div className="muted">See docs/LOCAL_SETUP.md, step 2.</div>
          </>
        ) : (
          <>
            Missing models:{" "}
            {missing.map((m) => (
              <code key={m.name} style={{ marginRight: 6 }}>
                ollama pull {m.name}
              </code>
            ))}
          </>
        )}
      </div>
      <button type="button" className="ghost" onClick={onRefresh}>
        Recheck
      </button>
    </div>
  );
}

export default function Home() {
  const [health, setHealth] = useState<Health | null>(null);
  const [docs, setDocs] = useState<Doc[]>([]);
  const [selected, setSelected] = useState<string>("");
  const [uploadStatus, setUploadStatus] = useState("");
  const [uploading, setUploading] = useState(false);
  const [question, setQuestion] = useState("");
  const [msgs, setMsgs] = useState<Msg[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const checkHealth = useCallback(async () => {
    setHealth(null);
    try {
      setHealth(await (await fetch("/api/health")).json());
    } catch {
      setHealth({
        ok: false,
        baseUrl: "",
        chatModel: { name: "", available: false },
        embedModel: { name: "", available: false },
        error: "Could not reach this app's API.",
      });
    }
  }, []);

  const refreshDocs = useCallback(async () => {
    const res = await fetch("/api/documents");
    if (res.ok) setDocs(((await res.json()) as { documents: Doc[] }).documents);
  }, []);

  useEffect(() => {
    void checkHealth();
    void refreshDocs();
  }, [checkHealth, refreshDocs]);

  async function upload(file: File) {
    setUploading(true);
    setError("");
    setUploadStatus(`Reading and embedding ${file.name}…`);
    try {
      const fd = new FormData();
      fd.append("file", file);
      const res = await fetch("/api/upload", { method: "POST", body: fd });
      const data = await res.json();
      if (!res.ok) {
        setUploadStatus("");
        setError(data.error ?? "Upload failed");
        return;
      }
      setUploadStatus(
        `${data.document.name}: ${data.document.pages} pages, ${data.document.chunks} chunks, embedded in ${seconds(data.embedMs)}.`,
      );
      setSelected(data.document.id);
      await refreshDocs();
    } catch (err) {
      setUploadStatus("");
      setError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setUploading(false);
    }
  }

  async function remove(id: string) {
    await fetch(`/api/documents/${id}`, { method: "DELETE" });
    if (selected === id) setSelected("");
    await refreshDocs();
  }

  function setLast(update: Partial<Msg>) {
    setMsgs((m) => {
      const next = [...m];
      next[next.length - 1] = { ...next[next.length - 1], ...update };
      return next;
    });
  }

  async function ask(e: React.FormEvent) {
    e.preventDefault();
    const q = question.trim();
    if (!q) return;
    setQuestion("");
    setError("");
    setMsgs((m) => [...m, { role: "user", text: q }, { role: "assistant", text: "" }]);
    setLoading(true);
    const started = performance.now();

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q, documentId: selected || null }),
      });

      if ((res.headers.get("content-type") ?? "").includes("application/json")) {
        const data = await res.json();
        if (!res.ok) {
          setMsgs((m) => m.slice(0, -1));
          setError(data.error ?? "Chat failed");
        } else {
          setLast({ text: data.answer, sources: data.sources });
        }
        return;
      }

      const header = res.headers.get("X-Rag-Sources");
      const sources = header ? decodeSources(header) : [];
      const model = res.headers.get("X-Chat-Model") ?? "model";
      const firstTokenMs = Number(res.headers.get("X-First-Token-Ms") ?? 0);
      setLast({ sources });

      const reader = res.body!.getReader();
      const decoder = new TextDecoder();
      let text = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        text += decoder.decode(value, { stream: true });
        setLast({ text });
      }
      setLast({
        text,
        timing: `${model} · first token ${seconds(firstTokenMs)} · total ${seconds(performance.now() - started)}`,
      });
    } catch (err) {
      setMsgs((m) => m.slice(0, -1));
      setError(err instanceof Error ? err.message : "Chat failed");
    } finally {
      setLoading(false);
    }
  }

  const searchable = docs.filter((d) => !d.stale);

  return (
    <main>
      <h1>Local LLM RAG</h1>
      <p className="muted" style={{ marginTop: 0 }}>
        Chat with your PDFs on models running on your own machine. Answers cite source pages.
      </p>

      <div className="card">
        <StatusBar health={health} onRefresh={checkHealth} />
      </div>

      <div className="card">
        <input
          type="file"
          accept="application/pdf"
          disabled={uploading}
          onChange={(e) => {
            const file = e.target.files?.[0];
            e.target.value = "";
            if (file) void upload(file);
          }}
        />
        {uploadStatus && <p className="muted">{uploadStatus}</p>}
        {docs.length > 0 && (
          <ul className="docs">
            {docs.map((d) => (
              <li key={d.id}>
                <span>
                  {d.name}
                  <span className="muted">
                    {" "}
                    · {d.pages} pages · {d.chunks} chunks
                  </span>
                  {d.stale && <span className="badge">embedded with {d.embedModel}; re-upload</span>}
                </span>
                <button type="button" className="ghost" onClick={() => void remove(d.id)}>
                  Delete
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>

      <div className="card">
        <label className="muted" style={{ display: "block", marginBottom: 12 }}>
          Ask about
          <select
            value={selected}
            onChange={(e) => setSelected(e.target.value)}
            style={{ display: "block", width: "100%", marginTop: 4 }}
          >
            <option value="">All documents</option>
            {searchable.map((d) => (
              <option key={d.id} value={d.id}>
                {d.name}
              </option>
            ))}
          </select>
        </label>

        {msgs.length === 0 && <p className="muted">Ask something like “Summarize the key findings.”</p>}
        {msgs.map((m, i) => (
          <div key={i} className="msg">
            <strong className={m.role === "user" ? "user" : ""}>
              {m.role === "user" ? "You" : "Assistant"}:
            </strong>{" "}
            {m.text || (m.role === "assistant" && loading ? "Thinking… (the first answer loads the model)" : "")}
            {m.sources?.map((s) => (
              <div key={s.n} className="src">
                [{s.n}] {s.document}, page {s.page} (score {s.score.toFixed(2)}): {s.snippet}
                {s.snippet.length >= 300 ? "…" : ""}
              </div>
            ))}
            {m.timing && <div className="timing">{m.timing}</div>}
          </div>
        ))}
        {error && <p className="error">{error}</p>}
        <form onSubmit={ask} className="row">
          <input
            type="text"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Ask a question about your documents"
          />
          <button disabled={loading}>Ask</button>
        </form>
      </div>
    </main>
  );
}
