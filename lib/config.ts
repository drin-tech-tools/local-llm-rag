import path from "node:path";

/** All settings come from env vars with defaults that work against a local Ollama install. */
export interface Config {
  baseUrl: string;
  apiKey?: string;
  embedModel: string;
  chatModel: string;
  topK: number;
  firstTokenTimeoutMs: number;
  dataDir: string;
}

function intEnv(name: string, fallback: number): number {
  const raw = process.env[name]?.trim();
  if (!raw) return fallback;
  const n = Number(raw);
  if (!Number.isFinite(n) || n <= 0) throw new Error(`${name} must be a positive number, got "${raw}"`);
  return Math.floor(n);
}

export function config(): Config {
  return {
    baseUrl: (process.env.LLM_BASE_URL?.trim() || "http://localhost:11434/v1").replace(/\/+$/, ""),
    apiKey: process.env.LLM_API_KEY?.trim() || undefined,
    embedModel: process.env.EMBED_MODEL?.trim() || "nomic-embed-text",
    chatModel: process.env.CHAT_MODEL?.trim() || "llama3.1:8b",
    topK: intEnv("RAG_TOP_K", 5),
    firstTokenTimeoutMs: intEnv("CHAT_FIRST_TOKEN_TIMEOUT_MS", 120_000),
    dataDir: path.resolve(process.env.DATA_DIR?.trim() || "data"),
  };
}
