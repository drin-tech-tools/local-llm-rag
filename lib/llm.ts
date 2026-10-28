/**
 * Client for OpenAI-compatible model servers: Ollama, vLLM, llama.cpp server, LM Studio.
 * Plain fetch, no SDK, so moving to another server is only an LLM_BASE_URL change.
 */
import { config } from "./config";

export type EmbedTask = "document" | "query";

/** Chunks per `/embeddings` request. */
const EMBED_BATCH = 32;

export class ModelServerError extends Error {
  constructor(
    message: string,
    public readonly status?: number,
  ) {
    super(message);
    this.name = "ModelServerError";
  }
}

function headers(): Record<string, string> {
  const { apiKey } = config();
  return {
    "Content-Type": "application/json",
    ...(apiKey ? { Authorization: `Bearer ${apiKey}` } : {}),
  };
}

/** Pull a readable message out of an error body (OpenAI `{error:{message}}` or Ollama `{error}`). */
export function errorMessage(body: string): string {
  try {
    const parsed = JSON.parse(body) as { error?: { message?: string } | string };
    if (typeof parsed.error === "string") return parsed.error;
    if (parsed.error?.message) return parsed.error.message;
  } catch {
    /* not JSON */
  }
  return body.trim();
}

async function call(
  method: "GET" | "POST",
  path: string,
  body?: unknown,
  signal?: AbortSignal,
): Promise<Response> {
  const { baseUrl } = config();
  let res: Response;
  try {
    res = await fetch(`${baseUrl}${path}`, {
      method,
      headers: headers(),
      body: body === undefined ? undefined : JSON.stringify(body),
      signal,
      // Next.js caches GET fetches by default; model server state must always be live.
      cache: "no-store",
    });
  } catch (err) {
    if (signal?.aborted) throw err;
    throw new ModelServerError(
      `Cannot reach the model server at ${baseUrl}. Is Ollama running? Open the Ollama app or run \`ollama serve\`.`,
    );
  }
  if (!res.ok) {
    const message = errorMessage(await res.text().catch(() => "")) || res.statusText;
    const hint = /not found/i.test(message) ? " Pull it with `ollama pull <model>`." : "";
    throw new ModelServerError(`Model server error ${res.status}: ${message}.${hint}`, res.status);
  }
  return res;
}

/** Model ids the server has available (`/models`). */
export async function listModels(signal?: AbortSignal): Promise<string[]> {
  const res = await call("GET", "/models", undefined, signal);
  const json = (await res.json()) as { data?: Array<{ id?: string }> };
  return (json.data ?? []).map((m) => m.id ?? "").filter(Boolean);
}

/** Ollama lists `nomic-embed-text:latest` for a model pulled as `nomic-embed-text`. */
export function hasModel(available: string[], wanted: string): boolean {
  const name = wanted.includes(":") ? wanted : `${wanted}:latest`;
  return available.some((id) => id === wanted || id === name);
}

/**
 * nomic-embed-text was trained with task prefixes; retrieval is noticeably worse without them.
 * Other models get the text unchanged.
 */
export function withTaskPrefix(model: string, text: string, task: EmbedTask): string {
  if (!model.startsWith("nomic-embed")) return text;
  return `${task === "query" ? "search_query" : "search_document"}: ${text}`;
}

/** Embed texts in batches; vectors come back in input order. */
export async function embed(
  texts: string[],
  task: EmbedTask,
  signal?: AbortSignal,
): Promise<number[][]> {
  const { embedModel } = config();
  const out: number[][] = [];
  for (let i = 0; i < texts.length; i += EMBED_BATCH) {
    const input = texts.slice(i, i + EMBED_BATCH).map((t) => withTaskPrefix(embedModel, t, task));
    const res = await call("POST", "/embeddings", { model: embedModel, input }, signal);
    const json = (await res.json()) as { data?: Array<{ embedding?: number[]; index?: number }> };
    const data = [...(json.data ?? [])].sort((a, b) => (a.index ?? 0) - (b.index ?? 0));
    if (data.length !== input.length) {
      throw new ModelServerError(`Expected ${input.length} embeddings, got ${data.length}`);
    }
    for (const d of data) {
      if (!d.embedding?.length) throw new ModelServerError("Embedding response missing values");
      out.push(d.embedding);
    }
  }
  return out;
}

export interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

/** Split a server-sent-events buffer into complete `data:` payloads plus the unfinished tail. */
export function parseSse(buffer: string): { payloads: string[]; rest: string } {
  const lines = buffer.split("\n");
  const rest = lines.pop() ?? "";
  const payloads: string[] = [];
  for (const raw of lines) {
    const line = raw.trim();
    if (line.startsWith("data:")) payloads.push(line.slice(5).trim());
  }
  return { payloads, rest };
}

/**
 * Stream answer text from `/chat/completions`. Resolves once the server accepts the
 * request; the returned generator yields text deltas until the answer ends.
 */
export async function chatStream(
  messages: ChatMessage[],
  signal: AbortSignal,
): Promise<AsyncGenerator<string, void, unknown>> {
  const { chatModel } = config();
  const res = await call(
    "POST",
    "/chat/completions",
    { model: chatModel, stream: true, temperature: 0.2, messages },
    signal,
  );
  if (!res.body) throw new ModelServerError("Model server returned no stream body");
  const reader = res.body.getReader();
  const decoder = new TextDecoder();

  async function* tokens(): AsyncGenerator<string, void, unknown> {
    let buffer = "";
    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) return;
        buffer += decoder.decode(value, { stream: true });
        const { payloads, rest } = parseSse(buffer);
        buffer = rest;
        for (const payload of payloads) {
          if (payload === "[DONE]") return;
          const chunk = JSON.parse(payload) as {
            error?: { message?: string };
            choices?: Array<{ delta?: { content?: string | null } }>;
          };
          if (chunk.error) throw new ModelServerError(chunk.error.message ?? "Stream error");
          const text = chunk.choices?.[0]?.delta?.content;
          if (text) yield text;
        }
      }
    } finally {
      reader.cancel().catch(() => {});
    }
  }

  return tokens();
}
