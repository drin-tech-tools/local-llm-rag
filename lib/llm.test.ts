import assert from "node:assert/strict";
import { afterEach, describe, it } from "node:test";
import { chatStream, embed, errorMessage, hasModel, parseSse, withTaskPrefix } from "./llm";

const realFetch = globalThis.fetch;
afterEach(() => {
  globalThis.fetch = realFetch;
  delete process.env.EMBED_MODEL;
});

function sseResponse(lines: string[]): Response {
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      const enc = new TextEncoder();
      // Split mid-line to prove partial chunks are buffered.
      const text = lines.join("\n") + "\n";
      const mid = Math.floor(text.length / 2);
      controller.enqueue(enc.encode(text.slice(0, mid)));
      controller.enqueue(enc.encode(text.slice(mid)));
      controller.close();
    },
  });
  return new Response(body, { headers: { "Content-Type": "text/event-stream" } });
}

describe("parseSse", () => {
  it("returns complete data payloads and keeps the unfinished tail", () => {
    const { payloads, rest } = parseSse('data: {"a":1}\n\ndata: [DONE]\ndata: {"b"');
    assert.deepEqual(payloads, ['{"a":1}', "[DONE]"]);
    assert.equal(rest, 'data: {"b"');
  });
});

describe("hasModel", () => {
  it("matches Ollama's implicit :latest tag", () => {
    const available = ["nomic-embed-text:latest", "llama3.1:8b"];
    assert.equal(hasModel(available, "nomic-embed-text"), true);
    assert.equal(hasModel(available, "llama3.1:8b"), true);
    assert.equal(hasModel(available, "llama3.2:3b"), false);
  });
});

describe("withTaskPrefix", () => {
  it("adds nomic task prefixes and leaves other models alone", () => {
    assert.equal(withTaskPrefix("nomic-embed-text", "hi", "query"), "search_query: hi");
    assert.equal(withTaskPrefix("nomic-embed-text:v1.5", "hi", "document"), "search_document: hi");
    assert.equal(withTaskPrefix("mxbai-embed-large", "hi", "query"), "hi");
  });
});

describe("errorMessage", () => {
  it("reads OpenAI and Ollama error shapes", () => {
    assert.equal(errorMessage('{"error":{"message":"model not found"}}'), "model not found");
    assert.equal(errorMessage('{"error":"boom"}'), "boom");
    assert.equal(errorMessage("plain text"), "plain text");
  });
});

describe("embed", () => {
  it("batches requests, prefixes for nomic and restores input order", async () => {
    const calls: Array<{ model: string; input: string[] }> = [];
    globalThis.fetch = (async (_url: string, init: RequestInit) => {
      const body = JSON.parse(String(init.body)) as { model: string; input: string[] };
      calls.push(body);
      // Return out of order to check sorting by index.
      const data = body.input.map((t, index) => ({ index, embedding: [t.length, index] })).reverse();
      return Response.json({ data });
    }) as typeof fetch;

    const texts = Array.from({ length: 40 }, (_, i) => `t${i}`);
    const out = await embed(texts, "document");
    assert.equal(calls.length, 2);
    assert.equal(calls[0].model, "nomic-embed-text");
    assert.equal(calls[0].input[0], "search_document: t0");
    assert.equal(out.length, 40);
    assert.deepEqual(out[33], ["search_document: t33".length, 1]);
  });

  it("explains an unreachable server", async () => {
    globalThis.fetch = (async () => {
      throw new TypeError("fetch failed");
    }) as typeof fetch;
    await assert.rejects(() => embed(["x"], "query"), /Cannot reach the model server/);
  });

  it("suggests ollama pull when a model is missing", async () => {
    globalThis.fetch = (async () =>
      Response.json({ error: { message: 'model "nomic-embed-text" not found' } }, { status: 404 })) as typeof fetch;
    await assert.rejects(() => embed(["x"], "query"), /404.*ollama pull/);
  });
});

describe("chatStream", () => {
  it("yields content deltas until [DONE]", async () => {
    globalThis.fetch = (async () =>
      sseResponse([
        'data: {"choices":[{"delta":{"role":"assistant"}}]}',
        "",
        'data: {"choices":[{"delta":{"content":"Hello"}}]}',
        "",
        'data: {"choices":[{"delta":{"content":" world"}}]}',
        "",
        "data: [DONE]",
      ])) as typeof fetch;
    let text = "";
    for await (const t of await chatStream([{ role: "user", content: "hi" }], new AbortController().signal)) {
      text += t;
    }
    assert.equal(text, "Hello world");
  });

  it("surfaces an error sent inside the stream", async () => {
    globalThis.fetch = (async () => sseResponse(['data: {"error":{"message":"out of memory"}}'])) as typeof fetch;
    const tokens = await chatStream([{ role: "user", content: "hi" }], new AbortController().signal);
    await assert.rejects(async () => {
      for await (const _ of tokens) {
        /* drain */
      }
    }, /out of memory/);
  });
});
