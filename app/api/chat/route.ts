import { config } from "@/lib/config";
import { chatStream, embed, ModelServerError } from "@/lib/llm";
import { buildMessages } from "@/lib/prompt";
import { listDocuments, search } from "@/lib/store";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

function msSince(start: number) {
  return Math.round(performance.now() - start);
}

export async function POST(req: Request) {
  let body: { question?: unknown; documentId?: unknown };
  try {
    body = await req.json();
  } catch {
    return Response.json({ error: "Invalid JSON" }, { status: 400 });
  }
  const question = typeof body.question === "string" ? body.question.trim() : "";
  const documentId = typeof body.documentId === "string" && body.documentId ? body.documentId : null;
  if (!question) return Response.json({ error: "question is required" }, { status: 400 });

  const cfg = config();
  const started = performance.now();
  const abort = new AbortController();
  const onClientGone = () => abort.abort();
  req.signal.addEventListener("abort", onClientGone, { once: true });
  let timedOut = false;
  const timer = setTimeout(() => {
    timedOut = true;
    abort.abort();
  }, cfg.firstTokenTimeoutMs);

  try {
    const [queryVector] = await embed([question], "query", abort.signal);
    const embedMs = msSince(started);

    const sources = await search(queryVector, { embedModel: cfg.embedModel, k: cfg.topK, documentId });
    if (!sources.length) {
      clearTimeout(timer);
      const docs = await listDocuments();
      const answer = !docs.length
        ? "No documents indexed yet. Upload a PDF first."
        : docs.every((d) => d.embedModel !== cfg.embedModel) || documentId
          ? `These documents were embedded with a different model than ${cfg.embedModel}. Upload them again to search them.`
          : "No matching chunks found.";
      return Response.json({ answer, sources: [] });
    }

    const tokens = await chatStream(buildMessages(question, sources), abort.signal);
    // Wait for the first token before answering, so a model error still becomes a JSON error.
    const first = await tokens.next();
    clearTimeout(timer);
    const firstTokenMs = msSince(started);
    console.log(
      `[chat] model=${cfg.chatModel} embed_ms=${embedMs} first_token_ms=${firstTokenMs} sources=${sources.length}`,
    );

    const encoder = new TextEncoder();
    const stream = new ReadableStream<Uint8Array>({
      async start(controller) {
        try {
          if (!first.done && first.value) controller.enqueue(encoder.encode(first.value));
          for await (const text of tokens) controller.enqueue(encoder.encode(text));
          console.log(`[chat] total_ms=${msSince(started)}`);
        } catch (err) {
          if (!abort.signal.aborted) {
            const message = err instanceof Error ? err.message : "Generation failed";
            controller.enqueue(encoder.encode(`\n\n[Error] ${message}`));
          }
        } finally {
          req.signal.removeEventListener("abort", onClientGone);
          controller.close();
        }
      },
      cancel() {
        abort.abort();
      },
    });

    const payload = sources.map((s, i) => ({
      n: i + 1,
      document: s.documentName,
      page: s.page,
      score: s.score,
      snippet: s.content.slice(0, 300),
    }));
    return new Response(stream, {
      headers: {
        "Content-Type": "text/plain; charset=utf-8",
        "Cache-Control": "no-cache, no-transform",
        // base64(UTF-8 JSON): snippets may contain characters HTTP headers cannot carry.
        "X-Rag-Sources": Buffer.from(JSON.stringify(payload), "utf8").toString("base64"),
        "X-Chat-Model": cfg.chatModel,
        "X-First-Token-Ms": String(firstTokenMs),
      },
    });
  } catch (err) {
    clearTimeout(timer);
    req.signal.removeEventListener("abort", onClientGone);
    if (timedOut) {
      return Response.json(
        {
          error: `No answer from ${cfg.chatModel} within ${cfg.firstTokenTimeoutMs / 1000}s. The model may still be loading; try again, or use a smaller CHAT_MODEL.`,
        },
        { status: 504 },
      );
    }
    const message = err instanceof Error ? err.message : "Chat failed";
    console.error("[chat] error:", message);
    return Response.json({ error: message }, { status: err instanceof ModelServerError ? 502 : 500 });
  }
}
