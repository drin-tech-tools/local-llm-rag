import { chunkPages } from "@/lib/chunk";
import { config } from "@/lib/config";
import { embed, ModelServerError } from "@/lib/llm";
import { extractPages } from "@/lib/pdf";
import { addDocument } from "@/lib/store";

export const runtime = "nodejs";

const MAX_BYTES = 25 * 1024 * 1024;

export async function POST(req: Request) {
  try {
    const form = await req.formData();
    const file = form.get("file");
    if (!(file instanceof File) || file.type !== "application/pdf")
      return Response.json({ error: "Please upload a PDF file" }, { status: 400 });
    if (file.size > MAX_BYTES)
      return Response.json({ error: "Max 25 MB per PDF" }, { status: 400 });

    const started = performance.now();
    const pages = await extractPages(Buffer.from(await file.arrayBuffer()));
    const chunks = chunkPages(pages);
    if (!chunks.length)
      return Response.json({ error: "No extractable text (scanned PDF?)" }, { status: 422 });

    const embedStart = performance.now();
    const vectors = await embed(
      chunks.map((c) => c.content),
      "document",
      req.signal,
    );
    const embedMs = Math.round(performance.now() - embedStart);

    const { embedModel } = config();
    const doc = await addDocument(
      { name: file.name, pages: pages.length, embedModel },
      chunks.map((c, i) => ({ ...c, vector: vectors[i] })),
    );
    const totalMs = Math.round(performance.now() - started);
    console.log(
      `[upload] ${doc.name} pages=${doc.pages} chunks=${doc.chunks} dims=${doc.dims} embed_ms=${embedMs} total_ms=${totalMs}`,
    );

    return Response.json({ document: doc, embedMs, totalMs });
  } catch (e: unknown) {
    const message = e instanceof Error ? e.message : "Upload failed";
    console.error("[upload] error:", message);
    return Response.json({ error: message }, { status: e instanceof ModelServerError ? 502 : 500 });
  }
}
