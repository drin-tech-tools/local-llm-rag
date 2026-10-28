/**
 * File-backed vector store: one JSON file under DATA_DIR, loaded into memory and searched
 * by brute-force cosine similarity. Plenty for a POC with a few thousand chunks; swap for
 * pgvector when documents grow (see docs/ARCHITECTURE.md).
 */
import { randomUUID } from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";
import { config } from "./config";
import { cosine, decodeVector, encodeVector, norm, topK } from "./vector";

export interface DocumentRecord {
  id: string;
  name: string;
  pages: number;
  chunks: number;
  embedModel: string;
  dims: number;
  createdAt: string;
}

interface StoredChunk {
  documentId: string;
  page: number;
  index: number;
  content: string;
  embedding: string; // base64 Float32
}

interface StoreFile {
  version: 1;
  documents: DocumentRecord[];
  chunks: StoredChunk[];
}

interface LoadedChunk extends Omit<StoredChunk, "embedding"> {
  vector: Float32Array;
  norm: number;
}

interface State {
  documents: DocumentRecord[];
  chunks: LoadedChunk[];
  mtimeMs: number;
}

export interface Match {
  documentId: string;
  documentName: string;
  page: number;
  content: string;
  score: number;
}

// Survives Next.js dev hot reloads, which re-evaluate this module.
const g = globalThis as unknown as { __ragStore?: State; __ragStoreLock?: Promise<unknown> };

function storePath() {
  return path.join(config().dataDir, "store.json");
}

async function load(): Promise<State> {
  const file = storePath();
  let mtimeMs = 0;
  try {
    mtimeMs = (await fs.stat(file)).mtimeMs;
  } catch {
    return (g.__ragStore = { documents: [], chunks: [], mtimeMs: 0 });
  }
  if (g.__ragStore && g.__ragStore.mtimeMs === mtimeMs) return g.__ragStore;

  const parsed = JSON.parse(await fs.readFile(file, "utf8")) as StoreFile;
  const chunks = parsed.chunks.map(({ embedding, ...rest }) => {
    const vector = decodeVector(embedding);
    return { ...rest, vector, norm: norm(vector) };
  });
  return (g.__ragStore = { documents: parsed.documents, chunks, mtimeMs });
}

async function save(state: State): Promise<void> {
  const file = storePath();
  await fs.mkdir(path.dirname(file), { recursive: true });
  const out: StoreFile = {
    version: 1,
    documents: state.documents,
    chunks: state.chunks.map(({ vector, norm: _n, ...rest }) => ({
      ...rest,
      embedding: encodeVector(vector),
    })),
  };
  // Write then rename so a crash mid-write never leaves a half-written store.
  const tmp = `${file}.${process.pid}.tmp`;
  await fs.writeFile(tmp, JSON.stringify(out));
  await fs.rename(tmp, file);
  state.mtimeMs = (await fs.stat(file)).mtimeMs;
  g.__ragStore = state;
}

/** Serialize writes so two uploads finishing together cannot drop each other's data. */
function withLock<T>(fn: () => Promise<T>): Promise<T> {
  const run = (g.__ragStoreLock ?? Promise.resolve()).then(fn, fn);
  g.__ragStoreLock = run.catch(() => {});
  return run;
}

export async function listDocuments(): Promise<DocumentRecord[]> {
  const state = await load();
  return [...state.documents].sort((a, b) => b.createdAt.localeCompare(a.createdAt));
}

export async function addDocument(
  doc: Omit<DocumentRecord, "id" | "createdAt" | "chunks" | "dims">,
  chunks: Array<{ page: number; index: number; content: string; vector: number[] }>,
): Promise<DocumentRecord> {
  return withLock(async () => {
    const state = await load();
    const record: DocumentRecord = {
      ...doc,
      id: randomUUID(),
      chunks: chunks.length,
      dims: chunks[0]?.vector.length ?? 0,
      createdAt: new Date().toISOString(),
    };
    const loaded = chunks.map((c) => {
      const vector = Float32Array.from(c.vector);
      return { documentId: record.id, page: c.page, index: c.index, content: c.content, vector, norm: norm(vector) };
    });
    await save({
      documents: [...state.documents, record],
      chunks: [...state.chunks, ...loaded],
      mtimeMs: state.mtimeMs,
    });
    return record;
  });
}

export async function deleteDocument(id: string): Promise<boolean> {
  return withLock(async () => {
    const state = await load();
    if (!state.documents.some((d) => d.id === id)) return false;
    await save({
      documents: state.documents.filter((d) => d.id !== id),
      chunks: state.chunks.filter((c) => c.documentId !== id),
      mtimeMs: state.mtimeMs,
    });
    return true;
  });
}

/**
 * Top-k chunks by cosine similarity. Only documents embedded with `embedModel` are
 * searched: vectors from different models are not comparable.
 */
export async function search(
  query: number[],
  opts: { embedModel: string; k: number; documentId?: string | null },
): Promise<Match[]> {
  const state = await load();
  const docs = new Map(
    state.documents
      .filter((d) => d.embedModel === opts.embedModel && (!opts.documentId || d.id === opts.documentId))
      .map((d) => [d.id, d]),
  );
  const candidates = state.chunks.filter(
    (c) => docs.has(c.documentId) && c.vector.length === query.length,
  );
  const qNorm = norm(query);
  return topK(candidates, (c) => cosine(query, c.vector, qNorm, c.norm), opts.k).map((c) => ({
    documentId: c.documentId,
    documentName: docs.get(c.documentId)!.name,
    page: c.page,
    content: c.content,
    score: c.score,
  }));
}
