/** Pure vector helpers: similarity, ranking and compact serialization. */

export function norm(v: ArrayLike<number>): number {
  let s = 0;
  for (let i = 0; i < v.length; i++) s += v[i] * v[i];
  return Math.sqrt(s);
}

export function cosine(a: ArrayLike<number>, b: ArrayLike<number>, normA = norm(a), normB = norm(b)) {
  if (a.length !== b.length) throw new Error(`Vector length mismatch: ${a.length} vs ${b.length}`);
  if (!normA || !normB) return 0;
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
  return dot / (normA * normB);
}

/** Highest-scoring `k` items, best first. */
export function topK<T>(items: T[], score: (item: T) => number, k: number): Array<T & { score: number }> {
  return items
    .map((item) => ({ ...item, score: score(item) }))
    .sort((x, y) => y.score - x.score)
    .slice(0, k);
}

/** Float32 + base64 keeps a 768-dim vector at ~4 KB instead of ~15 KB of JSON numbers. */
export function encodeVector(v: ArrayLike<number>): string {
  return Buffer.from(Float32Array.from(v as ArrayLike<number>).buffer).toString("base64");
}

export function decodeVector(s: string): Float32Array {
  // Copy into a fresh buffer: Buffer.from may hand back a pooled slice whose
  // byteOffset is not 4-aligned, which Float32Array rejects.
  const bytes = new Uint8Array(Buffer.from(s, "base64"));
  return new Float32Array(bytes.buffer);
}
