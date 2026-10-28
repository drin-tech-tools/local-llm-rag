export interface Chunk { page: number; index: number; content: string }

/** Split each page into overlapping word windows so citations map to pages. */
export function chunkPages(pages: string[], size = 200, overlap = 40): Chunk[] {
  const chunks: Chunk[] = [];
  let index = 0;
  pages.forEach((text, p) => {
    const words = text.replace(/\s+/g, " ").trim().split(" ").filter(Boolean);
    for (let start = 0; start < words.length; start += size - overlap) {
      const content = words.slice(start, start + size).join(" ");
      if (content.length > 20) chunks.push({ page: p + 1, index: index++, content });
      if (start + size >= words.length) break;
    }
  });
  return chunks;
}
