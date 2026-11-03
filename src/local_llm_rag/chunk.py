"""Split pages into overlapping word windows; each chunk keeps its page number."""

from dataclasses import dataclass


@dataclass
class Chunk:
    page: int
    index: int
    content: str


def chunk_pages(pages: list[str], size: int = 200, overlap: int = 40) -> list[Chunk]:
    chunks: list[Chunk] = []
    index = 0
    for page_number, text in enumerate(pages, start=1):
        words = text.split()
        start = 0
        while start < len(words):
            content = " ".join(words[start : start + size])
            if len(content) > 20:
                chunks.append(Chunk(page=page_number, index=index, content=content))
                index += 1
            if start + size >= len(words):
                break
            start += size - overlap
    return chunks
