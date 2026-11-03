"""The RAG prompt: system rules plus one user turn with numbered sources and the question."""

from .store import Match

SYSTEM_PROMPT = """You answer questions using ONLY the numbered sources provided.
Cite sources inline like [1] or [2][3] right after each claim.
If the sources do not contain the answer, say you don't know. Do not use outside knowledge."""


def build_messages(question: str, sources: list[Match]) -> list[dict[str, str]]:
    context = "\n\n".join(
        f"[{i}] ({s.document_name}, p.{s.page})\n{s.content}"
        for i, s in enumerate(sources, start=1)
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Sources:\n{context}\n\nQuestion: {question}"},
    ]
