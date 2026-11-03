"""Document store: a SQLite table of documents plus Chroma collections for the vectors.

Each embedding model gets its own Chroma collection, because vectors from different models
are not comparable (and often differ in dimension). Documents embedded with a model other
than the current EMBED_MODEL are reported as stale and are not searched.
"""

import re
import sqlite3
import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings


@dataclass
class Document:
    id: str
    name: str
    pages: int
    chunks: int
    embed_model: str
    dims: int
    created_at: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ChunkInput:
    page: int
    index: int
    content: str
    vector: list[float]


@dataclass
class Match:
    document_id: str
    document_name: str
    page: int
    content: str
    score: float


def collection_name(embed_model: str) -> str:
    """Chroma names allow [a-zA-Z0-9._-], 3-512 chars, alphanumeric at both ends."""
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", embed_model).strip("._-") or "default"
    return f"chunks-{slug}"[:512]


class Store:
    def __init__(self, data_dir: Path):
        data_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(data_dir / "documents.sqlite3", check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute(
            """CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                pages INTEGER NOT NULL,
                chunks INTEGER NOT NULL,
                embed_model TEXT NOT NULL,
                dims INTEGER NOT NULL,
                created_at TEXT NOT NULL
            )"""
        )
        self._db.commit()
        self._chroma = chromadb.PersistentClient(
            path=str(data_dir / "chroma"),
            settings=ChromaSettings(anonymized_telemetry=False),
        )

    def _collection(self, embed_model: str):
        # We always pass our own vectors, so no embedding function (Chroma would otherwise
        # download its default model on first use).
        return self._chroma.get_or_create_collection(
            collection_name(embed_model),
            embedding_function=None,
            configuration={"hnsw": {"space": "cosine"}},
        )

    def list_documents(self) -> list[Document]:
        with self._lock:
            rows = self._db.execute("SELECT * FROM documents ORDER BY created_at DESC").fetchall()
        return [Document(**dict(r)) for r in rows]

    def add_document(
        self, name: str, pages: int, embed_model: str, chunks: list[ChunkInput]
    ) -> Document:
        if not chunks:
            raise ValueError("A document needs at least one chunk")
        doc = Document(
            id=str(uuid.uuid4()),
            name=name,
            pages=pages,
            chunks=len(chunks),
            embed_model=embed_model,
            dims=len(chunks[0].vector),
            created_at=datetime.now(UTC).isoformat(),
        )
        with self._lock:
            self._collection(embed_model).add(
                ids=[f"{doc.id}:{c.index}" for c in chunks],
                embeddings=[c.vector for c in chunks],
                documents=[c.content for c in chunks],
                metadatas=[
                    {"document_id": doc.id, "page": c.page, "index": c.index} for c in chunks
                ],
            )
            self._db.execute(
                "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    doc.id,
                    doc.name,
                    doc.pages,
                    doc.chunks,
                    doc.embed_model,
                    doc.dims,
                    doc.created_at,
                ),
            )
            self._db.commit()
        return doc

    def delete_document(self, document_id: str) -> bool:
        with self._lock:
            row = self._db.execute(
                "SELECT embed_model FROM documents WHERE id = ?", (document_id,)
            ).fetchone()
            if not row:
                return False
            self._collection(row["embed_model"]).delete(where={"document_id": document_id})
            self._db.execute("DELETE FROM documents WHERE id = ?", (document_id,))
            self._db.commit()
        return True

    def search(
        self, query: list[float], embed_model: str, k: int, document_id: str | None = None
    ) -> list[Match]:
        """Top-k chunks by cosine similarity among documents embedded with `embed_model`."""
        with self._lock:
            names = {
                r["id"]: r["name"]
                for r in self._db.execute(
                    "SELECT id, name FROM documents WHERE embed_model = ?", (embed_model,)
                )
            }
            if not names or (document_id and document_id not in names):
                return []
            collection = self._collection(embed_model)
            n = min(k, collection.count())
            if n == 0:
                return []
            res = collection.query(
                query_embeddings=[query],
                n_results=n,
                where={"document_id": document_id} if document_id else None,
                include=["documents", "metadatas", "distances"],
            )
        matches = []
        for content, meta, distance in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0], strict=True
        ):
            doc_id = str(meta["document_id"])
            if doc_id not in names:
                continue
            matches.append(
                Match(
                    document_id=doc_id,
                    document_name=names[doc_id],
                    page=int(meta["page"]),
                    content=content,
                    score=1 - float(distance),
                )
            )
        return matches
