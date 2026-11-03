"""FastAPI app: the web page plus a small JSON/streaming API."""

import base64
import json
import logging
import time
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from pypdf.errors import PdfReadError

from . import llm
from .chunk import chunk_pages
from .config import Settings, get_settings
from .pdf import extract_pages
from .prompt import build_messages
from .store import ChunkInput, Store

log = logging.getLogger("local_llm_rag")
logging.basicConfig(level=logging.INFO, format="%(levelname)s:     %(message)s")
# The openai SDK's httpx client logs every request at INFO; our own timing lines are enough.
logging.getLogger("httpx").setLevel(logging.WARNING)

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
INDEX_HTML = Path(__file__).parent / "static" / "index.html"

app = FastAPI(title="Local LLM RAG", version="0.1.0")


@lru_cache
def get_store() -> Store:
    return Store(get_settings().data_dir)


SettingsDep = Annotated[Settings, Depends(get_settings)]
StoreDep = Annotated[Store, Depends(get_store)]


def ms_since(start: float) -> int:
    return round((time.perf_counter() - start) * 1000)


@app.exception_handler(llm.ModelServerError)
def model_server_error(_request, err: llm.ModelServerError):
    log.error("model server: %s", err)
    return JSONResponse({"error": str(err)}, status_code=err.status_code)


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(INDEX_HTML)


@app.get("/api/health")
def health(settings: SettingsDep) -> dict:
    """Is the model server up, and are both configured models pulled?"""
    result = {
        "ok": False,
        "base_url": settings.base_url,
        "chat_model": {"name": settings.chat_model, "available": False},
        "embed_model": {"name": settings.embed_model, "available": False},
    }
    try:
        available = llm.list_models()
    except llm.ModelServerError as err:
        return {**result, "error": str(err)}
    result["chat_model"]["available"] = llm.has_model(available, settings.chat_model)
    result["embed_model"]["available"] = llm.has_model(available, settings.embed_model)
    result["ok"] = result["chat_model"]["available"] and result["embed_model"]["available"]
    return result


@app.get("/api/documents")
def documents(settings: SettingsDep, store: StoreDep) -> dict:
    return {
        "documents": [
            # Embedded with another model: not searchable until uploaded again.
            {**d.to_dict(), "stale": d.embed_model != settings.embed_model}
            for d in store.list_documents()
        ]
    }


@app.delete("/api/documents/{document_id}")
def delete_document(document_id: str, store: StoreDep) -> dict:
    if not store.delete_document(document_id):
        raise HTTPException(404, "Document not found")
    return {"ok": True}


@app.post("/api/upload")
def upload(file: Annotated[UploadFile, File()], settings: SettingsDep, store: StoreDep) -> dict:
    """PDF → text per page → chunks → embeddings → store."""
    if file.content_type != "application/pdf" and not (file.filename or "").lower().endswith(
        ".pdf"
    ):
        raise HTTPException(400, "Please upload a PDF file")
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(400, "Max 25 MB per PDF")

    started = time.perf_counter()
    try:
        pages = extract_pages(data)
    except PdfReadError as err:
        raise HTTPException(400, f"Could not read this PDF: {err}") from err
    chunks = chunk_pages(pages)
    if not chunks:
        raise HTTPException(422, "No extractable text (scanned PDF?)")

    embed_started = time.perf_counter()
    vectors = llm.embed([c.content for c in chunks], "document")
    embed_ms = ms_since(embed_started)

    doc = store.add_document(
        name=file.filename or "document.pdf",
        pages=len(pages),
        embed_model=settings.embed_model,
        chunks=[
            ChunkInput(c.page, c.index, c.content, v) for c, v in zip(chunks, vectors, strict=True)
        ],
    )
    total_ms = ms_since(started)
    log.info(
        "upload %s pages=%d chunks=%d dims=%d embed_ms=%d total_ms=%d",
        doc.name,
        doc.pages,
        doc.chunks,
        doc.dims,
        embed_ms,
        total_ms,
    )
    return {"document": doc.to_dict(), "embed_ms": embed_ms, "total_ms": total_ms}


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    document_id: str | None = None


@app.post("/api/chat", response_model=None)
def chat(req: ChatRequest, settings: SettingsDep, store: StoreDep) -> StreamingResponse | dict:
    """Embed the question, retrieve chunks, then stream the answer as plain text.

    Sources travel in the X-Rag-Sources header (base64 JSON) so the body is only the answer.
    """
    question = req.question.strip()
    if not question:
        raise HTTPException(400, "question is required")
    started = time.perf_counter()

    [query_vector] = llm.embed([question], "query")
    embed_ms = ms_since(started)
    sources = store.search(query_vector, settings.embed_model, settings.rag_top_k, req.document_id)

    if not sources:
        docs = store.list_documents()
        if not docs:
            answer = "No documents indexed yet. Upload a PDF first."
        elif req.document_id or all(d.embed_model != settings.embed_model for d in docs):
            answer = (
                f"These documents were embedded with a different model than "
                f"{settings.embed_model}. Upload them again to search them."
            )
        else:
            answer = "No matching chunks found."
        return {"answer": answer, "sources": []}

    tokens = llm.chat_stream(build_messages(question, sources))
    # Wait for the first token before responding, so a model error or load timeout
    # still becomes a clean JSON error instead of a half-finished answer.
    first = next(tokens, "")
    first_token_ms = ms_since(started)
    log.info(
        "chat model=%s embed_ms=%d first_token_ms=%d sources=%d",
        settings.chat_model,
        embed_ms,
        first_token_ms,
        len(sources),
    )

    def body() -> Iterator[str]:
        yield first
        try:
            yield from tokens
            log.info("chat total_ms=%d", ms_since(started))
        except llm.ModelServerError as err:
            yield f"\n\n[Error] {err}"

    payload = [
        {
            "n": i,
            "document": s.document_name,
            "page": s.page,
            "score": s.score,
            "snippet": s.content[:300],
        }
        for i, s in enumerate(sources, start=1)
    ]
    return StreamingResponse(
        body(),
        media_type="text/plain; charset=utf-8",
        headers={
            "Cache-Control": "no-cache, no-transform",
            # base64(UTF-8 JSON): snippets may contain characters HTTP headers cannot carry.
            "X-Rag-Sources": base64.b64encode(json.dumps(payload).encode()).decode(),
            "X-Chat-Model": settings.chat_model,
            "X-First-Token-Ms": str(first_token_ms),
        },
    )
