import base64
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from local_llm_rag import llm, main
from local_llm_rag.store import Store

SAMPLE_PDF = Path(__file__).parent / "fixtures" / "sample-report.pdf"


def fake_embed(texts, task, client=None):
    """Bag-of-words hashing: shared words give similar vectors, which is enough to rank."""
    vectors = []
    for text in texts:
        v = [0.0] * 64
        for word in text.lower().replace(":", " ").split():
            v[sum(map(ord, word.strip(".,"))) % 64] += 1
        vectors.append(v)
    return vectors


@pytest.fixture
def client(monkeypatch, tmp_path):
    store = Store(tmp_path / "store")
    main.app.dependency_overrides[main.get_store] = lambda: store
    monkeypatch.setattr(llm, "embed", fake_embed)
    monkeypatch.setattr(
        llm, "list_models", lambda client=None: ["nomic-embed-text:latest", "llama3.1:8b"]
    )
    monkeypatch.setattr(
        llm, "chat_stream", lambda messages, client=None: iter(["The ", "answer ", "[1]."])
    )
    yield TestClient(main.app)
    main.app.dependency_overrides.clear()


def upload(client, path=SAMPLE_PDF, content_type="application/pdf"):
    with open(path, "rb") as f:
        return client.post("/api/upload", files={"file": (path.name, f, content_type)})


def test_index_page(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "Local LLM RAG" in res.text


def test_health_ok(client):
    body = client.get("/api/health").json()
    assert body["ok"] is True
    assert body["chat_model"] == {"name": "llama3.1:8b", "available": True}


def test_health_reports_missing_model_and_unreachable_server(client, monkeypatch):
    monkeypatch.setattr(llm, "list_models", lambda client=None: ["llama3.1:8b"])
    body = client.get("/api/health").json()
    assert body["ok"] is False
    assert body["embed_model"]["available"] is False

    def down(client=None):
        raise llm.ModelServerError("Cannot reach the model server")

    monkeypatch.setattr(llm, "list_models", down)
    body = client.get("/api/health").json()
    assert body["ok"] is False
    assert "Cannot reach" in body["error"]


def test_upload_then_chat_streams_answer_with_sources(client):
    res = upload(client)
    assert res.status_code == 200, res.text
    doc = res.json()["document"]
    assert (doc["pages"], doc["chunks"], doc["embed_model"]) == (3, 3, "nomic-embed-text")

    listed = client.get("/api/documents").json()["documents"]
    assert [d["stale"] for d in listed] == [False]

    res = client.post("/api/chat", json={"question": "bridge repair program pesos"})
    assert res.status_code == 200
    assert res.text == "The answer [1]."
    assert res.headers["x-chat-model"] == "llama3.1:8b"
    sources = json.loads(base64.b64decode(res.headers["x-rag-sources"]))
    assert sources[0]["page"] == 1
    assert "bridge repair" in sources[0]["snippet"]


def test_chat_without_documents(client):
    res = client.post("/api/chat", json={"question": "anything?"})
    assert res.json() == {"answer": "No documents indexed yet. Upload a PDF first.", "sources": []}


def test_chat_with_only_other_model_documents(client, monkeypatch):
    upload(client)
    monkeypatch.setenv("EMBED_MODEL", "mxbai-embed-large")
    main.get_settings.cache_clear()
    assert client.get("/api/documents").json()["documents"][0]["stale"] is True
    res = client.post("/api/chat", json={"question": "bridge?"})
    assert "different model than mxbai-embed-large" in res.json()["answer"]


def test_chat_model_error_is_json(client, monkeypatch):
    upload(client)

    def broken(messages, client=None):
        raise llm.ModelServerError("No answer within 120s", status_code=504)

    monkeypatch.setattr(llm, "chat_stream", broken)
    res = client.post("/api/chat", json={"question": "bridge?"})
    assert res.status_code == 504
    assert res.json() == {"error": "No answer within 120s"}


def test_rejects_non_pdf_and_empty_question(client, tmp_path):
    txt = tmp_path / "notes.txt"
    txt.write_text("hello")
    assert upload(client, txt, "text/plain").status_code == 400
    assert client.post("/api/chat", json={"question": ""}).status_code == 422
    assert client.post("/api/chat", json={"question": "   "}).status_code == 400


def test_corrupt_pdf_is_a_clear_400(client, tmp_path):
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"%PDF-1.4 not really")
    res = upload(client, bad)
    assert res.status_code in (400, 422)


def test_delete(client):
    doc_id = upload(client).json()["document"]["id"]
    assert client.delete(f"/api/documents/{doc_id}").json() == {"ok": True}
    assert client.delete(f"/api/documents/{doc_id}").status_code == 404
