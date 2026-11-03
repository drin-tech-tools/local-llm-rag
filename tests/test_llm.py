import json

import httpx
import pytest
from openai import OpenAI

from local_llm_rag import llm


def client_for(handler) -> OpenAI:
    return OpenAI(
        base_url="http://model.test/v1",
        api_key="x",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def test_has_model_matches_ollama_latest_tag():
    available = ["nomic-embed-text:latest", "llama3.1:8b"]
    assert llm.has_model(available, "nomic-embed-text")
    assert llm.has_model(available, "llama3.1:8b")
    assert not llm.has_model(available, "llama3.2:3b")


def test_task_prefix_only_for_nomic():
    assert llm.with_task_prefix("nomic-embed-text", "hi", "query") == "search_query: hi"
    assert llm.with_task_prefix("nomic-embed-text:v1.5", "hi", "document") == "search_document: hi"
    assert llm.with_task_prefix("mxbai-embed-large", "hi", "query") == "hi"


def test_embed_batches_prefixes_and_restores_order():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(body)
        data = [
            {"object": "embedding", "index": i, "embedding": [len(t), i]}
            for i, t in enumerate(body["input"])
        ]
        data.reverse()  # out of order on purpose
        return httpx.Response(200, json={"object": "list", "data": data, "model": body["model"]})

    texts = [f"t{i}" for i in range(40)]
    out = llm.embed(texts, "document", client=client_for(handler))
    assert len(calls) == 2
    assert calls[0]["model"] == "nomic-embed-text"
    assert calls[0]["encoding_format"] == "float"
    assert calls[0]["input"][0] == "search_document: t0"
    assert len(out) == 40
    assert out[33] == [len("search_document: t33"), 1]


def test_unreachable_server_is_explained():
    def handler(request):
        raise httpx.ConnectError("refused")

    with pytest.raises(llm.ModelServerError, match="Cannot reach the model server"):
        llm.embed(["x"], "query", client=client_for(handler))


def test_missing_model_suggests_pull():
    def handler(request):
        return httpx.Response(
            404, json={"error": {"message": 'model "nomic-embed-text" not found'}}
        )

    with pytest.raises(llm.ModelServerError, match=r"404.*ollama pull"):
        llm.embed(["x"], "query", client=client_for(handler))


def test_timeout_maps_to_504():
    def handler(request):
        raise httpx.ReadTimeout("slow")

    with pytest.raises(llm.ModelServerError) as exc:
        llm.chat_stream([{"role": "user", "content": "hi"}], client=client_for(handler))
    assert exc.value.status_code == 504


def sse(*events: str) -> httpx.Response:
    body = "".join(f"data: {e}\n\n" for e in events)
    return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})


def chunk(content: str | None) -> str:
    delta = {"content": content} if content is not None else {"role": "assistant"}
    return json.dumps(
        {
            "id": "c1",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "m",
            "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
        }
    )


def test_chat_stream_yields_content_deltas():
    def handler(request):
        assert json.loads(request.content)["model"] == "llama3.1:8b"
        return sse(chunk(None), chunk("Hello"), chunk(" world"), "[DONE]")

    tokens = llm.chat_stream([{"role": "user", "content": "hi"}], client=client_for(handler))
    assert "".join(tokens) == "Hello world"


def test_error_inside_stream_is_raised():
    def handler(request):
        return sse(chunk("Hi"), json.dumps({"error": {"message": "out of memory"}}))

    tokens = llm.chat_stream([{"role": "user", "content": "hi"}], client=client_for(handler))
    with pytest.raises(llm.ModelServerError, match="out of memory"):
        list(tokens)
