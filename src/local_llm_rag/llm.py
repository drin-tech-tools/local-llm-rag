"""Client for OpenAI-compatible model servers: Ollama, vLLM, llama.cpp server, LM Studio.

Uses the official `openai` package pointed at LLM_BASE_URL, so switching servers is a
configuration change, not a code change.
"""

from collections.abc import Iterator
from typing import Literal

import openai
from openai import OpenAI

from .config import get_settings

EmbedTask = Literal["document", "query"]

# Chunks per /embeddings request.
EMBED_BATCH = 32


class ModelServerError(Exception):
    """A model server problem, with a message that says what to do about it."""

    def __init__(self, message: str, status_code: int = 502):
        super().__init__(message)
        self.status_code = status_code


def make_client(timeout: float | None = None) -> OpenAI:
    s = get_settings()
    return OpenAI(
        base_url=s.base_url,
        # The SDK insists on a key; Ollama accepts any value.
        api_key=s.llm_api_key or "ollama",
        timeout=timeout if timeout is not None else s.chat_timeout_seconds,
        max_retries=0,
    )


def _explain(err: Exception) -> ModelServerError:
    s = get_settings()
    if isinstance(err, openai.APITimeoutError):
        return ModelServerError(
            f"No answer from {s.base_url} within {s.chat_timeout_seconds:.0f}s. The model may "
            "still be loading; try again, or use a smaller CHAT_MODEL.",
            status_code=504,
        )
    if isinstance(err, openai.APIConnectionError):
        return ModelServerError(
            f"Cannot reach the model server at {s.base_url}. Is Ollama running? "
            "Open the Ollama app or run `ollama serve`."
        )
    if isinstance(err, openai.APIStatusError):
        message = _status_message(err)
        hint = " Pull it with `ollama pull <model>`." if "not found" in message.lower() else ""
        return ModelServerError(f"Model server error {err.status_code}: {message}.{hint}")
    return ModelServerError(str(err))


def _status_message(err: "openai.APIStatusError") -> str:
    body = err.body
    if isinstance(body, dict):
        inner = body.get("error", body)
        if isinstance(inner, dict) and inner.get("message"):
            return str(inner["message"])
        if isinstance(body.get("message"), str):
            return body["message"]
        if isinstance(inner, str):
            return inner
    return err.message


def list_models(client: OpenAI | None = None) -> list[str]:
    try:
        return [m.id for m in (client or make_client(timeout=5)).models.list()]
    except openai.OpenAIError as err:
        raise _explain(err) from err


def has_model(available: list[str], wanted: str) -> bool:
    """Ollama lists `nomic-embed-text:latest` for a model pulled as `nomic-embed-text`."""
    tagged = wanted if ":" in wanted else f"{wanted}:latest"
    return wanted in available or tagged in available


def with_task_prefix(model: str, text: str, task: EmbedTask) -> str:
    """nomic-embed-text was trained with task prefixes; retrieval is worse without them."""
    if not model.startswith("nomic-embed"):
        return text
    return f"{'search_query' if task == 'query' else 'search_document'}: {text}"


def embed(texts: list[str], task: EmbedTask, client: OpenAI | None = None) -> list[list[float]]:
    """Embed texts in batches; vectors come back in input order."""
    model = get_settings().embed_model
    client = client or make_client()
    out: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch = [with_task_prefix(model, t, task) for t in texts[i : i + EMBED_BATCH]]
        try:
            # The SDK asks for base64 unless told otherwise; not every server supports it.
            res = client.embeddings.create(model=model, input=batch, encoding_format="float")
        except openai.OpenAIError as err:
            raise _explain(err) from err
        data = sorted(res.data, key=lambda d: d.index)
        if len(data) != len(batch):
            raise ModelServerError(f"Expected {len(batch)} embeddings, got {len(data)}")
        out.extend(d.embedding for d in data)
    return out


def chat_stream(messages: list[dict[str, str]], client: OpenAI | None = None) -> Iterator[str]:
    """Yield answer text as it is generated.

    The request is sent when this is called, so connection and model errors surface here
    rather than on the first `next()`.
    """
    s = get_settings()
    try:
        stream = (client or make_client()).chat.completions.create(
            model=s.chat_model, messages=messages, stream=True, temperature=0.2
        )
    except openai.OpenAIError as err:
        raise _explain(err) from err

    def tokens() -> Iterator[str]:
        try:
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except openai.OpenAIError as err:
            raise _explain(err) from err
        finally:
            stream.close()

    return tokens()
