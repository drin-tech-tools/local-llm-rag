# Architecture

The app talks to models only through an OpenAI-compatible HTTP API. Which server answers, Ollama on a laptop or the team's own model server, is configuration, not code.

## Components

| Component | Today | Later | Why |
| --- | --- | --- | --- |
| Web app and API | FastAPI on `127.0.0.1:8000`, one static HTML page | Same app, behind auth when shared | Python matches the team's model work; FastAPI streams answers and documents its API at `/docs` |
| Model client | `llm.py`: the `openai` package with `base_url` set to `LLM_BASE_URL` | Unchanged | Ollama, vLLM, llama.cpp server and LM Studio all serve this API |
| Embeddings | `nomic-embed-text` (768 dimensions) on Ollama | Team embedding model | Small, fast and strong for retrieval |
| Chat | `llama3.1:8b` on Ollama | Team LLM | Sized for a 16 GB Mac |
| Vectors | Chroma, embedded and stored on disk, one collection per embedding model | pgvector or a Chroma server | No database to run for the POC |
| Document list | SQLite table | Postgres | Standard library, no setup |

## Request flow

**Upload:** PDF → text per page (pypdf) → ~200-word chunks with 40-word overlap, each tagged with its page → batches of 32 sent to `/embeddings` with the `search_document:` prefix nomic expects → vectors added to the embedding model's Chroma collection, and the document recorded in SQLite with that model's name.

**Question:** question → `/embeddings` with the `search_query:` prefix → Chroma cosine search in the current model's collection, optionally filtered to one document → top `RAG_TOP_K` chunks → numbered sources plus the question sent to `/chat/completions` with streaming → tokens streamed to the browser. Sources travel in the `X-Rag-Sources` header (base64 JSON), and time to first token in `X-First-Token-Ms`.

The chat endpoint waits for the first token before it responds. A model error or a load timeout therefore comes back as a clear JSON error instead of a half-finished answer.

## Design decisions

- **OpenAI-compatible API through the official SDK.** The base URL is the only switch between servers. Embeddings are requested as floats explicitly, because the SDK otherwise asks for base64, which not every server supports.
- **One Chroma collection per embedding model.** Vectors from different models are not comparable and often differ in size. Documents from another model are flagged "re-upload" and are not searched, instead of silently returning bad matches.
- **Plain Python pipeline.** No LangChain or LlamaIndex: chunk, embed, search and prompt are each a few lines the team can read and change.
- **Synchronous endpoints.** FastAPI runs them in a thread pool, which is plenty for a single-user demo and keeps the code simple.
- **Localhost only.** No auth in the POC, so both the app and Ollama bind to `127.0.0.1`. The page builds all PDF- and user-provided text with `textContent`, never `innerHTML`.

## Moving past the POC

| Need | Change |
| --- | --- |
| Use the team's model server | Set `LLM_BASE_URL`, `LLM_API_KEY`, `CHAT_MODEL`, `EMBED_MODEL`. No code change. |
| More documents or several users | Move vectors to pgvector (or a Chroma server) and documents to Postgres behind the same `Store` methods. |
| Share it with the team | Add authentication, run it next to the model server, and put both behind HTTPS. Do not tunnel a laptop's Ollama to the internet. |
| Better answers | A larger chat model on the team server, then a re-ranking step over the top 20 chunks. |
| Measure quality | Add a small question/answer set and score retrieval and answers (for example with ragas) whenever the model or chunking changes. |
