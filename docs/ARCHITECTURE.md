# Architecture

The app talks to models only through an OpenAI-compatible HTTP API. Which server answers, Ollama on a laptop or the team's own model server, is configuration, not code.

## Components

| Component | Today | Later | Why |
| --- | --- | --- | --- |
| UI and API | Next.js 14 app on `127.0.0.1:3000` | Same app, behind auth when shared | One codebase for the demo and the team version |
| Model client | `lib/llm.ts`: `/models`, `/embeddings`, `/chat/completions` over plain fetch | Unchanged | Ollama, vLLM, llama.cpp server and LM Studio all serve this API |
| Embeddings | `nomic-embed-text` (768 dimensions) on Ollama | Team embedding model | Small, fast and strong for retrieval |
| Chat | `llama3.1:8b` on Ollama | Team LLM | Sized for a 16 GB Mac |
| Vector store | `lib/store.ts`: one JSON file, brute-force cosine search in memory | pgvector | Zero setup for the POC |

## Request flow

**Upload:** PDF → text per page (`pdf-parse`) → ~200-word chunks with 40-word overlap, each tagged with its page → batches of 32 sent to `/embeddings` with the `search_document:` prefix nomic expects → saved with the embedding model's name.

**Question:** question → `/embeddings` with the `search_query:` prefix → cosine similarity against chunks from documents embedded with the *same* model → top `RAG_TOP_K` chunks → numbered sources plus the question sent to `/chat/completions` with streaming → tokens streamed to the browser. Sources travel in the `X-Rag-Sources` header (base64 JSON), and time to first token in `X-First-Token-Ms`.

The chat route waits for the first token before it responds. A model error or a load timeout therefore comes back as a clear JSON error instead of a half-finished answer.

## Design decisions

- **OpenAI-compatible API, no SDK.** About 150 lines of fetch code cover every server we care about, and the base URL is the only switch.
- **Embedding model stored per document.** Vectors from different models are not comparable. Documents from another model are flagged "re-upload" and excluded from search instead of silently returning bad matches.
- **File store first.** At POC scale (hundreds of pages), brute-force search over a few thousand 768-dimension vectors takes milliseconds. Vectors are stored as base64 Float32 (~4 KB each) to keep the file small.
- **Localhost only.** No auth in the POC, so both the app and Ollama bind to `127.0.0.1`.

## Moving past the POC

| Need | Change |
| --- | --- |
| Use the team's model server | Set `LLM_BASE_URL`, `LLM_API_KEY`, `CHAT_MODEL`, `EMBED_MODEL`. No code change. |
| More documents or several users | Replace `lib/store.ts` with pgvector: a `chunks` table with `embedding vector(768)` and an HNSW index, keeping the same `search()` signature. |
| Share it with the team | Add authentication, run it next to the model server, and put both behind HTTPS. Do not tunnel a laptop's Ollama to the internet. |
| Better answers | A larger chat model on the team server, then a re-ranking step over the top 20 chunks. |
