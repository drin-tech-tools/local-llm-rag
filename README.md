# Local LLM RAG

Chat with your PDFs using models that run on your own machine. Answers stream in and cite the source pages. No cloud AI provider, no API keys, no rate limits.

It is a Python proof of concept for retrieval-augmented generation (RAG) on self-hosted models. Every model call goes through an **OpenAI-compatible API**, so the same app runs against [Ollama](https://ollama.com) on a laptop today and against a team model server (vLLM, llama.cpp server) later by changing one URL.

```
Browser ──► FastAPI app ──► OpenAI-compatible API ──► Ollama on your Mac (now)
               │                (LLM_BASE_URL)    └─► team vLLM / llama.cpp server (later)
               └──► Chroma vectors + SQLite document list (data/)
```

## What it does

- **Upload a PDF.** Text is extracted per page with pypdf, split into overlapping ~200-word chunks, and embedded with `nomic-embed-text` (768 dimensions).
- **Ask a question.** The question is embedded, Chroma finds the closest chunks by cosine similarity, and a local chat model (default `llama3.1:8b`) answers using only those chunks, citing them as [1], [2]…
- **See what's happening.** A status bar shows whether the model server is up and both models are pulled. Each answer shows time to first token and total time.

## Quick start

Full requirements and every step, including installing Ollama and uv on a Mac, are in **[docs/LOCAL_SETUP.md](docs/LOCAL_SETUP.md)**. In short:

```bash
# 1. Ollama running, with both models pulled
ollama pull nomic-embed-text
ollama pull llama3.1:8b

# 2. The app (uv installs Python 3.12 and the dependencies)
git clone https://github.com/drin-tech-tools/local-llm-rag.git
cd local-llm-rag
uv sync
uv run doctor    # checks your machine and the models, and warms the chat model
uv run serve     # http://127.0.0.1:8000
```

No `.env` file is needed for a default Ollama install. To change models or servers, copy `.env.example` to `.env.local`.

## Stack

| Layer | Choice |
| --- | --- |
| Web app and API | [FastAPI](https://fastapi.tiangolo.com) + Uvicorn, one HTML page with vanilla JavaScript |
| Model client | Official [`openai`](https://github.com/openai/openai-python) package pointed at `LLM_BASE_URL` |
| Vector store | [Chroma](https://www.trychroma.com) (embedded, on disk), one collection per embedding model |
| Document list | SQLite (Python standard library) |
| PDF text | [pypdf](https://pypdf.readthedocs.io) |
| Tooling | [uv](https://docs.astral.sh/uv/), pytest, ruff |

The RAG pipeline is plain Python (no LangChain or LlamaIndex), so every step is readable in a few files.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `LLM_BASE_URL` | `http://localhost:11434/v1` | OpenAI-compatible server. Point at a vLLM or llama.cpp host to switch servers. |
| `LLM_API_KEY` | (none) | Bearer token, if the server requires one. Ollama ignores it. |
| `EMBED_MODEL` | `nomic-embed-text` | Embedding model. Changing it means re-uploading documents. |
| `CHAT_MODEL` | `llama3.1:8b` | Chat model. Use `llama3.2:3b` on 8 GB machines or for faster answers. |
| `RAG_TOP_K` | `5` | Chunks retrieved per question. |
| `CHAT_TIMEOUT_SECONDS` | `120` | How long to wait for the model (covers loading it into memory). |
| `DATA_DIR` | `./data` | Where document metadata and vectors are stored. |

## Commands

| Command | What it does |
| --- | --- |
| `uv run serve` | Web app on http://127.0.0.1:8000 (`--reload` while editing code, `--port` to change port) |
| `uv run doctor` | Checks Python, memory, disk, the model server and both models, then warms the chat model |
| `uv run pytest` | Tests (chunking, model client against a mocked server, store, API) |
| `uv run ruff check .` / `uv run ruff format .` | Lint and format |

Interactive API docs are at http://127.0.0.1:8000/docs while the app runs.

## Project structure

```
src/local_llm_rag/
  main.py          # FastAPI app: page, /api/health, /api/upload, /api/chat, /api/documents
  llm.py           # OpenAI-compatible client: models, embeddings, streaming chat, clear errors
  store.py         # SQLite document list + Chroma vectors, search
  chunk.py pdf.py  # chunking and per-page PDF text
  prompt.py        # RAG prompt
  config.py        # settings from env / .env.local
  doctor.py cli.py # `uv run doctor` and `uv run serve`
  static/index.html
tests/             # pytest suite and a sample PDF
docs/              # setup guide and architecture notes
```

## Security note

There is no login. The app and Ollama both listen on `127.0.0.1` only, so nothing is reachable from your network. Do not expose either to the internet as-is. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for what a shared deployment needs.

## License

[MIT](LICENSE)
