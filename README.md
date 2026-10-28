# Local LLM RAG

Chat with your PDFs using models that run on your own machine. Answers stream in and cite the source pages. No cloud AI provider, no API keys, no rate limits.

It is a proof of concept for running retrieval-augmented generation (RAG) on self-hosted models. Every model call goes through an **OpenAI-compatible API**, so the same app runs against [Ollama](https://ollama.com) on a laptop today and against a team model server (vLLM, llama.cpp server) later by changing one URL.

```
Browser ──► Next.js app ──► OpenAI-compatible API ──► Ollama on your Mac (now)
               │                (LLM_BASE_URL)    └─► team vLLM / llama.cpp server (later)
               └──► local vector store (data/store.json)
```

## What it does

- **Upload a PDF.** Text is extracted per page, split into overlapping ~200-word chunks, and embedded with `nomic-embed-text` (768 dimensions).
- **Ask a question.** The question is embedded, the closest chunks are found by cosine similarity, and a local chat model (default `llama3.1:8b`) answers using only those chunks, citing them as [1], [2]…
- **See what's happening.** A status bar shows whether the model server is up and both models are pulled. Each answer shows time to first token and total time.

## Quick start

Full requirements and every step, including installing Ollama and Node.js on a Mac, are in **[docs/LOCAL_SETUP.md](docs/LOCAL_SETUP.md)**. In short:

```bash
# 1. Ollama running, with both models pulled
ollama pull nomic-embed-text
ollama pull llama3.1:8b

# 2. The app
git clone https://github.com/drin-tech-tools/local-llm-rag.git
cd local-llm-rag
npm install
npm run doctor   # checks your machine and the models, and warms the chat model
npm run dev      # http://localhost:3000
```

No `.env` file is needed for a default Ollama install. To change models or servers, copy `.env.example` to `.env.local`.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `LLM_BASE_URL` | `http://localhost:11434/v1` | OpenAI-compatible server. Point at a vLLM or llama.cpp host to switch servers. |
| `LLM_API_KEY` | (none) | Bearer token, if the server requires one. Ollama ignores it. |
| `EMBED_MODEL` | `nomic-embed-text` | Embedding model. Changing it means re-uploading documents. |
| `CHAT_MODEL` | `llama3.1:8b` | Chat model. Use `llama3.2:3b` on 8 GB machines or for faster answers. |
| `RAG_TOP_K` | `5` | Chunks retrieved per question. |
| `CHAT_FIRST_TOKEN_TIMEOUT_MS` | `120000` | How long to wait for the first token (covers loading the model). |
| `DATA_DIR` | `./data` | Where documents and embeddings are stored. |

## Scripts

| Command | What it does |
| --- | --- |
| `npm run dev` | Development server on http://localhost:3000 (bound to 127.0.0.1 only) |
| `npm run doctor` | Checks Node, memory, disk, the model server and both models, then warms the chat model |
| `npm test` | Unit tests (vectors, chunking, model client, store) |
| `npm run typecheck` / `npm run lint` | Static checks |
| `npm run build` && `npm start` | Production build, also bound to 127.0.0.1 |

## Project structure

```
app/
  page.tsx                 # status bar, upload, document list, streaming chat
  api/health               # model server + model availability
  api/upload               # PDF → pages → chunks → embeddings → store
  api/chat                 # embed question → search → stream answer with sources
  api/documents            # list / delete
lib/
  llm.ts                   # OpenAI-compatible client: models, embeddings, streaming chat
  store.ts                 # file-backed vector store (data/store.json)
  vector.ts                # cosine similarity, top-k, compact vector encoding
  chunk.ts  pdf.ts         # chunking and PDF text extraction
  prompt.ts  config.ts     # prompt template, env settings
scripts/doctor.ts          # setup and pre-demo checks
docs/                      # setup guide and architecture notes
```

## Security note

There is no login. The app and Ollama both listen on `127.0.0.1` only, so nothing is reachable from your network. Do not expose either to the internet as-is. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for what a shared deployment needs.

## License

[MIT](LICENSE)
