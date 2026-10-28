# Local setup

Everything you need to run Local LLM RAG on a Mac, from a clean machine to a working demo. Written for the reference machine below. Other Apple Silicon Macs work the same way; only the model choice changes (see [Choosing a model](#choosing-a-model)).

**Reference machine:** MacBook Air (M3, 2024), 8-core CPU, 10-core GPU, 16 GB unified memory, macOS 26.

## Requirements

| Requirement | Minimum | Reference machine | Check with |
| --- | --- | --- | --- |
| Chip | Apple Silicon (M1 or newer) | Apple M3 | `uname -m` → `arm64` |
| Memory | 8 GB (3B model); 16 GB recommended (8B model) | 16 GB | `sysctl -n hw.memsize` |
| Free disk | 8 GB (models ~5.2 GB + app ~0.5 GB) | 62 GB free | `df -h /` |
| macOS | 14 Sonoma or newer | 26.6 | `sw_vers -productVersion` |
| Node.js | 20 or newer (22 LTS recommended) | install in step 1 | `node -v` |
| Git | any recent | ships with Xcode tools | `git --version` |
| Ollama | latest | install in step 2 | `ollama --version` |
| Network | only for the one-time downloads | | |

Intel Macs are not recommended: Ollama runs on the CPU only there and answers are too slow for a demo.

## 1. Install Node.js 22

Skip this if `node -v` already prints v20 or newer.

```bash
# Homebrew, if you don't have it yet (https://brew.sh)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

brew install node@22
echo 'export PATH="/opt/homebrew/opt/node@22/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
node -v    # v22.x
```

Using nvm instead? The repo has an `.nvmrc`, so `nvm install && nvm use` picks the right version.

## 2. Install and start Ollama

1. Download the macOS app from [ollama.com/download](https://ollama.com/download), move it to Applications and open it. A llama icon appears in the menu bar, and the `ollama` command is installed.
2. Confirm it is serving:

   ```bash
   ollama --version
   curl http://localhost:11434/v1/models    # returns JSON, not "connection refused"
   ```

Prefer Homebrew? `brew install ollama` then `ollama serve` in its own terminal tab (or `brew services start ollama` to keep it running).

### Optional: keep the model loaded longer

By default Ollama unloads a model after 5 minutes idle, and the next question pays the load time again (20–60 s on a MacBook Air). For a demo, keep it loaded for 30 minutes:

```bash
launchctl setenv OLLAMA_KEEP_ALIVE 30m
# then quit Ollama from the menu bar icon and open it again
```

If you raise `RAG_TOP_K` above 8, also raise the context window the same way: `launchctl setenv OLLAMA_CONTEXT_LENGTH 8192`.

## 3. Pull the models

```bash
ollama pull nomic-embed-text   # embeddings, ~0.3 GB
ollama pull llama3.1:8b        # chat, ~4.9 GB
ollama list                    # both should be listed
```

Quick sanity check that the chat model runs (the first run loads it into memory):

```bash
ollama run llama3.1:8b "Say hello in five words."
```

## 4. Get the app running

```bash
git clone https://github.com/drin-tech-tools/local-llm-rag.git
cd local-llm-rag
npm install
npm run doctor
```

`npm run doctor` checks Node, memory, disk, the model server and both models, runs one embedding and one chat request, and leaves the chat model warm. Expected output ends with `All checks passed.` Each failure line says what to run.

Then start the app:

```bash
npm run dev
```

Open **http://localhost:3000** (or http://127.0.0.1:3000). The status bar should show a green dot with `llama3.1:8b` and `nomic-embed-text`.

No `.env.local` is needed with the defaults. To change a setting, `cp .env.example .env.local` and uncomment the line. `npm run dev` and `npm run doctor` both read it; restart `npm run dev` after editing.

## 5. Try it

1. Choose a PDF with selectable text (scanned image-only PDFs have no text to extract).
2. Wait for "N pages, M chunks, embedded in X s". A 40-page PDF usually takes a few seconds.
3. Ask a question. Sources appear under the answer with page numbers, followed by the timing line.

Documents are stored in `data/store.json`. Delete them in the UI, or delete that file to start over.

## Choosing a model

Set `CHAT_MODEL` in `.env.local`, pull the model, and restart `npm run dev`. Sizes and speeds are approximate for 4-bit models on Apple Silicon. Model names are from the [Ollama library](https://ollama.com/library); check there for current tags.

| Model | Download | Fits on | Approx. speed on M3 Air | Notes |
| --- | --- | --- | --- | --- |
| `llama3.1:8b` | ~4.9 GB | 16 GB | 15–20 tokens/s | **Default.** Good answers with citations |
| `qwen2.5:7b` | ~4.7 GB | 16 GB | 15–20 tokens/s | Strong at following the "cite your sources" format |
| `gemma3:4b` | ~3.3 GB | 8–16 GB | 25–30 tokens/s | Middle ground |
| `llama3.2:3b` | ~2.0 GB | 8 GB | 30–40 tokens/s | Fastest; weaker on questions that combine several sources |

Avoid "thinking" models (`qwen3`, `deepseek-r1`) for the demo: they reason silently before answering, so the first visible word can take a minute.

Changing `EMBED_MODEL` is different: vectors from different embedding models are not comparable, so existing documents are marked "re-upload" and skipped until you upload them again. Keep a 768-dimension model if you plan to move to pgvector with the schema in [ARCHITECTURE.md](ARCHITECTURE.md).

## Before a demo (16 GB MacBook Air)

A 16 GB Air runs an 8B model comfortably only when memory is not already full.

- [ ] Quit heavy apps and close extra browser tabs. Aim for at least 6 GB free (`memory_pressure | tail -1` shows the free percentage; `npm run doctor` warns when it is low).
- [ ] Plug in the charger and turn off Low Power Mode. The Air has no fan and slows down on battery under sustained load.
- [ ] Run `npm run doctor` a minute before you present, so the first question doesn't pay the model load time.
- [ ] Check `ollama ps`: it should list `llama3.1:8b` as loaded.
- [ ] Have the PDF already uploaded, plus one spare to upload live.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| Red dot: "Cannot reach the model server" | Open the Ollama app (or run `ollama serve`). Check `curl http://localhost:11434/v1/models`. |
| Red dot: "Missing models" | Run the `ollama pull …` commands the status bar shows, then click **Recheck**. |
| First answer takes 30–60 s, later ones are quick | Normal: the model is loading into memory. Run `npm run doctor` first, and set `OLLAMA_KEEP_ALIVE` (step 2). |
| "No answer from llama3.1:8b within 120s" | Memory is too full and macOS is swapping. Quit apps, or switch to `CHAT_MODEL=llama3.2:3b`. |
| Answers are slow throughout (under 5 tokens/s) | Same cause: low free memory, battery or Low Power Mode. See the demo checklist. |
| "No extractable text (scanned PDF?)" | The PDF is images only. Use a text PDF, or OCR it first (Preview can't; Adobe Acrobat or `ocrmypdf` can). |
| Documents show "re-upload" | `EMBED_MODEL` changed since they were uploaded. Upload them again. |
| `npm run dev` fails on `node` version | `node -v` must be 20+; see step 1. |
| Port 3000 is busy | `npm run dev -- -p 3001` |

## Pointing at a team model server

When the team's server is up, nothing in the code changes. In `.env.local`:

```bash
LLM_BASE_URL=http://<server>:8000/v1     # vLLM's default port
LLM_API_KEY=<key, if the server requires one>
CHAT_MODEL=<model name the server exposes>
EMBED_MODEL=<embedding model the server exposes>
```

Then run `npm run doctor`, re-upload documents if the embedding model changed, and start the app.
