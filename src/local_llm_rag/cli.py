"""`uv run serve`: start the web app on http://127.0.0.1:8000."""

import argparse

import uvicorn


def serve() -> None:
    parser = argparse.ArgumentParser(description="Run the Local LLM RAG web app.")
    # 127.0.0.1 only: the app has no login, so keep it off the network.
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="restart on code changes")
    args = parser.parse_args()
    uvicorn.run("local_llm_rag.main:app", host=args.host, port=args.port, reload=args.reload)
