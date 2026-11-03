"""Settings from environment variables (or .env / .env.local), with defaults for a local Ollama."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=(".env", ".env.local"), extra="ignore")

    # OpenAI-compatible model server. Ollama serves this on port 11434.
    llm_base_url: str = "http://localhost:11434/v1"
    # Only needed if the server requires one; Ollama ignores it.
    llm_api_key: str = ""
    embed_model: str = "nomic-embed-text"
    chat_model: str = "llama3.1:8b"
    rag_top_k: int = Field(default=5, gt=0)
    # Covers loading the model into memory on the first question (30-60 s on a MacBook Air).
    chat_timeout_seconds: float = Field(default=120, gt=0)
    data_dir: Path = Path("data")

    @property
    def base_url(self) -> str:
        return self.llm_base_url.rstrip("/")


@lru_cache
def get_settings() -> Settings:
    return Settings()
