import pytest

from local_llm_rag.config import get_settings


@pytest.fixture(autouse=True)
def clean_settings(monkeypatch, tmp_path):
    """Each test gets default settings and its own data directory."""
    for name in ("LLM_BASE_URL", "LLM_API_KEY", "EMBED_MODEL", "CHAT_MODEL", "RAG_TOP_K"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    # Don't pick up a developer's .env.local.
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
