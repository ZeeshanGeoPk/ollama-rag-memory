from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    middleware_host: str
    middleware_port: int
    llm_ollama_host: str
    embed_ollama_host: str
    llm_model: str
    embed_model: str
    ollama_models_dir: Path
    chroma_dir: Path
    sqlite_path: Path
    recent_turns_to_keep: int
    retrieval_top_k: int
    sentence_score_threshold: float
    max_context_tokens: int
    default_conversation_id: str
    ollama_bootstrap: bool

    @classmethod
    def load(cls, base_dir: Path | None = None) -> "Settings":
        root = base_dir or Path.cwd()
        _load_dotenv(root / ".env")
        return cls(
            middleware_host=os.getenv("MIDDLEWARE_HOST", "127.0.0.1"),
            middleware_port=int(os.getenv("MIDDLEWARE_PORT", "8080")),
            llm_ollama_host=os.getenv("LLM_OLLAMA_HOST", "http://localhost:8000"),
            embed_ollama_host=os.getenv("EMBED_OLLAMA_HOST", "http://localhost:8001"),
            llm_model=os.getenv("LLM_MODEL", "phi4-mini:3.8b"),
            embed_model=os.getenv("EMBED_MODEL", "nomic-embed-text:v1.5"),
            ollama_models_dir=(root / os.getenv("OLLAMA_MODELS_DIR", ".data/ollama_models")).resolve(),
            chroma_dir=(root / os.getenv("CHROMA_DIR", ".data/chroma")).resolve(),
            sqlite_path=(root / os.getenv("SQLITE_PATH", ".data/context.sqlite")).resolve(),
            recent_turns_to_keep=int(os.getenv("RECENT_TURNS_TO_KEEP", "6")),
            retrieval_top_k=int(os.getenv("RETRIEVAL_TOP_K", "20")),
            sentence_score_threshold=float(os.getenv("SENTENCE_SCORE_THRESHOLD", "0.75")),
            max_context_tokens=int(os.getenv("MAX_CONTEXT_TOKENS", "12000")),
            default_conversation_id=os.getenv("DEFAULT_CONVERSATION_ID", "default"),
            ollama_bootstrap=_bool_env("OLLAMA_BOOTSTRAP", True),
        )
