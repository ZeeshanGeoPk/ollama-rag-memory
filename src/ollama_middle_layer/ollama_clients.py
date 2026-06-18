from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class OllamaClients:
    llm: Any
    embed: Any


def create_ollama_clients(llm_host: str, embed_host: str) -> OllamaClients:
    try:
        from ollama import Client
    except ImportError as exc:
        raise RuntimeError("Install the ollama Python package first: pip install ollama") from exc
    return OllamaClients(
        llm=Client(host=llm_host),
        embed=Client(host=embed_host),
    )


class EmbeddingService:
    """Apply retrieval prefixes and normalize Ollama embedding responses."""

    def __init__(self, client: Any, model: str) -> None:
        self.client = client
        self.model = model

    def embed_one(self, text: str) -> list[float]:
        return self._embed_one(text)

    def embed_query(self, text: str) -> list[float]:
        # nomic-embed-text uses asymmetric prefixes for retrieval tasks.
        return self._embed_one(f"search_query: {text}")

    def embed_document(self, text: str) -> list[float]:
        return self._embed_one(f"search_document: {text}")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed_many([f"search_document: {text}" for text in texts])

    def _embed_one(self, text: str) -> list[float]:
        response = self.client.embed(model=self.model, input=text)
        # Current Ollama responses use a list even when only one input was sent.
        embeddings = getattr(response, "embeddings", None)
        if embeddings is None and isinstance(response, dict):
            embeddings = response.get("embeddings")
        if embeddings:
            return list(embeddings[0])

        # Older Ollama SDK/API versions returned a singular `embedding` field.
        legacy = getattr(response, "embedding", None)
        if legacy is None and isinstance(response, dict):
            legacy = response.get("embedding")
        if legacy:
            return list(legacy)
        raise RuntimeError("Ollama embedding response did not include embeddings.")

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return self._embed_many(texts)

    def _embed_many(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self.client.embed(model=self.model, input=texts)
        embeddings = getattr(response, "embeddings", None)
        if embeddings is None and isinstance(response, dict):
            embeddings = response.get("embeddings")
        if embeddings:
            return [list(item) for item in embeddings]
        # Fall back to individual requests for servers without batch embedding.
        return [self._embed_one(text) for text in texts]
