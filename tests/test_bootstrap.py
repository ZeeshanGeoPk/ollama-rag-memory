from __future__ import annotations

from ollama_middle_layer.bootstrap import ensure_model


class FakeClient:
    def __init__(self, names: list[str]) -> None:
        self.names = names
        self.pulled: list[str] = []

    def list(self):
        return {"models": [{"model": name} for name in self.names]}

    def pull(self, model: str) -> None:
        self.pulled.append(model)


def test_ensure_model_skips_existing_model() -> None:
    client = FakeClient(["phi4-mini:3.8b"])

    pulled = ensure_model(client, "phi4-mini:3.8b")

    assert pulled is False
    assert client.pulled == []


def test_ensure_model_pulls_missing_model() -> None:
    client = FakeClient([])

    pulled = ensure_model(client, "nomic-embed-text:v1.5")

    assert pulled is True
    assert client.pulled == ["nomic-embed-text:v1.5"]
