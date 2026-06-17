from __future__ import annotations

from ollama_middle_layer.pruning import (
    chunk_text,
    prune_by_budget,
    remove_filler_and_duplicates,
)


def test_chunk_text_splits_long_text() -> None:
    chunks = chunk_text("a" * 3000, max_chars=1000, overlap=100)

    assert len(chunks) > 1
    assert chunks[0] == "a" * 1000


def test_remove_filler_and_duplicates() -> None:
    chunks = [
        "Hello!",
        "Run npm install.",
        "Run npm install.",
        "How can I help you today?",
    ]

    assert remove_filler_and_duplicates(chunks) == ["Run npm install."]


def test_prune_by_budget_truncates() -> None:
    sections = ["a" * 40, "b" * 400]

    result = prune_by_budget(sections, max_tokens=50)

    assert result[0] == "a" * 40
    assert len(result[1]) <= 160
