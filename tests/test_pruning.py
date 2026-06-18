from __future__ import annotations

from ollama_middle_layer.pruning import (
    chunk_text,
    compact_context,
    is_global_history_query,
    prune_by_budget,
    relevance_score,
    remove_filler_and_duplicates,
)


def test_chunk_text_splits_long_text() -> None:
    chunks = chunk_text("a" * 3000, max_chars=1000, overlap=100)

    assert len(chunks) > 1
    assert chunks[0] == "a" * 1000


def test_chunk_text_prefers_sentence_boundaries() -> None:
    text = "First sentence. Second sentence is longer. Third sentence."

    chunks = chunk_text(text, max_chars=35, overlap=0)

    assert chunks == ["First sentence.", "Second sentence is longer.", "Third sentence."]


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


def test_relevance_score_maps_cosine_to_percentage_range() -> None:
    score = relevance_score("python cache", "python cache", [1.0, 0.0], [1.0, 0.0])

    assert score == 1.0


def test_compact_context_removes_formatting_noise_but_keeps_code() -> None:
    text = "  Key:    value  \n\n\n```python\n  print('x')\n```"

    assert compact_context(text) == "Key: value\n```python\n  print('x')\n```"


def test_detects_global_history_queries() -> None:
    assert is_global_history_query("Can you summarize what I asked in this whole chat?")
    assert not is_global_history_query("Can you summarize Tesla coils?")
