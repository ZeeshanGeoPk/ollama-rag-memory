from __future__ import annotations

from types import SimpleNamespace

from ollama_middle_layer.context_pipeline import ContextPipeline
from ollama_middle_layer.pruning import chunk_text
from ollama_middle_layer.storage import RetrievedChunk, Turn


def _pipeline(neighbors: int = 1, include_pair: bool = True) -> ContextPipeline:
    settings = SimpleNamespace(
        retrieval_chunk_neighbors=neighbors,
        retrieval_include_turn_pair=include_pair,
    )
    return ContextPipeline(settings, None, None, None)  # type: ignore[arg-type]


def _turn(turn_id: int, role: str, content: str) -> Turn:
    return Turn(
        id=turn_id,
        conversation_id="conv",
        role=role,
        content=content,
        created_at=float(turn_id),
    )


def _hit(turn_id: int, chunk_index: int, text: str = "match") -> RetrievedChunk:
    return RetrievedChunk(
        text=text,
        distance=0.1,
        metadata={"turn_id": turn_id, "chunk_index": chunk_index},
    )


def test_retrieval_expands_user_hit_to_complete_exchange() -> None:
    pipeline = _pipeline()
    turns = [
        _turn(1, "user", "How should authentication work?"),
        _turn(2, "assistant", "Use short-lived access tokens and rotating refresh tokens."),
        _turn(3, "user", "Unrelated later question."),
    ]

    passages = pipeline._expand_retrieved_history([_hit(1, 0)], turns)

    assert passages == [
        "[MEMORY turns 1-2]\n"
        "U (turn 1): How should authentication work?\n"
        "A (turn 2): Use short-lived access tokens and rotating refresh tokens."
    ]


def test_retrieval_expands_assistant_hit_to_preceding_user() -> None:
    pipeline = _pipeline()
    turns = [
        _turn(10, "user", "Which database did we choose?"),
        _turn(11, "assistant", "We chose PostgreSQL."),
    ]

    passages = pipeline._expand_retrieved_history([_hit(11, 0)], turns)

    assert "U (turn 10): Which database did we choose?" in passages[0]
    assert "A (turn 11): We chose PostgreSQL." in passages[0]
    assert passages[0].index("U (turn 10)") < passages[0].index("A (turn 11)")


def test_retrieval_includes_chunks_around_vector_hit() -> None:
    content = " ".join(f"section-{index} " + ("x" * 360) for index in range(12))
    chunks = chunk_text(content)
    assert len(chunks) >= 3
    middle = len(chunks) // 2
    pipeline = _pipeline(neighbors=1, include_pair=False)
    turn = _turn(20, "assistant", content)

    passages = pipeline._expand_retrieved_history(
        [_hit(20, middle, chunks[middle])],
        [turn],
    )

    assert chunks[middle - 1] in passages[0]
    assert chunks[middle] in passages[0]
    assert chunks[middle + 1] in passages[0]
    if middle >= 2:
        assert chunks[middle - 2] not in passages[0]


def test_multiple_hits_in_same_exchange_are_deduplicated() -> None:
    pipeline = _pipeline()
    turns = [
        _turn(30, "user", "Remember alpha."),
        _turn(31, "assistant", "Alpha is the deployment codename."),
    ]

    passages = pipeline._expand_retrieved_history(
        [_hit(30, 0), _hit(31, 0)],
        turns,
    )

    assert len(passages) == 1
    assert passages[0].count("Remember alpha.") == 1
    assert passages[0].count("Alpha is the deployment codename.") == 1
