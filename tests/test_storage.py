from __future__ import annotations

from ollama_middle_layer.storage import TurnStore


def test_turn_store_keeps_recent_order(tmp_path) -> None:
    store = TurnStore(tmp_path / "context.sqlite")
    for index in range(5):
        store.add_turn("conv", "user", f"message {index}")

    recent = store.recent_turns("conv", limit=3)

    assert [turn.content for turn in recent] == ["message 2", "message 3", "message 4"]
