from __future__ import annotations

from ollama_middle_layer.storage import ConversationStore, TurnStore


def test_turn_store_keeps_recent_order(tmp_path) -> None:
    store = TurnStore(tmp_path / "context.sqlite")
    for index in range(5):
        store.add_turn("conv", "user", f"message {index}")

    recent = store.recent_turns("conv", limit=3)

    assert [turn.content for turn in recent] == ["message 2", "message 3", "message 4"]


def test_conversation_store_persists_history(tmp_path) -> None:
    store = ConversationStore(tmp_path / "context.sqlite")
    conversation = store.create("Test chat")
    store.add_message(conversation.id, "user", "Hello", "middleware")
    store.add_message(conversation.id, "assistant", "Hi there", "middleware")

    saved = store.get(conversation.id)
    messages = store.messages(conversation.id)

    assert saved is not None
    assert saved.message_count == 2
    assert saved.preview == "Hi there"
    assert [message.role for message in messages] == ["user", "assistant"]


def test_conversation_store_deletes_history(tmp_path) -> None:
    store = ConversationStore(tmp_path / "context.sqlite")
    conversation = store.create()
    store.add_message(conversation.id, "user", "Delete me", "ollama")

    store.delete(conversation.id)

    assert store.get(conversation.id) is None
    assert store.messages(conversation.id) == []
