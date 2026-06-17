from __future__ import annotations

from ollama_middle_layer.storage import conversation_id_from_payload


def test_conversation_id_uses_payload_value() -> None:
    assert conversation_id_from_payload({"conversation_id": "abc"}) == "abc"


def test_conversation_id_uses_options_value() -> None:
    assert conversation_id_from_payload({"options": {"conversation_id": "abc"}}) == "abc"


def test_conversation_id_uses_default_for_missing_id() -> None:
    assert conversation_id_from_payload({"messages": []}, default="default") == "default"
