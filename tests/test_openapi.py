from __future__ import annotations

from ollama_middle_layer.app import app


def test_post_routes_show_request_bodies_in_openapi() -> None:
    schema = app.openapi()

    for path in ["/api/chat", "/api/generate", "/api/embed", "/api/embeddings"]:
        request_body = schema["paths"][path]["post"].get("requestBody")
        assert request_body is not None
        assert "application/json" in request_body["content"]


def test_chat_schema_includes_messages_and_conversation_id() -> None:
    schema = app.openapi()
    chat_schema = schema["components"]["schemas"]["OllamaChatRequest"]

    assert "messages" in chat_schema["properties"]
    assert "conversation_id" in chat_schema["properties"]
