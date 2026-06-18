from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from ollama_middle_layer.app import _ui_chat_stream


class FakeResponse:
    status_code = 200

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    async def aiter_lines(self):
        yield json.dumps({"message": {"content": "Hello "}, "done": False})
        yield json.dumps(
            {
                "message": {"content": "there"},
                "done": True,
                "eval_count": 2,
            }
        )


class FakeHttpClient:
    def stream(self, method, url, json):
        return FakeResponse()


class FakeConversationStore:
    def __init__(self) -> None:
        self.messages = []

    def add_message(self, conversation_id, role, content, mode):
        self.messages.append((conversation_id, role, content, mode))


class FakePipeline:
    def __init__(self) -> None:
        self.messages = []

    def ingest_messages(self, conversation_id, messages):
        self.messages.append((conversation_id, messages))


def test_ui_stream_persists_completed_assistant_message(monkeypatch) -> None:
    store = FakeConversationStore()
    pipeline = FakePipeline()
    state = SimpleNamespace(
        settings=SimpleNamespace(llm_ollama_host="http://localhost:8000"),
        http_client=FakeHttpClient(),
        conversation_store=store,
        pipeline=pipeline,
    )

    async def run_inline(function, *args):
        return function(*args)

    monkeypatch.setattr("ollama_middle_layer.app.asyncio.to_thread", run_inline)

    async def collect():
        return [
            json.loads(chunk)
            async for chunk in _ui_chat_stream(
                state=state,
                conversation_id="chat-1",
                mode="middleware",
                payload={"messages": [], "stream": True},
                context_data={"mode": "middleware"},
            )
        ]

    events = asyncio.run(collect())

    assert [event["type"] for event in events] == ["meta", "token", "token", "done"]
    assert store.messages == [("chat-1", "assistant", "Hello there", "middleware")]
    assert pipeline.messages == [
        ("chat-1", [{"role": "assistant", "content": "Hello there"}])
    ]
