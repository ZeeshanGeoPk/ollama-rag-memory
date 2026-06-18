from __future__ import annotations

from contextlib import asynccontextmanager
import asyncio
import json
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import httpx
from pydantic import BaseModel, ConfigDict, Field

from .bootstrap import OllamaBootstrapError, OllamaServerManager, ensure_model
from .config import Settings
from .context_pipeline import ContextPipeline, ContextPreview
from .gpu import read_gpu_stats
from .ollama_clients import EmbeddingService, create_ollama_clients
from .pruning import estimate_tokens
from .storage import (
    ChromaContextStore,
    Conversation,
    ConversationMessage,
    ConversationStore,
    TurnStore,
    conversation_id_from_payload,
)


WEB_DIR = Path(__file__).parent / "web"


# ---------------------------------------------------------------------------
# Application state and request models
# ---------------------------------------------------------------------------

class AppState:
    """Services created once during startup and shared by all API handlers."""

    settings: Settings
    manager: OllamaServerManager | None
    pipeline: ContextPipeline
    conversation_store: ConversationStore
    http_client: httpx.AsyncClient


class ChatMessage(BaseModel):
    role: str = Field(..., examples=["user"])
    content: str = Field(..., examples=["What is my project about?"])


class OllamaChatRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str | None = Field(default=None, examples=["phi4-mini:3.8b"])
    messages: list[ChatMessage] = Field(
        default_factory=list,
        examples=[[{"role": "user", "content": "Remember that this project prunes local Ollama context."}]],
    )
    stream: bool = Field(default=False, examples=[False])
    conversation_id: str | None = Field(
        default=None,
        description="Stable middleware conversation id for context storage and retrieval.",
        examples=["my-chat-1"],
    )
    options: dict[str, Any] | None = Field(
        default=None,
        description="Ollama options. You may also include conversation_id here.",
        examples=[{"temperature": 0.2}],
    )


class OllamaGenerateRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str | None = Field(default=None, examples=["phi4-mini:3.8b"])
    prompt: str = Field(default="", examples=["Summarize what this project does."])
    stream: bool = Field(default=False, examples=[False])
    conversation_id: str | None = Field(default=None, examples=["my-chat-1"])
    options: dict[str, Any] | None = Field(default=None, examples=[{"temperature": 0.2}])


class OllamaEmbedRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str | None = Field(default=None, examples=["nomic-embed-text:v1.5"])
    input: str | list[str] = Field(
        ...,
        examples=["Context pruning middleware for local Ollama models."],
    )


class OllamaEmbeddingsRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str | None = Field(default=None, examples=["nomic-embed-text:v1.5"])
    prompt: str = Field(..., examples=["Context pruning middleware for local Ollama models."])


class UIConversationCreate(BaseModel):
    title: str = Field(default="New chat", max_length=120)


class UIConversationUpdate(BaseModel):
    title: str = Field(..., min_length=1, max_length=120)


class UIChatRequest(BaseModel):
    conversation_id: str | None = None
    message: str = Field(..., min_length=1)
    mode: Literal["middleware", "ollama"] = "middleware"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start local services, initialize persistence, and release owned resources."""
    settings = Settings.load()
    manager: OllamaServerManager | None = None
    clients = create_ollama_clients(settings.llm_ollama_host, settings.embed_ollama_host)

    # Bootstrap is optional so tests or externally managed Ollama servers can use
    # the middleware without this process owning their lifecycle.
    if settings.ollama_bootstrap:
        manager = OllamaServerManager(settings.ollama_models_dir)
        try:
            await asyncio.to_thread(manager.start, settings.llm_ollama_host)
            await asyncio.to_thread(manager.start, settings.embed_ollama_host)
            await asyncio.to_thread(ensure_model, clients.llm, settings.llm_model)
            await asyncio.to_thread(ensure_model, clients.embed, settings.embed_model)
        except OllamaBootstrapError:
            if manager:
                manager.shutdown()
            raise

    embedding_service = EmbeddingService(clients.embed, settings.embed_model)
    turn_store = TurnStore(settings.sqlite_path)
    conversation_store = ConversationStore(settings.sqlite_path)
    chroma_store = ChromaContextStore(settings.chroma_dir, embedding_service)
    # SQLite is the source of truth. Rebuild Chroma only when its collection is
    # empty, which covers a deleted vector directory without duplicating records.
    await asyncio.to_thread(chroma_store.reindex_if_empty, turn_store.every_turn())
    app.state.middle_layer = AppState()
    app.state.middle_layer.settings = settings
    app.state.middle_layer.manager = manager
    app.state.middle_layer.conversation_store = conversation_store
    app.state.middle_layer.pipeline = ContextPipeline(
        settings=settings,
        turn_store=turn_store,
        chroma_store=chroma_store,
        embedding_service=embedding_service,
    )
    app.state.middle_layer.http_client = httpx.AsyncClient(timeout=None)
    try:
        yield
    finally:
        await app.state.middle_layer.http_client.aclose()
        if manager:
            manager.shutdown()


app = FastAPI(title="Ollama Context-Pruning Middleware", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Ollama-compatible proxy API
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def web_ui() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/health")
async def health() -> dict[str, Any]:
    state: AppState = app.state.middle_layer
    return {
        "ok": True,
        "llm_host": state.settings.llm_ollama_host,
        "embed_host": state.settings.embed_ollama_host,
        "llm_model": state.settings.llm_model,
        "embed_model": state.settings.embed_model,
    }


@app.get("/api/tags")
async def tags() -> JSONResponse:
    state: AppState = app.state.middle_layer
    response = await state.http_client.get(f"{state.settings.llm_ollama_host}/api/tags")
    return JSONResponse(status_code=response.status_code, content=response.json())


@app.post("/api/chat")
async def chat(request: OllamaChatRequest) -> Any:
    payload = _payload(request)
    state: AppState = app.state.middle_layer
    payload.setdefault("model", state.settings.llm_model)
    conversation_id = conversation_id_from_payload(
        payload, default=state.settings.default_conversation_id
    )
    original_messages = list(payload.get("messages") or [])
    try:
        if original_messages:
            # Sync only prior history first. The current prompt must not be in the
            # retrieval corpus while we are building context for that same prompt.
            prior_messages, current_message = _split_chat_history(original_messages)
            await asyncio.to_thread(
                state.pipeline.sync_history,
                conversation_id,
                prior_messages,
            )
            payload = await asyncio.to_thread(
                state.pipeline.augment_chat_payload, conversation_id, payload
            )
            if current_message:
                await asyncio.to_thread(
                    state.pipeline.ingest_messages,
                    conversation_id,
                    [current_message],
                )
    except Exception as exc:
        payload = state.pipeline.fallback_chat_payload(payload)
        payload["_context_pruning_error"] = str(exc)
    return await _forward_ollama("/api/chat", payload)


@app.post("/api/generate")
async def generate(request: OllamaGenerateRequest) -> Any:
    payload = _payload(request)
    state: AppState = app.state.middle_layer
    payload.setdefault("model", state.settings.llm_model)
    conversation_id = conversation_id_from_payload(
        payload, default=state.settings.default_conversation_id
    )
    prompt = str(payload.get("prompt", ""))
    try:
        if prompt:
            payload = await asyncio.to_thread(
                state.pipeline.augment_generate_payload, conversation_id, payload
            )
            await asyncio.to_thread(state.pipeline.ingest_prompt, conversation_id, prompt)
    except Exception as exc:
        payload["_context_pruning_error"] = str(exc)
    return await _forward_ollama("/api/generate", payload)


@app.post("/api/embed")
async def embed(request: OllamaEmbedRequest) -> Any:
    payload = _payload(request)
    state: AppState = app.state.middle_layer
    payload.setdefault("model", state.settings.embed_model)
    return await _forward_embedding("/api/embed", payload)


@app.post("/api/embeddings")
async def embeddings(request: OllamaEmbeddingsRequest) -> Any:
    payload = _payload(request)
    state: AppState = app.state.middle_layer
    payload.setdefault("model", state.settings.embed_model)
    return await _forward_embedding("/api/embeddings", payload)


# ---------------------------------------------------------------------------
# Browser UI API
# ---------------------------------------------------------------------------

@app.get("/ui/api/conversations")
async def list_conversations() -> list[dict[str, Any]]:
    state: AppState = app.state.middle_layer
    conversations = await asyncio.to_thread(state.conversation_store.list)
    return [_conversation_json(conversation) for conversation in conversations]


@app.post("/ui/api/conversations")
async def create_conversation(request: UIConversationCreate) -> dict[str, Any]:
    state: AppState = app.state.middle_layer
    conversation = await asyncio.to_thread(
        state.conversation_store.create, request.title.strip() or "New chat"
    )
    return _conversation_json(conversation)


@app.get("/ui/api/conversations/{conversation_id}")
async def get_conversation(conversation_id: str) -> dict[str, Any]:
    state: AppState = app.state.middle_layer
    conversation = await asyncio.to_thread(state.conversation_store.get, conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    messages = await asyncio.to_thread(state.conversation_store.messages, conversation_id)
    return {
        **_conversation_json(conversation),
        "messages": [_message_json(message) for message in messages],
    }


@app.patch("/ui/api/conversations/{conversation_id}")
async def update_conversation(
    conversation_id: str,
    request: UIConversationUpdate,
) -> dict[str, Any]:
    state: AppState = app.state.middle_layer
    if await asyncio.to_thread(state.conversation_store.get, conversation_id) is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    await asyncio.to_thread(
        state.conversation_store.set_title,
        conversation_id,
        request.title.strip(),
    )
    conversation = await asyncio.to_thread(state.conversation_store.get, conversation_id)
    return _conversation_json(conversation)


@app.delete("/ui/api/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str) -> dict[str, bool]:
    state: AppState = app.state.middle_layer
    await asyncio.to_thread(state.conversation_store.delete, conversation_id)
    try:
        await asyncio.to_thread(state.pipeline.delete_conversation, conversation_id)
    except Exception:
        # The visible conversation should still be deletable if Chroma is
        # temporarily unavailable; stale vectors are scoped by conversation id.
        pass
    return {"ok": True}


@app.post("/ui/api/chat")
async def ui_chat(request: UIChatRequest) -> StreamingResponse:
    state: AppState = app.state.middle_layer
    conversation = None  # Created lazily for the first message of a new chat.
    if request.conversation_id:
        conversation = await asyncio.to_thread(
            state.conversation_store.get, request.conversation_id
        )
        if conversation is None:
            raise HTTPException(status_code=404, detail="Conversation not found.")
    else:
        conversation = await asyncio.to_thread(
            state.conversation_store.create, _conversation_title(request.message)
        )

    prior_messages = await asyncio.to_thread(
        state.conversation_store.messages, conversation.id
    )  # UI-visible transcript; direct mode sends this entire list.
    if not prior_messages and conversation.title == "New chat":
        await asyncio.to_thread(
            state.conversation_store.set_title,
            conversation.id,
            _conversation_title(request.message),
        )

    await asyncio.to_thread(
        state.conversation_store.add_message,
        conversation.id,
        "user",
        request.message,
        request.mode,
    )

    context_data: dict[str, Any]
    if request.mode == "middleware":
        try:
            # Build context before indexing this prompt to avoid retrieving the
            # current question as its own most relevant historical memory.
            preview = await asyncio.to_thread(
                state.pipeline.build_preview, conversation.id, request.message
            )
            payload_messages = [{"role": "user", "content": request.message}]
            if preview.pruned_context:
                # Retrieved memory is a system message, never a fabricated user turn.
                payload_messages.insert(0, _context_system_message(preview))
            context_data = _context_json(preview, "middleware")
            await asyncio.to_thread(
                state.pipeline.ingest_messages,
                conversation.id,
                [{"role": "user", "content": request.message}],
            )
        except Exception as exc:
            payload_messages = [{"role": "user", "content": request.message}]
            context_data = {
                "mode": "middleware",
                "error": str(exc),
                "estimated_tokens": estimate_tokens(request.message),
                "original_tokens": estimate_tokens(request.message),
                "reduction_percent": 0.0,
                "pruned_context": "",
                "retrieved_chunks": [],
                "recent_messages": [],
            }
    else:
        # Direct mode is the comparison baseline: it sends every UI message and
        # intentionally bypasses retrieval, pruning, and the RAG turn store.
        current_messages = [*prior_messages]  # Copy before appending the unsaved prompt.
        current_messages.append(
            ConversationMessage(
                id=0,
                conversation_id=conversation.id,
                role="user",
                content=request.message,
                mode="ollama",
                created_at=0,
            )
        )
        payload_messages = [
            {"role": message.role, "content": message.content}
            for message in current_messages
        ]
        full_context = "\n".join(
            f"{message['role']}: {message['content']}" for message in payload_messages
        )
        context_data = {
            "mode": "ollama",
            "estimated_tokens": estimate_tokens(full_context),
            "original_tokens": estimate_tokens(full_context),
            "reduction_percent": 0.0,
            "pruned_context": full_context,
            "retrieved_chunks": [],
            "recent_messages": payload_messages,
        }

    payload = {
        "model": state.settings.llm_model,
        "messages": payload_messages,
        "stream": True,
    }
    return StreamingResponse(
        _ui_chat_stream(
            state=state,
            conversation_id=conversation.id,
            mode=request.mode,
            payload=payload,
            context_data=context_data,
        ),
        media_type="application/x-ndjson",
    )


@app.get("/ui/api/conversations/{conversation_id}/context")
async def conversation_context(conversation_id: str) -> dict[str, Any]:
    state: AppState = app.state.middle_layer
    messages = await asyncio.to_thread(state.conversation_store.messages, conversation_id)
    if not messages:
        return {
            "mode": "middleware",
            "estimated_tokens": 0,
            "original_tokens": 0,
            "reduction_percent": 0.0,
            "pruned_context": "",
            "retrieved_chunks": [],
            "recent_messages": [],
        }
    latest_user = next(
        (message for message in reversed(messages) if message.role == "user"),
        messages[-1],
    )
    if latest_user.mode == "ollama":
        content = "\n".join(f"{message.role}: {message.content}" for message in messages)
        return {
            "mode": "ollama",
            "estimated_tokens": estimate_tokens(content),
            "original_tokens": estimate_tokens(content),
            "reduction_percent": 0.0,
            "pruned_context": content,
            "retrieved_chunks": [],
            "recent_messages": [
                {"role": message.role, "content": message.content}
                for message in messages
            ],
        }
    try:
        preview = await asyncio.to_thread(
            state.pipeline.build_preview,
            conversation_id,
            latest_user.content,
            True,
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return _context_json(preview, "middleware")


@app.get("/ui/api/gpu")
async def gpu_stats() -> dict[str, Any]:
    return await asyncio.to_thread(read_gpu_stats)


@app.get("/debug/context-preview")
async def context_preview(conversation_id: str, q: str) -> dict[str, Any]:
    state: AppState = app.state.middle_layer
    try:
        preview = await asyncio.to_thread(state.pipeline.build_preview, conversation_id, q)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {
        "conversation_id": preview.conversation_id,
        "current_text": preview.current_text,
        "recent_messages": preview.recent_messages,
        "retrieved_chunks": preview.retrieved_chunks,
        "pruned_context": preview.pruned_context,
        "estimated_tokens": preview.estimated_tokens,
        "original_tokens": preview.original_tokens,
        "reduction_percent": preview.reduction_percent,
    }


@app.post("/admin/reset")
async def reset() -> dict[str, bool]:
    state: AppState = app.state.middle_layer
    await asyncio.to_thread(state.pipeline.clear)
    await asyncio.to_thread(state.conversation_store.clear)
    return {"ok": True}


# ---------------------------------------------------------------------------
# Upstream forwarding and streaming
# ---------------------------------------------------------------------------

async def _forward_ollama(path: str, payload: dict[str, Any]) -> Any:
    state: AppState = app.state.middle_layer
    payload.pop("_context_pruning_error", None)
    # conversation_id belongs to this middleware and is not an Ollama option.
    payload.pop("conversation_id", None)
    options = payload.get("options")
    if isinstance(options, dict):
        options.pop("conversation_id", None)
    url = f"{state.settings.llm_ollama_host}{path}"
    if payload.get("stream"):
        return StreamingResponse(
            _stream_ollama(url, payload),
            media_type="application/x-ndjson",
        )
    response = await state.http_client.post(url, json=payload)
    return JSONResponse(
        status_code=response.status_code,
        content=response.json(),
    )


async def _forward_embedding(path: str, payload: dict[str, Any]) -> JSONResponse:
    state: AppState = app.state.middle_layer
    url = f"{state.settings.embed_ollama_host}{path}"
    response = await state.http_client.post(url, json=payload)
    return JSONResponse(
        status_code=response.status_code,
        content=response.json(),
    )


async def _stream_ollama(url: str, payload: dict[str, Any]):
    state: AppState = app.state.middle_layer
    async with state.http_client.stream("POST", url, json=payload) as response:
        async for chunk in response.aiter_bytes():
            yield chunk


async def _ui_chat_stream(
    state: AppState,
    conversation_id: str,
    mode: str,
    payload: dict[str, Any],
    context_data: dict[str, Any],
):
    # Metadata is sent first so the UI can display the exact context while the
    # assistant response is still streaming.
    yield _ndjson(
        {
            "type": "meta",
            "conversation_id": conversation_id,
            "context": context_data,
        }
    )
    assistant_text = ""  # Full response accumulated from incremental token events.
    url = f"{state.settings.llm_ollama_host}/api/chat"
    try:
        async with state.http_client.stream("POST", url, json=payload) as response:
            if response.status_code >= 400:
                body = await response.aread()
                yield _ndjson(
                    {
                        "type": "error",
                        "message": body.decode("utf-8", errors="replace"),
                    }
                )
                return
            async for line in response.aiter_lines():
                if not line:
                    continue
                data = json.loads(line)  # One Ollama JSON event per NDJSON line.
                content = data.get("message", {}).get("content", "")
                if content:
                    assistant_text += content
                    yield _ndjson({"type": "token", "content": content})
                if data.get("done"):
                    if assistant_text:
                        # Persist only completed responses. Aborted or failed
                        # streams remain visible client-side but do not pollute RAG.
                        await asyncio.to_thread(
                            state.conversation_store.add_message,
                            conversation_id,
                            "assistant",
                            assistant_text,
                            mode,
                        )
                        if mode == "middleware":
                            await asyncio.to_thread(
                                state.pipeline.ingest_messages,
                                conversation_id,
                                [{"role": "assistant", "content": assistant_text}],
                            )
                    yield _ndjson(
                        {
                            "type": "done",
                            "conversation_id": conversation_id,
                            "context": context_data,
                            "metrics": {
                                "total_duration": data.get("total_duration"),
                                "load_duration": data.get("load_duration"),
                                "prompt_eval_count": data.get("prompt_eval_count"),
                                "eval_count": data.get("eval_count"),
                                "eval_duration": data.get("eval_duration"),
                            },
                        }
                    )
    except (httpx.HTTPError, json.JSONDecodeError) as exc:
        yield _ndjson({"type": "error", "message": str(exc)})


# ---------------------------------------------------------------------------
# Serialization and payload helpers
# ---------------------------------------------------------------------------

def _payload(model: BaseModel) -> dict[str, Any]:
    return model.model_dump(exclude_none=True)


def _ndjson(data: dict[str, Any]) -> bytes:
    return (json.dumps(data, ensure_ascii=True) + "\n").encode("utf-8")


def _conversation_title(message: str) -> str:
    clean = " ".join(message.split())
    return clean[:52] + ("..." if len(clean) > 52 else "")


def _conversation_json(conversation: Conversation) -> dict[str, Any]:
    return {
        "id": conversation.id,
        "title": conversation.title,
        "created_at": conversation.created_at,
        "updated_at": conversation.updated_at,
        "message_count": conversation.message_count,
        "preview": conversation.preview,
    }


def _message_json(message: ConversationMessage) -> dict[str, Any]:
    return {
        "id": message.id,
        "conversation_id": message.conversation_id,
        "role": message.role,
        "content": message.content,
        "mode": message.mode,
        "created_at": message.created_at,
    }


def _context_system_message(preview: ContextPreview) -> dict[str, str]:
    return {
        "role": "system",
        "content": (
            "Use this locally retrieved and pruned context only when relevant. "
            "It may contain older conversation details.\n\n"
            f"{preview.pruned_context}"
        ),
    }


def _context_json(preview: ContextPreview, mode: str) -> dict[str, Any]:
    return {
        "mode": mode,
        "estimated_tokens": preview.estimated_tokens,
        "original_tokens": preview.original_tokens,
        "reduction_percent": preview.reduction_percent,
        "pruned_context": preview.pruned_context,
        "retrieved_chunks": preview.retrieved_chunks,
        "recent_messages": preview.recent_messages,
    }


def _split_chat_history(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Separate the active user prompt from history supplied by API clients."""
    # The last user message is the active request; trailing assistant/tool
    # messages, if supplied by a client, remain part of prior history.
    current_index = next(
        (
            index
            for index in range(len(messages) - 1, -1, -1)
            if messages[index].get("role") == "user"
        ),
        len(messages) - 1,
    )
    current = messages[current_index] if messages else None
    prior = [
        message
        for index, message in enumerate(messages)
        if index != current_index and message.get("role") != "system"
    ]
    return prior, current


app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
