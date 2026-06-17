from __future__ import annotations

from contextlib import asynccontextmanager
import asyncio
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import httpx
from pydantic import BaseModel, ConfigDict, Field

from .bootstrap import OllamaBootstrapError, OllamaServerManager, ensure_model
from .config import Settings
from .context_pipeline import ContextPipeline
from .ollama_clients import EmbeddingService, create_ollama_clients
from .storage import ChromaContextStore, TurnStore, conversation_id_from_payload


class AppState:
    settings: Settings
    manager: OllamaServerManager | None
    pipeline: ContextPipeline
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings.load()
    manager: OllamaServerManager | None = None
    clients = create_ollama_clients(settings.llm_ollama_host, settings.embed_ollama_host)

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
    chroma_store = ChromaContextStore(settings.chroma_dir, embedding_service)
    app.state.middle_layer = AppState()
    app.state.middle_layer.settings = settings
    app.state.middle_layer.manager = manager
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
            payload = await asyncio.to_thread(
                state.pipeline.augment_chat_payload, conversation_id, payload
            )
            await asyncio.to_thread(state.pipeline.ingest_messages, conversation_id, original_messages)
    except Exception as exc:
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
    }


@app.post("/admin/reset")
async def reset() -> dict[str, bool]:
    state: AppState = app.state.middle_layer
    await asyncio.to_thread(state.pipeline.clear)
    return {"ok": True}


async def _forward_ollama(path: str, payload: dict[str, Any]) -> Any:
    state: AppState = app.state.middle_layer
    payload.pop("_context_pruning_error", None)
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


def _payload(model: BaseModel) -> dict[str, Any]:
    return model.model_dump(exclude_none=True)
