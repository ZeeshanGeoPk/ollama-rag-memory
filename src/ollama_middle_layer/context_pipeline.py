from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import Settings
from .ollama_clients import EmbeddingService
from .pruning import (
    cosine_similarity,
    estimate_tokens,
    prune_by_budget,
    remove_filler_and_duplicates,
    split_sentences,
)
from .storage import ChromaContextStore, TurnStore


@dataclass(frozen=True)
class ContextPreview:
    conversation_id: str
    current_text: str
    recent_messages: list[dict[str, str]]
    retrieved_chunks: list[str]
    pruned_context: str
    estimated_tokens: int


class ContextPipeline:
    def __init__(
        self,
        settings: Settings,
        turn_store: TurnStore,
        chroma_store: ChromaContextStore,
        embedding_service: EmbeddingService,
    ) -> None:
        self.settings = settings
        self.turn_store = turn_store
        self.chroma_store = chroma_store
        self.embedding_service = embedding_service

    def ingest_messages(self, conversation_id: str, messages: list[dict[str, Any]]) -> None:
        for message in messages:
            role = str(message.get("role", "user"))
            content = str(message.get("content", "")).strip()
            if not content:
                continue
            turn_id = self.turn_store.add_turn(conversation_id, role, content)
            self.chroma_store.add_turn(conversation_id, turn_id, role, content)

    def ingest_prompt(self, conversation_id: str, prompt: str) -> None:
        content = prompt.strip()
        if not content:
            return
        turn_id = self.turn_store.add_turn(conversation_id, "user", content)
        self.chroma_store.add_turn(conversation_id, turn_id, "user", content)

    def build_preview(self, conversation_id: str, current_text: str) -> ContextPreview:
        recent_turns = self.turn_store.recent_turns(
            conversation_id, self.settings.recent_turns_to_keep
        )
        retrieved = self.chroma_store.query(
            conversation_id,
            current_text,
            self.settings.retrieval_top_k,
        )
        recent_texts = {turn.content for turn in recent_turns}
        raw_chunks = [chunk.text for chunk in retrieved if chunk.text not in recent_texts]
        deduped = remove_filler_and_duplicates(raw_chunks)
        pruned_chunks = self._semantic_sentence_prune(current_text, deduped)
        context_sections = []
        if pruned_chunks:
            context_sections.append("Relevant older context:\n" + "\n\n".join(pruned_chunks))
        if recent_turns:
            recent = "\n".join(f"{turn.role}: {turn.content}" for turn in recent_turns)
            context_sections.append("Recent conversation:\n" + recent)
        kept_sections = prune_by_budget(context_sections, self.settings.max_context_tokens)
        pruned_context = "\n\n".join(kept_sections).strip()
        return ContextPreview(
            conversation_id=conversation_id,
            current_text=current_text,
            recent_messages=[
                {"role": turn.role, "content": turn.content}
                for turn in recent_turns
            ],
            retrieved_chunks=raw_chunks,
            pruned_context=pruned_context,
            estimated_tokens=estimate_tokens(pruned_context) if pruned_context else 0,
        )

    def _semantic_sentence_prune(self, query: str, chunks: list[str]) -> list[str]:
        if not chunks:
            return []
        query_embedding = self.embedding_service.embed_one(query)
        output: list[str] = []
        for chunk in chunks:
            sentences = split_sentences(chunk)
            if not sentences:
                continue
            sentence_embeddings = self.embedding_service.embed_many(sentences)
            kept = [
                sentence
                for sentence, embedding in zip(sentences, sentence_embeddings)
                if cosine_similarity(query_embedding, embedding) >= self.settings.sentence_score_threshold
            ]
            if kept:
                output.append("\n".join(kept))
        return output

    def augment_chat_payload(self, conversation_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        messages = list(payload.get("messages") or [])
        current_text = _latest_user_text(messages)
        if not current_text:
            return payload
        preview = self.build_preview(conversation_id, current_text)
        if not preview.pruned_context:
            return payload
        augmented = dict(payload)
        augmented["messages"] = [
            {
                "role": "system",
                "content": (
                    "Use this locally retrieved and pruned context only when relevant. "
                    "It may contain older conversation details.\n\n"
                    f"{preview.pruned_context}"
                ),
            },
            *messages,
        ]
        return augmented

    def augment_generate_payload(self, conversation_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        prompt = str(payload.get("prompt", ""))
        if not prompt.strip():
            return payload
        preview = self.build_preview(conversation_id, prompt)
        if not preview.pruned_context:
            return payload
        augmented = dict(payload)
        augmented["prompt"] = (
            "Use this locally retrieved and pruned context only when relevant:\n"
            f"{preview.pruned_context}\n\nCurrent request:\n{prompt}"
        )
        return augmented

    def clear(self) -> None:
        self.turn_store.clear()
        self.chroma_store.clear()


def _latest_user_text(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return str(message.get("content", ""))
    if messages:
        return str(messages[-1].get("content", ""))
    return ""
