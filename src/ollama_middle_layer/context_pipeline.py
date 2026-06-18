from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import Settings
from .ollama_clients import EmbeddingService
from .pruning import (
    compact_context,
    estimate_tokens,
    is_near_duplicate,
    is_global_history_query,
    limit_items_by_budget,
    normalized_sentence,
    prune_by_budget,
    relevance_score,
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
    original_tokens: int
    reduction_percent: float


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

    def sync_history(self, conversation_id: str, messages: list[dict[str, Any]]) -> None:
        stored = self.turn_store.all_turns(conversation_id)
        normalized = [
            {
                "role": str(message.get("role", "user")),
                "content": str(message.get("content", "")).strip(),
            }
            for message in messages
            if str(message.get("content", "")).strip()
        ]
        common = 0
        for existing, incoming in zip(stored, normalized):
            if existing.role != incoming["role"] or existing.content != incoming["content"]:
                break
            common += 1
        self.ingest_messages(conversation_id, normalized[common:])

    def ingest_prompt(self, conversation_id: str, prompt: str) -> None:
        content = prompt.strip()
        if not content:
            return
        turn_id = self.turn_store.add_turn(conversation_id, "user", content)
        self.chroma_store.add_turn(conversation_id, turn_id, "user", content)

    def build_preview(
        self,
        conversation_id: str,
        current_text: str,
        current_is_stored: bool = False,
    ) -> ContextPreview:
        recent_turns = self.turn_store.recent_turns(
            conversation_id, self.settings.recent_turns_to_keep
        )
        all_turns = self.turn_store.all_turns(conversation_id)
        if (
            current_is_stored
            and all_turns
            and all_turns[-1].role == "user"
            and all_turns[-1].content == current_text
        ):
            current_turn_id = all_turns[-1].id
            all_turns = all_turns[:-1]
            recent_turns = [
                turn for turn in recent_turns if turn.id != current_turn_id
            ]
        recent_ids = {turn.id for turn in recent_turns}
        older_turns = [turn for turn in all_turns if turn.id not in recent_ids]
        context_query = self._context_query(current_text, recent_turns)
        global_history = is_global_history_query(current_text)
        if global_history:
            raw_chunks = [turn.content for turn in older_turns]
            pruned_chunks = self._history_overview(older_turns)
        else:
            retrieved = self.chroma_store.query(
                conversation_id,
                context_query,
                self.settings.retrieval_top_k,
            )
            raw_chunks = [
                chunk.text
                for chunk in retrieved
                if int(chunk.metadata.get("turn_id", -1)) not in recent_ids
            ]
            deduped = remove_filler_and_duplicates(raw_chunks)
            pruned_chunks = self._semantic_sentence_prune(context_query, deduped)
        context_sections = []
        recent: list[str] = []
        if recent_turns:
            recent = self._prune_recent_turns(context_query, recent_turns)
            recent = limit_items_by_budget(
                recent,
                min(
                    self.settings.recent_context_tokens,
                    self.settings.max_context_tokens,
                ),
            )
            if recent:
                context_sections.append("RECENT_TURNS:\n" + "\n".join(recent))
        if pruned_chunks:
            recent_sentences = [
                sentence
                for item in recent
                for sentence in split_sentences(item)
            ]
            pruned_chunks = self._remove_cross_section_duplicates(
                pruned_chunks,
                recent_sentences,
            )
            pruned_chunks = limit_items_by_budget(
                pruned_chunks,
                min(
                    self.settings.retrieved_context_tokens,
                    self.settings.max_context_tokens,
                ),
                separator="\n\n",
            )
            section_name = "HISTORY_OVERVIEW" if global_history else "RELEVANT_HISTORY"
            if pruned_chunks:
                context_sections.append(f"{section_name}:\n" + "\n\n".join(pruned_chunks))
        kept_sections = prune_by_budget(context_sections, self.settings.max_context_tokens)
        pruned_context = "\n\n".join(kept_sections).strip()
        original_text = "\n".join(
            f"{turn.role}: {turn.content}" for turn in all_turns
        )
        if current_text:
            original_text = f"{original_text}\nuser: {current_text}".strip()
        original_tokens = estimate_tokens(original_text) if original_text else 0
        forwarded_tokens = estimate_tokens(
            f"{pruned_context}\nuser: {current_text}".strip()
        )
        reduction = (
            max(0.0, (1.0 - forwarded_tokens / original_tokens) * 100.0)
            if original_tokens
            else 0.0
        )
        return ContextPreview(
            conversation_id=conversation_id,
            current_text=current_text,
            recent_messages=[
                {"role": turn.role, "content": turn.content}
                for turn in recent_turns
            ],
            retrieved_chunks=raw_chunks,
            pruned_context=pruned_context,
            estimated_tokens=forwarded_tokens,
            original_tokens=original_tokens,
            reduction_percent=round(reduction, 1),
        )

    def _semantic_sentence_prune(self, query: str, chunks: list[str]) -> list[str]:
        if not chunks:
            return []
        query_embedding = self.embedding_service.embed_query(query)
        output: list[str] = []
        seen_sentences: set[str] = set()
        for chunk in chunks:
            kept = self._prune_text(query, chunk, query_embedding)
            unique = []
            for sentence in kept:
                normalized = normalized_sentence(sentence)
                if not normalized or normalized in seen_sentences:
                    continue
                seen_sentences.add(normalized)
                unique.append(sentence)
            if unique:
                output.append("\n".join(unique))
            if len(seen_sentences) >= self.settings.max_retrieved_sentences:
                break
        return output

    def _prune_recent_turns(self, query: str, turns: list[Any]) -> list[str]:
        if not turns:
            return []
        protected_start = len(turns) - 1
        for index in range(len(turns) - 1, -1, -1):
            if turns[index].role == "user":
                protected_start = index
                break

        query_embedding = self.embedding_service.embed_query(query)
        output: list[str] = []
        for index, turn in enumerate(turns):
            role = _role_label(turn.role)
            if turn.role == "user" and index >= protected_start:
                content = compact_context(turn.content)
                if content:
                    output.append(f"{role}: {content}")
                continue
            sentence_limit = (
                self.settings.max_recent_assistant_sentences
                if turn.role == "assistant" and index >= protected_start
                else 2
            )
            kept = self._ranked_sentences(
                query,
                turn.content,
                query_embedding,
                sentence_limit,
                keep_fallback=turn.role == "assistant" and index >= protected_start,
            )
            if kept:
                output.append(f"{role}: {' '.join(kept)}")
        return output

    def _prune_text(
        self,
        query: str,
        text: str,
        query_embedding: list[float],
    ) -> list[str]:
        sentences = split_sentences(text)
        if not sentences:
            return []
        sentence_embeddings = self.embedding_service.embed_documents(sentences)
        return [
            compact_context(sentence)
            for sentence, embedding in zip(sentences, sentence_embeddings)
            if relevance_score(query, sentence, query_embedding, embedding)
            >= self.settings.sentence_score_threshold
        ]

    def _ranked_sentences(
        self,
        query: str,
        text: str,
        query_embedding: list[float],
        limit: int,
        keep_fallback: bool = False,
    ) -> list[str]:
        sentences = split_sentences(text)
        if not sentences:
            return []
        embeddings = self.embedding_service.embed_documents(sentences)
        ranked = [
            (
                index,
                relevance_score(query, sentence, query_embedding, embedding),
                compact_context(sentence),
            )
            for index, (sentence, embedding) in enumerate(zip(sentences, embeddings))
        ]
        selected = [
            item
            for item in ranked
            if item[1] >= self.settings.sentence_score_threshold
        ]
        if keep_fallback and not selected:
            selected = ranked
        selected = sorted(selected, key=lambda item: item[1], reverse=True)[:limit]
        return [
            item[2]
            for item in sorted(selected, key=lambda item: item[0])
            if item[2]
        ]

    def _remove_cross_section_duplicates(
        self,
        chunks: list[str],
        recent_sentences: list[str],
    ) -> list[str]:
        accepted = list(recent_sentences)
        output: list[str] = []
        sentence_count = 0
        for chunk in chunks:
            unique: list[str] = []
            for sentence in split_sentences(chunk):
                if any(is_near_duplicate(sentence, existing) for existing in accepted):
                    continue
                unique.append(sentence)
                accepted.append(sentence)
                sentence_count += 1
                if sentence_count >= self.settings.max_retrieved_sentences:
                    break
            if unique:
                output.append("\n".join(unique))
            if sentence_count >= self.settings.max_retrieved_sentences:
                break
        return output

    def _context_query(self, current_text: str, recent_turns: list[Any]) -> str:
        previous_user = next(
            (
                turn.content
                for turn in reversed(recent_turns)
                if turn.role == "user" and turn.content != current_text
            ),
            "",
        )
        if not previous_user:
            return current_text
        return f"{previous_user}\nFollow-up request: {current_text}"

    def _history_overview(self, turns: list[Any]) -> list[str]:
        overview: list[str] = []
        seen: set[str] = set()
        for turn in turns:
            content = compact_context(turn.content)
            if not content:
                continue
            normalized = " ".join(content.lower().split())
            if normalized in seen:
                continue
            seen.add(normalized)
            overview.append(f"{_role_label(turn.role)}: {content}")
        return overview

    def augment_chat_payload(self, conversation_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        messages = list(payload.get("messages") or [])
        current_text = _latest_user_text(messages)
        if not current_text:
            return payload
        preview = self.build_preview(conversation_id, current_text)
        augmented = dict(payload)
        system_messages = [
            message for message in messages if message.get("role") == "system"
        ]
        current_message = next(
            (
                message
                for message in reversed(messages)
                if message.get("role") == "user"
            ),
            messages[-1],
        )
        optimized_messages = [*system_messages]
        if preview.pruned_context:
            optimized_messages.append(
                {
                    "role": "system",
                    "content": (
                        "Relevant compressed memory follows. Use it only when it helps "
                        "answer the current request.\n\n"
                        f"{preview.pruned_context}"
                    ),
                }
            )
        optimized_messages.append(current_message)
        augmented["messages"] = optimized_messages
        return augmented

    def fallback_chat_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        messages = list(payload.get("messages") or [])
        system_messages = [
            message for message in messages if message.get("role") == "system"
        ]
        conversational = [
            message for message in messages if message.get("role") != "system"
        ]
        selected: list[dict[str, Any]] = []
        used_tokens = sum(
            estimate_tokens(str(message.get("content", "")))
            for message in system_messages
        )
        for message in reversed(conversational):
            content = str(message.get("content", ""))
            cost = estimate_tokens(content)
            if selected and used_tokens + cost > self.settings.max_context_tokens:
                break
            if used_tokens + cost > self.settings.max_context_tokens:
                remaining = max(1, self.settings.max_context_tokens - used_tokens)
                message = {**message, "content": content[-remaining * 4 :]}
                cost = remaining
            selected.append(message)
            used_tokens += cost
            if len(selected) >= self.settings.recent_turns_to_keep:
                break
        augmented = dict(payload)
        augmented["messages"] = [*system_messages, *reversed(selected)]
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

    def delete_conversation(self, conversation_id: str) -> None:
        self.turn_store.delete_conversation(conversation_id)
        self.chroma_store.delete_conversation(conversation_id)


def _latest_user_text(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return str(message.get("content", ""))
    if messages:
        return str(messages[-1].get("content", ""))
    return ""


def _role_label(role: str) -> str:
    return {"user": "U", "assistant": "A", "system": "S"}.get(role, role[:1].upper())
