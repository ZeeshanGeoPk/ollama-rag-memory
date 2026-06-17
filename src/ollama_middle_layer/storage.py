from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sqlite3
import time
from typing import Any

from .pruning import chunk_text, content_hash


@dataclass(frozen=True)
class Turn:
    id: int
    conversation_id: str
    role: str
    content: str
    created_at: float


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    distance: float
    metadata: dict[str, Any]


class TurnStore:
    def __init__(self, sqlite_path: Path) -> None:
        sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self.sqlite_path = sqlite_path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.sqlite_path)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_turns_conversation ON turns(conversation_id, id)"
            )

    def add_turn(self, conversation_id: str, role: str, content: str) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO turns (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (conversation_id, role, content, time.time()),
            )
            return int(cursor.lastrowid)

    def recent_turns(self, conversation_id: str, limit: int) -> list[Turn]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, conversation_id, role, content, created_at
                FROM turns
                WHERE conversation_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (conversation_id, limit),
            ).fetchall()
        return [
            Turn(id=row[0], conversation_id=row[1], role=row[2], content=row[3], created_at=row[4])
            for row in reversed(rows)
        ]

    def clear(self) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM turns")


class ChromaContextStore:
    def __init__(self, chroma_dir: Path, embedding_service: Any) -> None:
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError("Install chromadb first: pip install chromadb") from exc
        chroma_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_service = embedding_service
        self.client = chromadb.PersistentClient(path=str(chroma_dir))
        self.collection = self.client.get_or_create_collection(name="conversation_context")

    def add_turn(self, conversation_id: str, turn_id: int, role: str, content: str) -> None:
        chunks = chunk_text(content)
        if not chunks:
            return
        embeddings = self.embedding_service.embed_many(chunks)
        ids = [f"{conversation_id}:{turn_id}:{index}" for index in range(len(chunks))]
        metadatas = [
            {
                "conversation_id": conversation_id,
                "turn_id": turn_id,
                "role": role,
                "chunk_index": index,
                "hash": content_hash(chunk),
            }
            for index, chunk in enumerate(chunks)
        ]
        self.collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def query(self, conversation_id: str, query_text: str, top_k: int) -> list[RetrievedChunk]:
        embedding = self.embedding_service.embed_one(query_text)
        result = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            where={"conversation_id": conversation_id},
            include=["documents", "distances", "metadatas"],
        )
        docs = result.get("documents", [[]])[0]
        distances = result.get("distances", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        return [
            RetrievedChunk(text=doc, distance=float(distance), metadata=metadata or {})
            for doc, distance, metadata in zip(docs, distances, metadatas)
            if doc
        ]

    def clear(self) -> None:
        self.client.delete_collection(name="conversation_context")
        self.collection = self.client.get_or_create_collection(name="conversation_context")


def conversation_id_from_payload(payload: dict[str, Any], default: str = "default") -> str:
    options = payload.get("options")
    if isinstance(options, dict) and options.get("conversation_id"):
        return str(options["conversation_id"])
    if payload.get("conversation_id"):
        return str(payload["conversation_id"])
    return default
