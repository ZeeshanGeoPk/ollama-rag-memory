from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sqlite3
import time
from typing import Any
from uuid import uuid4

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


@dataclass(frozen=True)
class Conversation:
    id: str
    title: str
    created_at: float
    updated_at: float
    message_count: int
    preview: str


@dataclass(frozen=True)
class ConversationMessage:
    id: int
    conversation_id: str
    role: str
    content: str
    mode: str
    created_at: float


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

    def all_turns(self, conversation_id: str) -> list[Turn]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, conversation_id, role, content, created_at
                FROM turns
                WHERE conversation_id = ?
                ORDER BY id
                """,
                (conversation_id,),
            ).fetchall()
        return [
            Turn(id=row[0], conversation_id=row[1], role=row[2], content=row[3], created_at=row[4])
            for row in rows
        ]

    def every_turn(self) -> list[Turn]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, conversation_id, role, content, created_at
                FROM turns
                ORDER BY id
                """
            ).fetchall()
        return [
            Turn(id=row[0], conversation_id=row[1], role=row[2], content=row[3], created_at=row[4])
            for row in rows
        ]

    def clear(self) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM turns")

    def delete_conversation(self, conversation_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM turns WHERE conversation_id = ?", (conversation_id,))


class ConversationStore:
    def __init__(self, sqlite_path: Path) -> None:
        sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self.sqlite_path = sqlite_path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.sqlite_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_conversation_messages
                ON conversation_messages(conversation_id, id)
                """
            )

    def create(self, title: str = "New chat") -> Conversation:
        conversation_id = uuid4().hex
        now = time.time()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO conversations (id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (conversation_id, title, now, now),
            )
        return Conversation(conversation_id, title, now, now, 0, "")

    def list(self) -> list[Conversation]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    c.id,
                    c.title,
                    c.created_at,
                    c.updated_at,
                    COUNT(m.id) AS message_count,
                    COALESCE((
                        SELECT content
                        FROM conversation_messages latest
                        WHERE latest.conversation_id = c.id
                        ORDER BY latest.id DESC
                        LIMIT 1
                    ), '') AS preview
                FROM conversations c
                LEFT JOIN conversation_messages m ON m.conversation_id = c.id
                GROUP BY c.id
                ORDER BY c.updated_at DESC
                """
            ).fetchall()
        return [
            Conversation(
                id=row["id"],
                title=row["title"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                message_count=row["message_count"],
                preview=row["preview"],
            )
            for row in rows
        ]

    def get(self, conversation_id: str) -> Conversation | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT
                    c.id,
                    c.title,
                    c.created_at,
                    c.updated_at,
                    COUNT(m.id) AS message_count,
                    COALESCE((
                        SELECT content
                        FROM conversation_messages latest
                        WHERE latest.conversation_id = c.id
                        ORDER BY latest.id DESC
                        LIMIT 1
                    ), '') AS preview
                FROM conversations c
                LEFT JOIN conversation_messages m ON m.conversation_id = c.id
                WHERE c.id = ?
                GROUP BY c.id
                """,
                (conversation_id,),
            ).fetchone()
        if row is None:
            return None
        return Conversation(
            id=row["id"],
            title=row["title"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            message_count=row["message_count"],
            preview=row["preview"],
        )

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        mode: str,
    ) -> ConversationMessage:
        now = time.time()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO conversation_messages
                    (conversation_id, role, content, mode, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (conversation_id, role, content, mode, now),
            )
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (now, conversation_id),
            )
            message_id = int(cursor.lastrowid)
        return ConversationMessage(message_id, conversation_id, role, content, mode, now)

    def messages(self, conversation_id: str) -> list[ConversationMessage]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, conversation_id, role, content, mode, created_at
                FROM conversation_messages
                WHERE conversation_id = ?
                ORDER BY id
                """,
                (conversation_id,),
            ).fetchall()
        return [
            ConversationMessage(
                id=row["id"],
                conversation_id=row["conversation_id"],
                role=row["role"],
                content=row["content"],
                mode=row["mode"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def set_title(self, conversation_id: str, title: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
                (title, time.time(), conversation_id),
            )

    def delete(self, conversation_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM conversation_messages WHERE conversation_id = ?",
                (conversation_id,),
            )
            conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))

    def clear(self) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM conversation_messages")
            conn.execute("DELETE FROM conversations")


class ChromaContextStore:
    def __init__(self, chroma_dir: Path, embedding_service: Any) -> None:
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError("Install chromadb first: pip install chromadb") from exc
        chroma_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_service = embedding_service
        self.client = chromadb.PersistentClient(path=str(chroma_dir))
        self.collection_name = "conversation_context_v3"
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_turn(self, conversation_id: str, turn_id: int, role: str, content: str) -> None:
        chunks = chunk_text(content)
        if not chunks:
            return
        embeddings = self.embedding_service.embed_documents(chunks)
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
        embedding = self.embedding_service.embed_query(query_text)
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
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def delete_conversation(self, conversation_id: str) -> None:
        self.collection.delete(where={"conversation_id": conversation_id})

    def reindex_if_empty(self, turns: list[Turn]) -> None:
        if self.collection.count() or not turns:
            return
        for turn in turns:
            self.add_turn(
                turn.conversation_id,
                turn.id,
                turn.role,
                turn.content,
            )


def conversation_id_from_payload(payload: dict[str, Any], default: str = "default") -> str:
    options = payload.get("options")
    if isinstance(options, dict) and options.get("conversation_id"):
        return str(options["conversation_id"])
    if payload.get("conversation_id"):
        return str(payload["conversation_id"])
    return default
