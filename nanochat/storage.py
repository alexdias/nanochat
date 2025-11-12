"""SQLite-backed storage helpers for persisting chat conversations."""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional


ISO_TIMESTAMP = "%Y-%m-%dT%H:%M:%S.%fZ"


def _utcnow() -> str:
    """Return the current UTC time formatted for lexicographic ordering."""

    return datetime.now(timezone.utc).strftime(ISO_TIMESTAMP)


class ConversationNotFoundError(RuntimeError):
    """Raised when a conversation lookup fails."""


@dataclass(slots=True)
class ConversationSummary:
    """A lightweight representation of a stored conversation."""

    id: int
    title: Optional[str]
    created_at: str
    updated_at: str
    last_message_preview: Optional[str]


@dataclass(slots=True)
class StoredMessage:
    """A single message stored in the database."""

    id: int
    conversation_id: int
    role: str
    content: str
    created_at: str


class ConversationStore:
    """SQLite-backed conversation storage helper."""

    def __init__(self, db_path: str | Path):
        path_str = str(db_path)
        self._is_uri = path_str.startswith("file:")
        self._in_memory = path_str == ":memory:"

        if not self._in_memory and not self._is_uri:
            path = Path(path_str).expanduser().resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            path_str = str(path)

        self._db_path = path_str
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            self._db_path,
            check_same_thread=False,
            uri=self._is_uri,
            detect_types=sqlite3.PARSE_DECLTYPES,
        )
        self._conn.row_factory = sqlite3.Row
        self._configure()
        self._create_tables()

    def _configure(self) -> None:
        with self._conn:
            self._conn.execute("PRAGMA foreign_keys = ON")
            self._conn.execute("PRAGMA journal_mode = WAL")
            self._conn.execute("PRAGMA busy_timeout = 5000")

    def _create_tables(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    deleted_at TEXT
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                )
                """
            )
            self._conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_messages_conversation_id
                    ON messages(conversation_id, created_at)
                """
            )

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "ConversationStore":
        return self

    def __exit__(self, *exc_info) -> None:  # type: ignore[override]
        self.close()

    def create_conversation(self, title: Optional[str] = None) -> int:
        """Create a new conversation row and return its identifier."""

        with self._lock, self._conn:
            now = _utcnow()
            cursor = self._conn.execute(
                "INSERT INTO conversations (title, created_at, updated_at) VALUES (?, ?, ?)",
                (title, now, now),
            )
            return int(cursor.lastrowid)

    def append_message(self, conversation_id: int, role: str, content: str) -> int:
        """Persist a message and update the conversation timestamp."""

        with self._lock, self._conn:
            if not self._conversation_exists(conversation_id):
                raise ConversationNotFoundError(f"Conversation {conversation_id} does not exist")

            now = _utcnow()
            cursor = self._conn.execute(
                """
                INSERT INTO messages (conversation_id, role, content, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (conversation_id, role, content, now),
            )
            self._conn.execute(
                "UPDATE conversations SET updated_at = ?, deleted_at = NULL WHERE id = ?",
                (now, conversation_id),
            )
            return int(cursor.lastrowid)

    def list_conversations(self, limit: int = 50, include_deleted: bool = False) -> List[ConversationSummary]:
        """Return conversation summaries ordered by last activity."""

        deleted_clause = "" if include_deleted else "WHERE c.deleted_at IS NULL"
        query = f"""
            SELECT
                c.id,
                c.title,
                c.created_at,
                c.updated_at,
                last_message.content AS last_message_preview
            FROM conversations AS c
            LEFT JOIN (
                SELECT m1.conversation_id, m1.content
                FROM messages AS m1
                WHERE m1.id = (
                    SELECT m2.id
                    FROM messages AS m2
                    WHERE m2.conversation_id = m1.conversation_id
                    ORDER BY m2.created_at DESC, m2.id DESC
                    LIMIT 1
                )
            ) AS last_message ON last_message.conversation_id = c.id
            {deleted_clause}
            ORDER BY c.updated_at DESC
            LIMIT ?
        """

        cursor = self._conn.execute(query, (limit,))
        rows = cursor.fetchall()
        return [
            ConversationSummary(
                id=row["id"],
                title=row["title"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                last_message_preview=row["last_message_preview"],
            )
            for row in rows
        ]

    def get_conversation_metadata(self, conversation_id: int) -> ConversationSummary:
        """Return a single conversation summary including its last message preview."""

        cursor = self._conn.execute(
            """
            SELECT
                c.id,
                c.title,
                c.created_at,
                c.updated_at,
                last_message.content AS last_message_preview
            FROM conversations AS c
            LEFT JOIN (
                SELECT m.content
                FROM messages AS m
                WHERE m.conversation_id = ?
                ORDER BY m.created_at DESC, m.id DESC
                LIMIT 1
            ) AS last_message ON 1=1
            WHERE c.id = ?
            """,
            (conversation_id, conversation_id),
        )
        row = cursor.fetchone()
        if row is None:
            raise ConversationNotFoundError(f"Conversation {conversation_id} does not exist")

        return ConversationSummary(
            id=row["id"],
            title=row["title"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            last_message_preview=row["last_message_preview"],
        )

    def get_conversation(self, conversation_id: int) -> List[StoredMessage]:
        """Fetch all messages for a conversation ordered by creation."""

        if not self._conversation_exists(conversation_id):
            raise ConversationNotFoundError(f"Conversation {conversation_id} does not exist")

        cursor = self._conn.execute(
            """
            SELECT id, conversation_id, role, content, created_at
            FROM messages
            WHERE conversation_id = ?
            ORDER BY created_at ASC, id ASC
            """,
            (conversation_id,),
        )
        return [
            StoredMessage(
                id=row["id"],
                conversation_id=row["conversation_id"],
                role=row["role"],
                content=row["content"],
                created_at=row["created_at"],
            )
            for row in cursor.fetchall()
        ]

    def rename_conversation(self, conversation_id: int, title: Optional[str]) -> None:
        """Rename a conversation and refresh its update timestamp."""

        with self._lock, self._conn:
            if not self._conversation_exists(conversation_id):
                raise ConversationNotFoundError(f"Conversation {conversation_id} does not exist")

            now = _utcnow()
            self._conn.execute(
                "UPDATE conversations SET title = ?, updated_at = ?, deleted_at = NULL WHERE id = ?",
                (title, now, conversation_id),
            )

    def soft_delete_conversation(self, conversation_id: int) -> None:
        """Mark a conversation as deleted without removing its messages."""

        with self._lock, self._conn:
            if not self._conversation_exists(conversation_id):
                raise ConversationNotFoundError(f"Conversation {conversation_id} does not exist")

            now = _utcnow()
            self._conn.execute(
                "UPDATE conversations SET deleted_at = ?, updated_at = ? WHERE id = ?",
                (now, now, conversation_id),
            )

    def restore_conversation(self, conversation_id: int) -> None:
        """Restore a soft-deleted conversation."""

        with self._lock, self._conn:
            if not self._conversation_exists(conversation_id):
                raise ConversationNotFoundError(f"Conversation {conversation_id} does not exist")

            self._conn.execute(
                "UPDATE conversations SET deleted_at = NULL, updated_at = ? WHERE id = ?",
                (_utcnow(), conversation_id),
            )

    def bulk_append(self, conversation_id: int, messages: Iterable[tuple[str, str]]) -> None:
        """Append a batch of messages preserving order."""

        with self._lock, self._conn:
            if not self._conversation_exists(conversation_id):
                raise ConversationNotFoundError(f"Conversation {conversation_id} does not exist")

            records = []
            for role, content in messages:
                records.append((conversation_id, role, content, _utcnow()))

            if records:
                self._conn.executemany(
                    "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                    records,
                )

            now = _utcnow()
            self._conn.execute(
                "UPDATE conversations SET updated_at = ?, deleted_at = NULL WHERE id = ?",
                (now, conversation_id),
            )

    def _conversation_exists(self, conversation_id: int) -> bool:
        cursor = self._conn.execute(
            "SELECT 1 FROM conversations WHERE id = ?",
            (conversation_id,),
        )
        return cursor.fetchone() is not None
