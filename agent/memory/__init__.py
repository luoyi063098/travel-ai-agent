import asyncio
import json
import os
import time
from datetime import datetime

import aiosqlite

from config import DB_PATH


class MemoryStore:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._lock = asyncio.Lock()

    async def init(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                )
            """)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS preferences (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_session
                ON messages(session_id, created_at)
            """)
            await db.commit()

    async def create_session(self, session_id: str):
        now = datetime.now().isoformat()
        async with self._lock:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    "INSERT OR IGNORE INTO sessions (id, created_at, updated_at) VALUES (?, ?, ?)",
                    (session_id, now, now),
                )
                await db.commit()

    async def add_message(
        self, session_id: str, role: str, content: str, metadata: dict | None = None
    ):
        now = datetime.now().isoformat()
        async with self._lock:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    "INSERT INTO messages (session_id, role, content, metadata, created_at) VALUES (?, ?, ?, ?, ?)",
                    (session_id, role, content, json.dumps(metadata or {}), now),
                )
                await db.execute(
                    "UPDATE sessions SET updated_at = ? WHERE id = ?", (now, session_id)
                )
                await db.commit()

    async def get_session_messages(
        self, session_id: str, limit: int = 20
    ) -> list[dict]:
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT role, content, metadata, created_at FROM messages WHERE session_id = ? ORDER BY created_at DESC LIMIT ?",
                (session_id, limit),
            )
            rows = await cursor.fetchall()
        return [
            {
                "role": r[0],
                "content": r[1],
                "metadata": json.loads(r[2]),
                "created_at": r[3],
            }
            for r in reversed(rows)
        ]

    async def set_preference(self, key: str, value: str):
        now = datetime.now().isoformat()
        async with self._lock:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    "INSERT OR REPLACE INTO preferences (key, value, updated_at) VALUES (?, ?, ?)",
                    (key, value, now),
                )
                await db.commit()

    async def get_preference(self, key: str) -> str | None:
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT value FROM preferences WHERE key = ?", (key,)
            )
            row = await cursor.fetchone()
        return row[0] if row else None

    async def get_all_preferences(self) -> dict[str, str]:
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("SELECT key, value FROM preferences")
            rows = await cursor.fetchall()
        return {r[0]: r[1] for r in rows}

    async def delete_preference(self, key: str):
        async with self._lock:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("DELETE FROM preferences WHERE key = ?", (key,))
                await db.commit()

    async def build_context(
        self, session_id: str, current_message: str
    ) -> list[dict]:
        """Build message context for LLM, including session history and preferences."""
        messages: list[dict] = []

        # Inject user preferences as system context
        prefs = await self.get_all_preferences()
        if prefs:
            pref_lines = "\n".join(f"- {k}: {v}" for k, v in prefs.items())
            messages.append({
                "role": "system",
                "content": f"## 用户偏好\n{pref_lines}",
            })

        # Add recent conversation history
        history = await self.get_session_messages(session_id, limit=20)
        for m in history:
            messages.append({"role": m["role"], "content": m["content"]})

        # Add current message if not already in history
        if not messages or messages[-1]["content"] != current_message:
            messages.append({"role": "user", "content": current_message})

        return messages


memory_store = MemoryStore()
