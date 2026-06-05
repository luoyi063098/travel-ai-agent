# agent/memory/__init__.py
# 记忆存储模块 —— 基于 SQLite 的会话历史、消息记录和用户偏好持久化
# 提供 CRUD 方法以及构建 LLM 上下文的功能

import asyncio          # 异步锁支持，保证并发安全
import json             # 序列化/反序列化消息元数据
import os               # 确保数据库目录存在
import time             # （预留）时间相关操作
from datetime import datetime  # 生成时间戳

import aiosqlite        # 异步 SQLite 驱动

from config import DB_PATH  # 从全局配置读取数据库路径


class MemoryStore:
    """记忆存储类，封装 SQLite 数据库操作，负责会话/消息/偏好的增删查改。"""

    def __init__(self, db_path: str = DB_PATH):
        # 数据库文件路径，默认使用全局配置
        self.db_path = db_path
        # 异步锁，防止并发写入时出现竞态条件
        self._lock = asyncio.Lock()

    async def init(self):
        """初始化数据库：创建 sessions、messages、preferences 三张表及索引。"""
        # 确保数据库所在目录存在，不存在则递归创建
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        # 以异步上下文管理器打开数据库连接
        async with aiosqlite.connect(self.db_path) as db:
            # 创建 sessions 表 —— 存储会话基本信息
            await db.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,          -- 会话唯一标识（UUID）
                    created_at TEXT NOT NULL,      -- 会话创建时间
                    updated_at TEXT NOT NULL       -- 会话最后更新时间
                )
            """)
            # 创建 messages 表 —— 存储每条对话消息
            await db.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,  -- 自增主键
                    session_id TEXT NOT NULL,               -- 所属会话 ID
                    role TEXT NOT NULL,                     -- 角色：system / user / assistant
                    content TEXT NOT NULL,                  -- 消息正文
                    metadata TEXT DEFAULT '{}',             -- 附加元数据（JSON 字符串）
                    created_at TEXT NOT NULL,               -- 消息创建时间
                    FOREIGN KEY (session_id) REFERENCES sessions(id)  -- 外键关联会话
                )
            """)
            # 创建 preferences 表 —— 存储用户偏好键值对
            await db.execute("""
                CREATE TABLE IF NOT EXISTS preferences (
                    key TEXT PRIMARY KEY,       -- 偏好键名（如 "city"、"budget"）
                    value TEXT NOT NULL,        -- 偏好值
                    updated_at TEXT NOT NULL    -- 最后更新时间
                )
            """)
            # 为 messages 表创建复合索引，加速按会话和时间排序的查询
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_session
                ON messages(session_id, created_at)
            """)
            # 提交所有 DDL 变更
            await db.commit()

    async def create_session(self, session_id: str):
        """创建新会话（如果已存在则忽略）。"""
        # 获取当前时间的 ISO 格式字符串作为时间戳
        now = datetime.now().isoformat()
        # 加锁确保并发安全
        async with self._lock:
            async with aiosqlite.connect(self.db_path) as db:
                # INSERT OR IGNORE —— 如果 session_id 已存在则不重复插入
                await db.execute(
                    "INSERT OR IGNORE INTO sessions (id, created_at, updated_at) VALUES (?, ?, ?)",
                    (session_id, now, now),
                )
                await db.commit()

    async def add_message(
        self, session_id: str, role: str, content: str, metadata: dict | None = None
    ):
        """向指定会话添加一条消息，同时更新会话的 updated_at。"""
        now = datetime.now().isoformat()
        async with self._lock:
            async with aiosqlite.connect(self.db_path) as db:
                # 插入消息记录，metadata 序列化为 JSON 字符串
                await db.execute(
                    "INSERT INTO messages (session_id, role, content, metadata, created_at) VALUES (?, ?, ?, ?, ?)",
                    (session_id, role, content, json.dumps(metadata or {}), now),
                )
                # 同步更新会话的最后活跃时间
                await db.execute(
                    "UPDATE sessions SET updated_at = ? WHERE id = ?", (now, session_id)
                )
                await db.commit()

    async def get_session_messages(
        self, session_id: str, limit: int = 20
    ) -> list[dict]:
        """获取指定会话的最新 N 条消息，按时间正序返回。"""
        async with aiosqlite.connect(self.db_path) as db:
            # 按 created_at 倒序查询最新 limit 条消息
            cursor = await db.execute(
                "SELECT role, content, metadata, created_at FROM messages WHERE session_id = ? ORDER BY created_at DESC LIMIT ?",
                (session_id, limit),
            )
            rows = await cursor.fetchall()
        # 因为查询结果是倒序，需要反转后返回，使消息按时间正序排列
        # 同时将 metadata 从 JSON 字符串反序列化为字典
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
        """设置用户偏好（键值对），已存在则覆盖。"""
        now = datetime.now().isoformat()
        async with self._lock:
            async with aiosqlite.connect(self.db_path) as db:
                # INSERT OR REPLACE —— 主键冲突时替换整行
                await db.execute(
                    "INSERT OR REPLACE INTO preferences (key, value, updated_at) VALUES (?, ?, ?)",
                    (key, value, now),
                )
                await db.commit()

    async def get_preference(self, key: str) -> str | None:
        """获取指定键的偏好值，不存在则返回 None。"""
        async with aiosqlite.connect(self.db_path) as db:
            # 按主键查询偏好值
            cursor = await db.execute(
                "SELECT value FROM preferences WHERE key = ?", (key,)
            )
            row = await cursor.fetchone()
        # 如果有记录返回 value，否则返回 None
        return row[0] if row else None

    async def get_all_preferences(self) -> dict[str, str]:
        """获取所有用户偏好，返回键值对字典。"""
        async with aiosqlite.connect(self.db_path) as db:
            # 查询所有偏好记录
            cursor = await db.execute("SELECT key, value FROM preferences")
            rows = await cursor.fetchall()
        # 将 [(key, value), ...] 转换为 {key: value, ...} 字典
        return {r[0]: r[1] for r in rows}

    async def delete_preference(self, key: str):
        """删除指定键的偏好。"""
        async with self._lock:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("DELETE FROM preferences WHERE key = ?", (key,))
                await db.commit()

    async def build_context(
        self, session_id: str, current_message: str
    ) -> list[dict]:
        """构建 LLM 消息上下文：注入用户偏好 + 历史记录 + 当前消息。"""
        # 初始化消息列表，每个元素是 {"role": ..., "content": ...} 格式
        messages: list[dict] = []

        # ---- 第 1 层：用户偏好作为 system 消息注入 ----
        prefs = await self.get_all_preferences()
        if prefs:
            # 将偏好格式化为 markdown 列表
            pref_lines = "\n".join(f"- {k}: {v}" for k, v in prefs.items())
            messages.append({
                "role": "system",
                "content": f"## 用户偏好\n{pref_lines}",
            })

        # ---- 第 2 层：最近的对话历史 ----
        history = await self.get_session_messages(session_id, limit=20)
        for m in history:
            messages.append({"role": m["role"], "content": m["content"]})

        # ---- 第 3 层：当前用户消息（如果历史最后一条不是当前消息） ----
        # 避免在消息已存在于历史中时重复添加
        if not messages or messages[-1]["content"] != current_message:
            messages.append({"role": "user", "content": current_message})

        return messages


# 模块级单例，方便全局使用
memory_store = MemoryStore()
