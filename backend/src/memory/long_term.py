"""长期上下文管理: 基于向量检索的用户记忆

将用户偏好/高频问题/重要决策存入独立 ChromaDB collection，
每次新对话时检索相关历史记忆作为上下文提示。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
except Exception:
    chromadb = None
    ChromaSettings = None


class VectorMemory:
    """跨会话的用户长期记忆。"""

    COLLECTION_NAME = "user_long_term_memory"

    def __init__(self, persist_dir: str | Path = ""):
        persist = str(persist_dir) if persist_dir else "backend/knowledge_base/long_term_memory"
        self._persist_dir = persist
        self._client = None

    @property
    def client(self):
        if self._client is None and chromadb is not None:
            Path(self._persist_dir).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=self._persist_dir,
                settings=ChromaSettings(anonymized_telemetry=False) if ChromaSettings else None,
            )
        return self._client

    @property
    def collection(self):
        if self.client is None:
            return None
        return self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def store(self, user_id: str, key: str, content: str, metadata: dict | None = None) -> None:
        """存储一条记忆。"""
        if self.collection is None:
            return
        memory_id = f"{user_id}::{key}::{self._hash_content(content)[:8]}"
        try:
            self.collection.upsert(
                ids=[memory_id],
                documents=[content],
                metadatas=[{"user_id": user_id, "key": key, **(metadata or {})}],
            )
        except Exception:
            pass

    def recall(self, user_id: str, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        """检索与当前查询相关的长期记忆。"""
        if self.collection is None or self.collection.count() == 0:
            return []

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=min(top_k, self.collection.count()),
                include=["documents", "metadatas"],
                where={"user_id": user_id},
            )
        except Exception:
            # Some ChromaDB versions don't support 'where' filter
            try:
                results = self.collection.query(
                    query_texts=[query],
                    n_results=min(top_k, self.collection.count()),
                    include=["documents", "metadatas"],
                )
            except Exception:
                return []

        if not results or not results.get("documents"):
            return []

        memories: list[dict[str, Any]] = []
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            memories.append({
                "content": doc,
                "key": meta.get("key", ""),
                "user_id": meta.get("user_id", ""),
            })

        return memories

    def extract_and_store(self, user_id: str, user_msg: str, ai_msg: str) -> None:
        """从一轮对话中自动提取并存储关键信息。

        规则: 如果对话包含偏好/决策/个人信息关键词，则存储。
        """
        keywords = ["偏好", "喜欢", "常用", "重要", "P0", "SLA", "退款", "密码", "权限"]
        combined = f"{user_msg} {ai_msg}"
        if any(kw in combined for kw in keywords):
            self.store(
                user_id=user_id,
                key="auto_extracted",
                content=f"用户问题: {user_msg[:200]}\n回答: {ai_msg[:200]}",
            )

    @staticmethod
    def _hash_content(content: str) -> str:
        import hashlib
        return hashlib.md5(content.encode()).hexdigest()
