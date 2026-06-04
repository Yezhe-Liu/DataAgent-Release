"""长期上下文管理: LLM 驱动的向量记忆

每轮对话后由 LLM 自动判断是否值得记忆，
存入独立 ChromaDB collection，跨会话检索召回。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
except Exception:
    chromadb = None
    ChromaSettings = None

_EXTRACT_PROMPT = """判断以下对话是否包含值得长期记忆的信息。

长期记忆包括:
- 用户偏好或习惯 (例如 "我喜欢用柱状图")
- 重要决策或结论 (例如 "P0 定义为服务完全不可用")
- 用户背景信息 (例如 "我们公司 SLA 是 99.9%")
- 待办或后续提醒

若包含: {"worth_remembering": true, "memory": "一句话概括(<100字)"}
若不包含: {"worth_remembering": false, "memory": ""}
仅输出 JSON。"""


class VectorMemory:
    """跨会话的用户长期记忆。"""

    COLLECTION_NAME = "user_long_term_memory"

    def __init__(self, persist_dir: str | Path = ""):
        persist = str(persist_dir) if persist_dir else "backend/knowledge_base/long_term_memory"
        self._persist_dir = persist
        self._client: Any = None
        self._model: Any = None

    def set_model(self, model: Any) -> None:
        self._model = model

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

    def store(self, user_id: str, content: str, metadata: dict | None = None) -> None:
        if self.collection is None:
            return
        memory_id = f"{user_id}::{self._hash_content(content)[:8]}"
        try:
            self.collection.upsert(
                ids=[memory_id],
                documents=[content],
                metadatas=[{"user_id": user_id, **(metadata or {})}],
            )
        except Exception:
            pass

    def recall(self, user_id: str, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        if self.collection is None or self.collection.count() == 0:
            return []
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
        memories: list[dict] = []
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            memories.append({"content": doc, "key": meta.get("key", ""), "user_id": meta.get("user_id", "")})
        return memories

    def extract_and_store(self, user_id: str, user_msg: str, ai_msg: str) -> str:
        """LLM 判断 + 存储。返回提取到的记忆内容（空字符串表示无）。"""
        if self._model is None:
            return self._fallback_extract(user_id, user_msg, ai_msg)

        try:
            response = self._model.invoke(_EXTRACT_PROMPT.format(
                user_msg=user_msg[:500], ai_msg=ai_msg[:500]
            ))
            raw = response.content if hasattr(response, "content") else str(response)
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            result = json.loads(raw)
        except Exception:
            return self._fallback_extract(user_id, user_msg, ai_msg)

        if result.get("worth_remembering") and result.get("memory"):
            memory = result["memory"]
            self.store(user_id=user_id, content=memory, metadata={"source": "llm_extracted"})
            return memory

        return ""

    def _fallback_extract(self, user_id: str, user_msg: str, ai_msg: str) -> str:
        """关键词匹配降级方案。"""
        keywords = ["偏好", "喜欢", "常用", "重要", "P0", "SLA", "退款", "记住", "以后", "默认"]
        combined = f"{user_msg} {ai_msg}"
        if any(kw in combined for kw in keywords):
            memory = f"用户: {user_msg[:100]}\n助手: {ai_msg[:100]}"
            self.store(user_id=user_id, content=memory, metadata={"source": "keyword"})
            return memory
        return ""

    @staticmethod
    def _hash_content(content: str) -> str:
        import hashlib
        return hashlib.md5(content.encode()).hexdigest()
