"""MemoryManager — 统一上下文管理入口

组合短期 (SummarizationBuffer) + 长期 (VectorMemory) 记忆。
"""

from __future__ import annotations

from typing import Any

from src.memory.long_term import VectorMemory
from src.memory.short_term import MemoryContext, SummarizationBuffer


class MemoryManager:
    """统一记忆管理器。

    用法:
        mgr = MemoryManager(model=llm)
        ctx = await mgr.get_context(user_id, session_id, current_query)
        # ctx.history / ctx.summary / ctx.long_term_hints
    """

    def __init__(self, model: Any = None, max_messages: int = 16, persist_dir: str = ""):
        self.model = model
        self.max_messages = max_messages
        self.short_term = SummarizationBuffer(
            model=model,
            max_messages=max_messages,
            summary_trigger=max(8, max_messages // 2),
        )
        self.long_term = VectorMemory(persist_dir=persist_dir)
        if model is not None:
            self.long_term.set_model(model)
        self._session_buffers: dict[str, SummarizationBuffer] = {}

    async def add_turn(
        self,
        user_id: str,
        session_id: str,
        user_msg: str,
        ai_msg: str,
    ) -> None:
        """存储一轮对话。"""
        # 长期: 自动提取关键信息
        self.long_term.extract_and_store(user_id, user_msg, ai_msg)

    def get_context(
        self,
        user_id: str,
        session_id: str,
        current_query: str,
        history: list[dict[str, str]] | None = None,
    ) -> MemoryContext:
        """构建发给 Agent 的上下文。

        Args:
            user_id: 用户 ID
            session_id: 会话 ID
            current_query: 当前用户问题
            history: 已加载的消息历史
        """
        msgs = history or []
        buffer = self._get_buffer(session_id)
        trimmed = buffer.add_messages(msgs)

        # 长期记忆检索
        hints = ""
        long_memories = self.long_term.recall(user_id, current_query, top_k=3)
        if long_memories:
            hints = "【用户相关历史记忆】\n" + "\n".join(
                f"- {m['content'][:200]}" for m in long_memories
            )

        return MemoryContext(
            history=trimmed,
            summary=buffer.summary,
            hints=hints,
        )

    def _get_buffer(self, session_id: str) -> SummarizationBuffer:
        if session_id not in self._session_buffers:
            self._session_buffers[session_id] = SummarizationBuffer(
                model=self.model,
                max_messages=self.max_messages,
                summary_trigger=max(8, self.max_messages // 2),
            )
        return self._session_buffers[session_id]
