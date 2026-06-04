"""短期上下文管理: ConversationSummaryBuffer

超过窗口阈值的消息不直接丢弃，而是用 LLM 压缩为一段摘要保留。
"""

from __future__ import annotations

from typing import Any

SUMMARIZE_PROMPT = """将以下对话历史压缩为一段简洁摘要。保留关键信息:
- 用户的核心问题或意图
- 你给出的核心答案或结论
- 重要的决策或上下文信息

对话:
{messages}

摘要:"""


class SummarizationBuffer:
    """带摘要的滑动窗口缓冲区。

    当消息数超过 max_messages 时:
      1. 取前半段消息用 LLM 生成摘要
      2. 摘要 + 后半段消息组成新的上下文窗口
    """

    def __init__(self, model: Any = None, max_messages: int = 16, summary_trigger: int = 12):
        self.model = model
        self.max_messages = max_messages
        self.summary_trigger = summary_trigger
        self._summary = ""

    @property
    def summary(self) -> str:
        return self._summary

    def add_messages(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        """处理消息列表，返回裁剪后的窗口（可能包含摘要）。"""
        if len(messages) <= self.max_messages:
            return messages

        # 触发摘要: 保留后半段，前半段压缩
        split = len(messages) - self.summary_trigger
        old_half = messages[:split]
        recent = messages[split:]

        if self.model and old_half:
            self._summary = self._generate_summary(old_half)

        if self._summary:
            # 摘要作为系统消息插入到窗口头部
            return [{"type": "system", "content": f"[对话摘要]\n{self._summary}"}] + recent

        return recent

    def _generate_summary(self, old_messages: list[dict[str, str]]) -> str:
        text = "\n".join(
            f"{'用户' if m.get('type') == 'human' else '助手'}: {m.get('content', '')[:300]}"
            for m in old_messages
        )
        try:
            response = self.model.invoke(SUMMARIZE_PROMPT.format(messages=text))
            return response.content if hasattr(response, "content") else str(response)
        except Exception:
            # 降级: 取最后一条用户消息作为摘要
            for m in reversed(old_messages):
                if m.get("type") == "human":
                    return m.get("content", "")[:200]
            return ""


class MemoryContext:
    """组合后的上下文，供 agent 使用。"""

    __slots__ = ("history", "summary", "long_term_hints")

    def __init__(self, history: list[dict], summary: str = "", hints: str = ""):
        self.history = history
        self.summary = summary
        self.long_term_hints = hints
