"""Worker 子图基类工具

提供:
  - extract_delta()      Worker 退出时提取增量消息，杜绝 add_messages 重叠翻倍
  - WorkerMarker         入口标记，记录基线消息数量
  - build_worker_input()  从 SupervisorState 构建 worker_input 上下文
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import BaseMessage

from src.graph.state import SupervisorState


# =============================================================================
# Delta 增量消息裁剪 — 父子图消息隔离核心
# =============================================================================


def extract_delta(
    worker_messages: list[BaseMessage],
    entry_baseline: int,
) -> dict[str, Any]:
    """从 Worker 子图消息中提取增量，通过单向强契约通道回传父图。

    核心安全约束:
      WorkerState.worker_messages 与 SupervisorState.messages 字段名不同，
      避免 LangGraph add_messages reducer 自动合并导致消息翻倍。
      退出时只回传 Worker 新产生的消息（AIMessage），内部 SystemMessage/HumanMessage
      不泄露到全局对话历史。

    Args:
        worker_messages: Worker 子图完整内部消息序列
        entry_baseline: Worker 入口前父图 messages 的数量（由 WorkerMarker 记录）

    Returns:
        dict 包含:
          - messages: list[BaseMessage] — 仅 Worker 新产生的增量消息
          - generation: str — 最终生成文本
          - delta_count: int — 增量消息数量 (可观测)

    Example:
        # Worker 入口
        entry_baseline = len(state["worker_messages"])  # 初始为 0

        # Worker 内部执行后
        # worker_messages = [SystemMessage, HumanMessage, AIMessage("结果")]

        delta = extract_delta(worker_messages, entry_baseline)
        # delta = {"messages": [AIMessage("结果")], "generation": "结果", "delta_count": 1}
    """
    if not worker_messages:
        return {"messages": [], "generation": "", "delta_count": 0}

    # 只取 Worker 自己产生的新消息（从 baseline 之后开始）
    new_messages = worker_messages[entry_baseline:]

    # 过滤：只保留 AIMessage（用户可见的回复），排除内部 SystemMessage/HumanMessage
    from langchain_core.messages import AIMessage

    user_visible = [m for m in new_messages if isinstance(m, AIMessage)]

    # 提取最终生成文本
    generation = ""
    for m in reversed(user_visible):
        content = getattr(m, "content", "")
        if content and isinstance(content, str) and content.strip():
            generation = content
            break

    return {
        "messages": user_visible,
        "generation": generation,
        "delta_count": len(user_visible),
    }


# =============================================================================
# Worker 入口标记
# =============================================================================


class WorkerMarker:
    """Worker 入口标记，记录基线以支持 delta 提取。

    在 Worker 子图入口节点中创建:
        marker = WorkerMarker(state)
        state["worker_messages"] = []  # Worker 从空消息开始

    退出时:
        delta = extract_delta(state["worker_messages"], marker.baseline)
    """

    __slots__ = ("baseline", "supervisor_msg_count", "worker_name")

    def __init__(self, state: dict[str, Any], worker_name: str = "unknown") -> None:
        worker_msgs = state.get("worker_messages", [])
        self.baseline = len(worker_msgs) if isinstance(worker_msgs, list) else 0
        self.supervisor_msg_count = len(state.get("messages", []))
        self.worker_name = worker_name


# =============================================================================
# Worker Input 构建器
# =============================================================================


def build_worker_input(supervisor_state: SupervisorState) -> dict[str, Any]:
    """从 SupervisorState 构建 worker_input 上下文字典。

    只提取 Worker 需要的字段，不传递整个 Supervisor 状态。
    """
    return {
        "query": "",
        "intent": supervisor_state.get("intent", "chat"),
        "pending_sql": supervisor_state.get("pending_sql", ""),
        "pending_sql_reasoning": supervisor_state.get("pending_sql_reasoning", ""),
        "active_worker": supervisor_state.get("active_worker", ""),
    }
