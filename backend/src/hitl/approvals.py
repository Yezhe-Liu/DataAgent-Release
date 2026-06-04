"""HITL (Human-in-the-Loop) 审批策略

提供可配置的中断策略和 LangGraph interrupt 配置生成。
纯函数模块，不依赖项目其他模块。
"""

from __future__ import annotations

from typing import Any

from src.config import get_env_text

# ---- 审批策略 ----

PolicyValue = str  # "always_ask" | "never" | "ask"

DEFAULT_POLICY: dict[str, PolicyValue] = {
    "tool_execute": "always_ask",
    "grade_all_irrelevant": "ask",
    "hallucination_low": "ask",
}

_HITL_DISPLAY: dict[str, str] = {
    "tool_execute": "是否允许执行数据分析工具？",
    "grade_all_irrelevant": "知识库未找到相关文档，是否联网搜索？",
    "hallucination_low": "回答可信度较低，是否继续返回？",
}


# ---- Public API ----


def get_hitl_policy() -> dict[str, str]:
    """从环境变量读取 HITL 策略，合并默认值。"""
    raw = get_env_text("HITL_POLICY", "")
    policy = dict(DEFAULT_POLICY)
    if raw:
        for item in raw.split(","):
            item = item.strip()
            if ":" in item:
                key, val = item.split(":", 1)
                key = key.strip()
                val = val.strip()
                if key in policy and val in {"always_ask", "never", "ask"}:
                    policy[key] = val
    return policy


def get_interrupt_nodes() -> list[str]:
    """返回需要配置 interrupt_before 的节点列表。"""
    policy = get_hitl_policy()
    return [node for node, rule in policy.items() if rule != "never"]


def should_interrupt(node_name: str, state: dict[str, Any]) -> bool:
    """运行时判断是否应该中断。

    Args:
        node_name: 当前节点名称
        state: 当前 AgentState

    Returns:
        True 表示应该触发 interrupt 等待人工确认
    """
    policy = get_hitl_policy()
    rule = policy.get(node_name, "never")

    if rule == "never":
        return False
    if rule == "always_ask":
        return True

    # rule == "ask": 根据状态动态判断
    if node_name == "grade_all_irrelevant":
        docs = state.get("graded_docs", [])
        return not any(d.get("relevance") == "relevant" for d in docs)

    if node_name == "hallucination_low":
        score = state.get("hallucination_score", 1.0)
        return score < 0.5

    return True


def get_interrupt_message(node_name: str) -> str:
    """返回给前端展示的中断提示消息。"""
    return _HITL_DISPLAY.get(node_name, f"Agent 在节点 [{node_name}] 暂停，等待确认。")


def format_approval_event(node_name: str, state: dict[str, Any]) -> dict[str, Any]:
    """生成前端 SSE 审批事件。"""
    detail = ""
    if node_name == "tool_execute":
        msgs = state.get("messages", [])
        if msgs:
            last = msgs[-1]
            detail = getattr(last, "content", str(last))[:200]
    elif node_name == "hallucination_low":
        detail = state.get("hallucination_detail", "")

    return {
        "node": node_name,
        "message": get_interrupt_message(node_name),
        "detail": detail,
    }
