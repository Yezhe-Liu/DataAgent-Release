"""条件路由逻辑

每个函数根据 AgentState 中的字段决定下一步跳转。
"""

from __future__ import annotations

from typing import Literal


def route_intent(state: dict) -> Literal["rewrite", "tool_execute", "text_to_sql", "generate"]:
    """根据 router 分类结果分发到不同执行路径。"""
    intent = state.get("intent", "chat")
    if intent == "structured_telecom_query":
        return "text_to_sql"
    if intent == "rag":
        return "rewrite"
    if intent == "tool":
        return "tool_execute"
    return "generate"


def grade_documents(state: dict) -> Literal["web_search", "generate"]:
    """检查是否有相关文档。全部无关时走外网搜索。"""
    graded_docs = state.get("graded_docs", [])
    if not graded_docs:
        return "web_search"
    relevant = [d for d in graded_docs if d.get("relevance") == "relevant"]
    if not relevant:
        return "web_search"
    return "generate"


def check_hallucination(state: dict) -> Literal["rewrite", "generate"]:
    """幻觉检查不通过时回退重写查询（最多 2 轮）。"""
    score = state.get("hallucination_score", 0.0)
    loop_count = state.get("loop_count", 0)
    max_loops = 2

    if score >= 0.6:
        return "generate"  # 通过 -> 直接返回最终回答

    if loop_count >= max_loops:
        print(f"[Edges] Max loops ({max_loops}) reached, returning best effort answer")
        return "generate"

    return "rewrite"
