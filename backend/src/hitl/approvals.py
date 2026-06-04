"""HITL (Human-in-the-Loop) 审批策略

提供可配置的中断策略和 LangGraph interrupt 配置生成。
纯函数模块，不依赖项目其他模块。

v2 架构说明:
  - SQL HITL 由 builder.py 自动配置 supervisor_handoff (仅当 db_tools 存在时)
  - tool_execute 已迁移到 ToolWorker 子图内部，其 HITL 由 ToolWorker 子图自行管理
  - interrupt_before 只能引用主图层节点，子图内部节点会导致 LangGraph validate 报错
"""

from __future__ import annotations

from typing import Any

from src.config import get_env_text

# ---- 审批策略 ----

PolicyValue = str  # "always_ask" | "never" | "ask"

# v2 默认策略: 空。SQL HITL 由 supervisor_handoff 自动处理，tool_execute 已下沉到子图
DEFAULT_POLICY: dict[str, PolicyValue] = {}

# 逻辑中断点 → 实际主图层 graph 节点映射
# 注意: 只能包含主图层 (SupervisorGraph) 节点，子图内部节点不在此配置
_POLICY_TO_NODE: dict[str, str] = {
    # "tool_execute": "tool_execute",  # v2: 已下沉到 ToolWorker 子图内部
}

_HITL_DISPLAY: dict[str, str] = {}


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
    """返回需要配置 interrupt_before 的实际 graph 节点列表。"""
    policy = get_hitl_policy()
    nodes: list[str] = []
    for key, rule in policy.items():
        if rule != "never" and key in _POLICY_TO_NODE:
            nodes.append(_POLICY_TO_NODE[key])
    return nodes


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

    # rule == "ask": 根据状态动态判断 (未来可扩展)
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

    return {
        "node": node_name,
        "message": get_interrupt_message(node_name),
        "detail": detail,
    }
