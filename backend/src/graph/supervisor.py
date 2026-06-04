"""Supervisor 主编排器 — 全局路由 + HITL 中断 + Worker 委派

主图拓扑 (唯一 interrupt_before=["supervisor_handoff"]):
  __start__
      │
      ▼
  [router] ─────────────────────────────────────────┐
      │                                              │
      ├── intent=structured_telecom_query             │
      │     ▼                                         │
      │  [text_to_sql] ──► [supervisor_handoff]      │
      │     (生成SQL)        ⏸ HITL 全局中断点        │
      │                     (pending_sql 已就绪)       │
      │                         │                     │
      │                         ▼                     │
      │                     [sql_worker]              │
      │                                              │
      ├── intent=rag ──► [rag_worker]                │
      ├── intent=tool ─► [tool_worker]               │
      └── intent=chat ─► [chat_worker]               │
                             │                       │
                             ▼                       │
                            END                      │

关键安全约束:
  - supervisor_handoff 节点仅在 SQL 路径上 → interrupt_before 只拦截 SQL 请求
  - text_to_sql 在主图执行 → HITL 断点时 pending_sql 已写入 state
  - 子图退出时 extract_delta 裁剪消息 → 杜绝 add_messages 重叠翻倍

Checkpointer 迁移路径:
  v1 (当前): MemorySaver — 进程内存, 开发/单机部署
  v2 (规划): RedisSaver — 分布式持久化, 多实例共享

  切换到 RedisSaver 步骤:
    1. pip install langgraph-checkpoint-redis
    2. 在 builder.py 中替换:
         from langgraph.checkpoint.redis import RedisSaver
         checkpointer = RedisSaver(redis_client=redis.from_url(REDIS_URL))
    3. HITL 中断持久化已在 hitl/persistence.py 中独立实现,
       不受 checkpointer 切换影响
"""

from __future__ import annotations

from typing import Any, Callable, Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from src.graph.edges import grade_documents, check_hallucination
from src.graph.nodes import (
    create_router_node,
    create_text_to_sql_node,
)
from src.graph.state import SupervisorState


def _route_from_router(
    state: dict,
) -> Literal["text_to_sql", "rag_worker", "tool_worker", "chat_worker"]:
    """Supervisor 级路由: 根据 intent 分发到对应路径。

    SQL 路径经过 text_to_sql → supervisor_handoff (HITL)，
    其他路径直接委派 Worker。
    """
    intent = state.get("intent", "chat")
    if intent == "structured_telecom_query":
        return "text_to_sql"
    if intent == "rag":
        return "rag_worker"
    if intent == "tool":
        return "tool_worker"
    return "chat_worker"


def _supervisor_handoff_node(state: SupervisorState) -> dict[str, Any]:
    """HITL 审批网关节点。

    此节点在 text_to_sql 之后执行，此时 pending_sql 已就绪。
    interrupt_before 在此节点前暂停，前端可读取 pending_sql 展示审批卡片。

    审批通过 (POST /chat/resume) 后，继续委派到 sql_worker。
    """
    pending_sql = state.get("pending_sql", "")
    intent = state.get("intent", "")

    return {
        "active_worker": "sql",
        "loop_count": state.get("loop_count", 0),
    }


def build_supervisor_graph(
    flash_model: BaseChatModel,
) -> StateGraph:
    """构建 Supervisor 主图。

    主图包含:
      - router:         意图分类 (flash_model, thinking=disabled)
      - text_to_sql:    NL→SQL 生成 (flash_model, thinking=disabled)
      - supervisor_handoff: HITL 中断网关 (纯路由, 无 LLM 调用)

    子图 (rag_worker / sql_worker / tool_worker / chat_worker)
    在 builder.py 的 assemble_supervisor_graph() 中嵌入。

    Args:
        flash_model: 快速模型实例 (deepseek-v4-flash)
    """
    workflow = StateGraph(SupervisorState)

    # 节点 — 主图只包含 router + text_to_sql + handoff 网关
    workflow.add_node("router", create_router_node(flash_model))
    workflow.add_node("text_to_sql", create_text_to_sql_node(flash_model))
    workflow.add_node("supervisor_handoff", _supervisor_handoff_node)

    # 入口
    workflow.set_entry_point("router")

    # 条件路由: router 分发到 text_to_sql 或直接到 Worker
    workflow.add_conditional_edges("router", _route_from_router, {
        "text_to_sql": "text_to_sql",
        "rag_worker": "rag_worker",
        "tool_worker": "tool_worker",
        "chat_worker": "chat_worker",
    })

    # SQL 路径: text_to_sql → supervisor_handoff
    # supervisor_handoff → sql_worker 的边在总装时添加 (Worker 嵌入后)
    workflow.add_edge("text_to_sql", "supervisor_handoff")

    # 非 SQL 路径的 Worker → END 边在总装时添加 (Worker 嵌入后)
    # 此处仅创建主图骨架，Worker 节点尚未嵌入

    return workflow
