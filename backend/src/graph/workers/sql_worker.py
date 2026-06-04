"""SQLWorker 子图 — 执行已审批 SQL + 生成结果

子图内部拓扑 (极简两节点):
  execute_sql → generate → END

关键安全约束:
  - text_to_sql 已从子图移除，留在 Supervisor 主图
  - Supervisor 通过 worker_input["pending_sql"] 注入已审批的 SQL
  - 子图只负责安全执行和结果生成，不涉及 SQL 生成
"""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from langchain_core.language_models import BaseChatModel

from src.graph.nodes import (
    create_execute_sql_tool_node,
    create_generate_node,
)
from src.graph.state import SQLWorkerState


def build_sql_worker(
    pro_model: BaseChatModel,
    db_tools: list[Any],
) -> StateGraph:
    """构建 SQLWorker 子图 — 仅执行已审批 SQL。

    子图接收 Supervisor 传入的 pending_sql（通过 worker_input），
    执行后调用 generate 节点基于 SQL 结果生成自然语言回答。

    Args:
        pro_model: 强推理模型 (thinking=enabled), 用于结果生成
        db_tools:  MCP 数据库查询工具列表 (query_telecom_weather_db, query_enterprise_policy_db)
    """
    workflow = StateGraph(SQLWorkerState)

    # execute_sql_tool: 无 LLM 调用，纯工具执行
    workflow.add_node("execute_sql", create_execute_sql_tool_node(db_tools or []))

    # generate: pro_model 生成自然语言回答
    workflow.add_node("generate", create_generate_node(pro_model))

    workflow.set_entry_point("execute_sql")
    workflow.add_edge("execute_sql", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile(checkpointer=MemorySaver())
