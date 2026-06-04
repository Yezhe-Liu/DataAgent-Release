"""StateGraph 构建器

组装完整的 Agentic RAG 工作流，含 Text-to-SQL 电信气象查询分支。
依赖通过参数注入。
"""

from __future__ import annotations

from typing import Any, Callable

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from src.graph.edges import check_hallucination, grade_documents, route_intent
from src.graph.nodes import (
    create_execute_sql_tool_node,
    create_generate_node,
    create_grade_node,
    create_hallucination_check_node,
    create_retrieve_node,
    create_rewrite_node,
    create_router_node,
    create_text_to_sql_node,
    create_tool_execute_node,
    create_web_search_node,
)
from src.graph.state import AgentState


def build_graph(
    model: BaseChatModel,
    data_tools: list[BaseTool],
    retrieve_func: Callable[..., list[Any]],
    search_func: Callable[..., str],
    db_tools: list[Any] | None = None,
    interrupt_before: list[str] | None = None,
):
    """构建 Agentic RAG StateGraph。

    Args:
        model: 注入的 LLM 实例
        data_tools: 数据分析工具列表
        retrieve_func: 知识库检索函数
        search_func: 外网搜索函数
        db_tool: 电信气象数据库 MCP 工具 (query_telecom_weather_db)
        interrupt_before: HITL 中断节点列表
    """
    workflow = StateGraph(AgentState)

    # 节点
    workflow.add_node("router", create_router_node(model))
    workflow.add_node("rewrite", create_rewrite_node(model))
    workflow.add_node("retrieve", create_retrieve_node(retrieve_func))
    workflow.add_node("grade", create_grade_node(model))
    workflow.add_node("web_search", create_web_search_node(search_func))
    workflow.add_node("text_to_sql", create_text_to_sql_node(model))
    workflow.add_node("execute_sql_tool", create_execute_sql_tool_node(db_tools or []))
    workflow.add_node("tool_execute", create_tool_execute_node(model, data_tools))
    workflow.add_node("generate", create_generate_node(model))
    workflow.add_node("hallucination_check", create_hallucination_check_node(model))

    # 入口
    workflow.set_entry_point("router")

    # 条件边: router 分发到四个分支
    workflow.add_conditional_edges("router", route_intent, {
        "text_to_sql": "text_to_sql",
        "rewrite": "rewrite",
        "tool_execute": "tool_execute",
        "generate": "generate",
    })

    # Text-to-SQL 路径: NL→SQL → (HITL断点) → 执行SQL → RAG检索
    workflow.add_edge("text_to_sql", "execute_sql_tool")
    workflow.add_edge("execute_sql_tool", "rewrite")

    # RAG 路径
    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("retrieve", "grade")
    workflow.add_conditional_edges("grade", grade_documents, {
        "web_search": "web_search",
        "generate": "generate",
    })
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", "hallucination_check")
    workflow.add_conditional_edges("hallucination_check", check_hallucination, {
        "rewrite": "rewrite",
        "generate": END,
    })

    # Tool 路径
    workflow.add_edge("tool_execute", END)

    compile_kwargs: dict = {"checkpointer": MemorySaver()}
    merged_interrupts = list(interrupt_before or [])
    # text_to_sql 分支: execute_sql_tool 前暂停等待审批
    if db_tools and "execute_sql_tool" not in merged_interrupts:
        merged_interrupts.append("execute_sql_tool")
    if merged_interrupts:
        compile_kwargs["interrupt_before"] = merged_interrupts
    return workflow.compile(**compile_kwargs)
