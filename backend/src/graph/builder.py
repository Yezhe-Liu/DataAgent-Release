"""StateGraph 构建器

组装 Router -> Rewrite -> Retrieve -> Grade -> Generate -> HallucinationCheck 的
完整 Agentic RAG 工作流。

依赖通过参数注入，不与 src.tools / src.rag_engine 直接耦合。
"""

from __future__ import annotations

from typing import Any, Callable

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from src.graph.edges import check_hallucination, grade_documents, route_intent
from src.graph.nodes import (
    create_generate_node,
    create_grade_node,
    create_hallucination_check_node,
    create_retrieve_node,
    create_rewrite_node,
    create_router_node,
    create_tool_execute_node,
    create_web_search_node,
)
from src.graph.state import AgentState


def build_graph(
    model: BaseChatModel,
    data_tools: list[BaseTool],
    retrieve_func: Callable[..., list[Any]],
    search_func: Callable[..., str],
    interrupt_before: list[str] | None = None,
):
    """构建 Agentic RAG StateGraph。

    Args:
        model: 注入的 LLM 实例
        data_tools: 数据分析工具列表 [python_inter, fig_inter]
        retrieve_func: 知识库检索函数
        search_func: 外网搜索函数
        interrupt_before: 需要 HITL 中断的节点列表 (如 ["tool_execute"])
    """
    workflow = StateGraph(AgentState)

    # 通过工厂注入依赖，创建各节点
    workflow.add_node("router", create_router_node(model))
    workflow.add_node("rewrite", create_rewrite_node(model))
    workflow.add_node("retrieve", create_retrieve_node(retrieve_func))
    workflow.add_node("grade", create_grade_node(model))
    workflow.add_node("web_search", create_web_search_node(search_func))
    workflow.add_node("tool_execute", create_tool_execute_node(model, data_tools))
    workflow.add_node("generate", create_generate_node(model))
    workflow.add_node("hallucination_check", create_hallucination_check_node(model))

    # 入口
    workflow.set_entry_point("router")

    # 条件边: router 分发到三个分支
    workflow.add_conditional_edges("router", route_intent, {
        "rewrite": "rewrite",
        "tool_execute": "tool_execute",
        "generate": "generate",
    })

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
        "generate": END,  # 通过 -> 直接结束
    })

    # Tool 路径
    workflow.add_edge("tool_execute", END)

    # 编译（MemorySaver 可在生产环境替换为 SqliteSaver）
    compile_kwargs: dict = {"checkpointer": MemorySaver()}
    if interrupt_before:
        compile_kwargs["interrupt_before"] = interrupt_before
    return workflow.compile(**compile_kwargs)
