"""RAGWorker 子图 — 企业知识库检索+生成+幻觉核查

子图内部拓扑:
  rewrite → retrieve → grade → [conditional: relevant?]
    → relevant:  generate
    → not:       web_search → generate
  → hallucination_check → [conditional: pass?]
    → pass:  END
    → fail:  rewrite (回退, ≤2 轮)
"""

from __future__ import annotations

from typing import Any, Callable

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from langchain_core.language_models import BaseChatModel

from src.graph.edges import check_hallucination, grade_documents
from src.graph.nodes import (
    create_generate_node,
    create_grade_node,
    create_hallucination_check_node,
    create_retrieve_node,
    create_rewrite_node,
    create_web_search_node,
)
from src.graph.state import RAGWorkerState

RetrieverFunc = Callable[..., list[Any]]
SearchFunc = Callable[..., str]


def build_rag_worker(
    flash_model: BaseChatModel,
    pro_model: BaseChatModel,
    retrieve_func: RetrieverFunc,
    search_func: SearchFunc,
) -> StateGraph:
    """构建 RAGWorker 子图。

    模型分配:
      flash_model → rewrite / grade (需要稳定结构化输出)
      pro_model  → generate / hallucination_check (需要深度推理)

    Args:
        flash_model: 快速模型 (thinking=disabled), 用于结构化输出节点
        pro_model:   强推理模型 (thinking=enabled), 用于深度推理节点
        retrieve_func: 知识库多路检索函数
        search_func:  外网搜索回退函数
    """
    workflow = StateGraph(RAGWorkerState)

    # flash_model 节点
    workflow.add_node("rewrite", create_rewrite_node(flash_model))
    workflow.add_node("grade", create_grade_node(flash_model))

    # 无 LLM 节点
    workflow.add_node("retrieve", create_retrieve_node(retrieve_func))
    workflow.add_node("web_search", create_web_search_node(search_func))

    # pro_model 节点
    workflow.add_node("generate", create_generate_node(pro_model))
    workflow.add_node("hallucination_check", create_hallucination_check_node(pro_model))

    # 入口: rewrite
    workflow.set_entry_point("rewrite")

    # 检索链: rewrite → retrieve → grade
    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("retrieve", "grade")

    # 文档评分分支
    workflow.add_conditional_edges("grade", grade_documents, {
        "web_search": "web_search",
        "generate": "generate",
    })
    workflow.add_edge("web_search", "generate")

    # 幻觉核查回退循环
    workflow.add_edge("generate", "hallucination_check")
    workflow.add_conditional_edges("hallucination_check", check_hallucination, {
        "rewrite": "rewrite",
        "generate": END,
    })

    return workflow.compile(checkpointer=MemorySaver())
