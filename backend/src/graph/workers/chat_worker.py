"""ChatWorker 子图 — 简单闲聊对话

子图内部 (单节点):
  generate → END
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from langchain_core.language_models import BaseChatModel

from src.graph.nodes import create_generate_node
from src.graph.state import ChatWorkerState


def build_chat_worker(
    pro_model: BaseChatModel,
) -> StateGraph:
    """构建 ChatWorker 子图。

    Args:
        pro_model: 强推理模型 (thinking=enabled), 用于自然对话生成
    """
    workflow = StateGraph(ChatWorkerState)

    workflow.add_node("generate", create_generate_node(pro_model))

    workflow.set_entry_point("generate")
    workflow.add_edge("generate", END)

    return workflow.compile(checkpointer=MemorySaver())
