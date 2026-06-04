"""ToolWorker 子图 — 数据分析/绘图工具调用 (mini ReAct)

子图内部 (单节点):
  tool_execute → END
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from src.graph.nodes import create_tool_execute_node
from src.graph.state import ToolWorkerState


def build_tool_worker(
    flash_model: BaseChatModel,
    data_tools: list[BaseTool],
) -> StateGraph:
    """构建 ToolWorker 子图。

    Args:
        flash_model: 快速模型 (thinking=disabled), 用于工具调用 mini ReAct
        data_tools:  数据分析/绘图工具列表
    """
    workflow = StateGraph(ToolWorkerState)

    workflow.add_node("tool_execute", create_tool_execute_node(flash_model, data_tools))

    workflow.set_entry_point("tool_execute")
    workflow.add_edge("tool_execute", END)

    return workflow.compile(checkpointer=MemorySaver())
