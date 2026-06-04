"""DataAgent 入口 —— 组合根 (Composition Root)

负责:
1. 创建 LLM 模型实例
2. 加载工具（本地 + MCP）
3. 将依赖注入到自定义 StateGraph
4. 对外暴露编译后的 graph 实例
"""

from __future__ import annotations

import asyncio

from src.config import get_chat_settings
from src.llm_factory import create_chat_model
from src.mcp_tools import load_mcp_tools
from src.rag_engine import retrieve_knowledge
from src.tools import external_search, fig_inter, python_inter

# ---- 本地数据分析工具 ----
DATA_TOOLS = [python_inter, fig_inter]

# ---- MCP 工具加载 ----
_TOOLS, _MCP_STATUS = load_mcp_tools(DATA_TOOLS)
print(f"[Agent] MCP status: {_MCP_STATUS}; total_tools={len(_TOOLS)}")

# ---- 图缓存（模型变更时重建） ----
_GRAPH = None
_GRAPH_SIGNATURE: tuple[str, str] | None = None


def get_agent_graph():
    """获取编译后的 Agentic RAG StateGraph（带模型变更检测缓存）。"""
    global _GRAPH, _GRAPH_SIGNATURE

    chat_settings = get_chat_settings()
    signature = (chat_settings.provider, chat_settings.model)

    if _GRAPH is not None and _GRAPH_SIGNATURE == signature:
        return _GRAPH

    model = create_chat_model()
    print(f"[Agent] provider={chat_settings.provider} model={chat_settings.model}")

    from src.graph.builder import build_graph

    _GRAPH = build_graph(
        model=model,
        data_tools=_TOOLS,
        retrieve_func=retrieve_knowledge,
        search_func=external_search.invoke,
    )
    _GRAPH_SIGNATURE = signature
    return _GRAPH


# 兼容旧代码：模块级 graph 懒加载
graph = None


def _init_graph():
    global graph
    if graph is None:
        graph = get_agent_graph()
    return graph


# ---- Tool 运行时元数据（供 server.py 状态展示使用） ----

_LOCAL_TOOL_NAMES = {tool.name for tool in DATA_TOOLS}
_MCP_TOOL_NAMES = {tool.name for tool in _TOOLS} - _LOCAL_TOOL_NAMES

TOOL_DISPLAY_NAMES = {
    "python_inter": "本地 Python 分析",
    "fig_inter": "本地绘图",
}


def get_tool_runtime_origin(tool_name: str) -> str:
    if tool_name in _MCP_TOOL_NAMES:
        return "mcp"
    if tool_name in _LOCAL_TOOL_NAMES:
        return "local"
    return "unknown"


def get_tool_display_name(tool_name: str) -> str:
    return TOOL_DISPLAY_NAMES.get(tool_name, tool_name)
