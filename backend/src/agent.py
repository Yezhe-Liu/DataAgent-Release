"""DataAgent 入口 —— 组合根 (Composition Root)

负责:
1. 创建 LLM 模型实例
2. 加载工具（本地 + MCP）
3. 将依赖注入到自定义 StateGraph
4. 对外暴露编译后的 graph 实例
5. 初始化 Tracer / MemoryManager 等基础设施
"""

from __future__ import annotations

import asyncio

from src.config import get_chat_settings, get_env_text
from src.llm_factory import create_chat_model, create_flash_model, create_pro_model
from src.mcp_tools import load_mcp_tools
from src.rag_engine import retrieve_knowledge
from src.tools import external_search, fig_inter, python_inter

# ---- 全局 Tracer ----
_TRACER = None


def get_tracer():
    global _TRACER
    if _TRACER is None:
        backend = get_env_text("TRACER_BACKEND", "console")
        if backend == "langfuse":
            from src.observability.langfuse import LangFuseTracer
            _TRACER = LangFuseTracer()
        elif backend == "console":
            from src.observability.console import ConsoleTracer
            _TRACER = ConsoleTracer()
        else:
            from src.observability.tracer import NoopTracer
            _TRACER = NoopTracer()
    return _TRACER

# ---- 本地数据分析工具 ----
DATA_TOOLS = [python_inter, fig_inter]

# ---- MCP 工具加载 ----
_TOOLS, _MCP_STATUS = load_mcp_tools(DATA_TOOLS)
print(f"[Agent] MCP status: {_MCP_STATUS}; total_tools={len(_TOOLS)}")

# ---- 图缓存（模型变更时重建） ----
_GRAPH = None
_GRAPH_SIGNATURE: tuple[str, str, str] | None = None  # (provider, flash_model, pro_model)


def get_agent_graph():
    """获取编译后的 Agentic RAG StateGraph（带模型变更检测缓存）。

    双模型架构:
      - flash_model: deepseek-v4-flash (thinking=disabled) → Router/Grade/Text-to-SQL/Rewrite
      - pro_model:   deepseek-v4-pro  (thinking=enabled)  → Generate/HallucinationCheck
    """
    global _GRAPH, _GRAPH_SIGNATURE

    chat_settings = get_chat_settings()
    signature = (chat_settings.provider, chat_settings.model, chat_settings.model_pro)

    if _GRAPH is not None and _GRAPH_SIGNATURE == signature:
        return _GRAPH

    flash_model = create_flash_model()
    pro_model = create_pro_model()
    print(f"[Agent] provider={chat_settings.provider} flash={chat_settings.model} pro={chat_settings.model_pro}")

    # 注入 flash 模型到 RAG 模块（HyDE / LLM Reranker 不需要深度推理）
    from src.rag_engine import set_rag_model
    set_rag_model(flash_model)

    # 注入 flash 模型到 MemoryManager（对话摘要 / 长期记忆提取）
    from src.memory.manager import MemoryManager
    if not hasattr(get_agent_graph, "_memory_manager"):
        get_agent_graph._memory_manager = MemoryManager(model=flash_model)  # type: ignore[attr-defined]

    from src.graph.builder import build_graph
    from src.hitl import get_interrupt_nodes

    # 查找所有 MCP 数据库查询工具 (query_*)
    _db_tools = [t for t in _TOOLS if getattr(t, "name", "").startswith("query_")]

    _GRAPH = build_graph(
        flash_model=flash_model,
        pro_model=pro_model,
        data_tools=_TOOLS,
        retrieve_func=retrieve_knowledge,
        search_func=external_search.invoke,
        db_tools=_db_tools,
        interrupt_before=get_interrupt_nodes(),
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
