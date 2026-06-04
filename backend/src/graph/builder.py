"""StateGraph 总装器 — Supervisor + Worker 主从多智能体网络

图拓扑:
  SupervisorGraph (主图, interrupt_before=["supervisor_handoff"])
    ├── router        → 意图分类 (flash_model)
    ├── text_to_sql  → NL→SQL 生成 (flash_model)
    ├── supervisor_handoff → HITL 网关 (仅 SQL 路径)
    ├── rag_worker   → 嵌入: RAGWorker 子图 (rewrite→retrieve→grade→generate→hal_check)
    ├── sql_worker   → 嵌入: SQLWorker 子图 (execute_sql→generate)
    ├── tool_worker  → 嵌入: ToolWorker 子图 (tool_execute)
    └── chat_worker  → 嵌入: ChatWorker 子图 (generate)

双模型注入:
  flash_model (thinking=disabled) → router, text_to_sql, rewrite, grade, tool_execute
  pro_model  (thinking=enabled)  → generate, hallucination_check
"""

from __future__ import annotations

from typing import Any, Callable

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from src.graph.supervisor import build_supervisor_graph
from src.graph.workers.rag_worker import build_rag_worker
from src.graph.workers.sql_worker import build_sql_worker
from src.graph.workers.tool_worker import build_tool_worker
from src.graph.workers.chat_worker import build_chat_worker
from src.graph.state import SupervisorState

# ---------------------------------------------------------------------------
# 向后兼容入口
# ---------------------------------------------------------------------------


def build_graph(
    flash_model: BaseChatModel,
    pro_model: BaseChatModel,
    data_tools: list[BaseTool],
    retrieve_func: Callable[..., list[Any]],
    search_func: Callable[..., str],
    db_tools: list[Any] | None = None,
    interrupt_before: list[str] | None = None,
):
    """向后兼容: 委托到 assemble_supervisor_graph()。"""
    return assemble_supervisor_graph(
        flash_model=flash_model,
        pro_model=pro_model,
        data_tools=data_tools,
        retrieve_func=retrieve_func,
        search_func=search_func,
        db_tools=db_tools,
        interrupt_before=interrupt_before,
    )


# ---------------------------------------------------------------------------
# 总装函数
# ---------------------------------------------------------------------------


def assemble_supervisor_graph(
    flash_model: BaseChatModel,
    pro_model: BaseChatModel,
    data_tools: list[BaseTool],
    retrieve_func: Callable[..., list[Any]],
    search_func: Callable[..., str],
    db_tools: list[Any] | None = None,
    interrupt_before: list[str] | None = None,
):
    """构建完整的 Supervisor + Worker 多智能体图网络。

    1. 构建 Supervisor 主图 (router + text_to_sql + supervisor_handoff)
    2. 编译 4 个 Worker 子图并嵌入为主图节点
    3. 配置 interrupt_before: supervisor_handoff 为全局 HITL 中断点（仅 SQL 路径触发）
    4. 注入 MemorySaver checkpointer

    Args:
        flash_model:   快速模型 (deepseek-v4-flash, thinking=disabled)
        pro_model:     强推理模型 (deepseek-v4-pro, thinking=enabled)
        data_tools:    数据分析工具列表
        retrieve_func: 知识库检索函数
        search_func:   外网搜索函数
        db_tools:      MCP 数据库查询工具列表
        interrupt_before: 额外 HITL 中断节点 (来自 HITL 策略配置)

    Returns:
        编译后的 StateGraph，可直接用于 astream_events()
    """
    # 1. 构建 Supervisor 主图骨架
    print("[Builder] Assembling Supervisor main graph...")
    supervisor = build_supervisor_graph(flash_model)

    # 2. 编译 Worker 子图并嵌入
    print("[Builder] Compiling RAGWorker subgraph...")
    rag_worker = build_rag_worker(flash_model, pro_model, retrieve_func, search_func)
    supervisor.add_node("rag_worker", rag_worker)

    print("[Builder] Compiling SQLWorker subgraph...")
    sql_worker = build_sql_worker(pro_model, db_tools or [])
    supervisor.add_node("sql_worker", sql_worker)

    print("[Builder] Compiling ToolWorker subgraph...")
    tool_worker = build_tool_worker(flash_model, data_tools)
    supervisor.add_node("tool_worker", tool_worker)

    print("[Builder] Compiling ChatWorker subgraph...")
    chat_worker = build_chat_worker(pro_model)
    supervisor.add_node("chat_worker", chat_worker)

    # 2.5 补齐 Worker 嵌入后的边 (这些边在 supervisor.py 中未定义，因为 Worker 节点尚不存在)
    supervisor.add_edge("supervisor_handoff", "sql_worker")
    supervisor.add_edge("rag_worker", END)
    supervisor.add_edge("sql_worker", END)
    supervisor.add_edge("tool_worker", END)
    supervisor.add_edge("chat_worker", END)

    # 3. 配置 HITL 中断点
    merged_interrupts = list(interrupt_before or [])

    # supervisor_handoff 为全局唯一物理中断点（仅在 SQL 路径上）
    if db_tools and "supervisor_handoff" not in merged_interrupts:
        merged_interrupts.append("supervisor_handoff")

    # 防御性过滤: 只保留主图层存在的节点，移除子图内部节点
    # 子图内部节点 (如 tool_execute) 由各 Worker 子图自行管理 interrupt_before
    supervisor_node_names = set(supervisor.nodes.keys()) if hasattr(supervisor, "nodes") else set()
    if supervisor_node_names:
        filtered = [n for n in merged_interrupts if n in supervisor_node_names]
        removed = set(merged_interrupts) - set(filtered)
        if removed:
            print(f"[Builder] Filtered subgraph-internal interrupts: {removed}")
        merged_interrupts = filtered

    # Checkpointer: v1 MemorySaver (进程内存), v2 切换为 RedisSaver (分布式)
    compile_kwargs: dict = {"checkpointer": MemorySaver()}
    if merged_interrupts:
        compile_kwargs["interrupt_before"] = merged_interrupts
        print(f"[Builder] HITL interrupt_before: {merged_interrupts}")

    print("[Builder] Supervisor graph assembled. Compiling...")
    compiled = supervisor.compile(**compile_kwargs)
    print("[Builder] Graph compilation complete.")

    return compiled
