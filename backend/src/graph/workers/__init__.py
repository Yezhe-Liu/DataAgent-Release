"""Worker 子图模块

Worker 注册表 + handoff 路由映射。

Worker 拓扑:
  - RAGWorker:    rewrite → retrieve → grade → web_search → generate → hallucination_check
  - SQLWorker:    execute_sql → generate  (仅执行已审批 SQL)
  - ToolWorker:   tool_execute (mini ReAct)
  - ChatWorker:   generate (直接对话)
"""

from __future__ import annotations

from src.graph.workers.base import (
    WorkerMarker,
    build_worker_input,
    extract_delta,
)

__all__ = [
    "WorkerMarker",
    "build_worker_input",
    "extract_delta",
]

# Worker 名称 → handoff 路由目标的映射 (在 supervisor.py 中消费)
WORKER_ROUTE_MAP: dict[str, str] = {
    "rag": "rag_worker",
    "structured_telecom_query": "sql_worker",
    "tool": "tool_worker",
    "chat": "chat_worker",
}
