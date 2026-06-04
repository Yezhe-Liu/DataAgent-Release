"""Console Tracer — 结构化日志输出

用于开发调试，替代散落的 print() 调用。
"""

from __future__ import annotations

import json
import time
from typing import Any

from src.observability.tracer import Tracer, TraceSpan


class ConsoleTracer(Tracer):
    """结构化控制台追踪器。"""

    def __init__(self):
        self._spans: dict[str, TraceSpan] = {}

    def trace_node_start(self, node_name: str, input_summary: str = "") -> str:
        span_id = f"{node_name}-{time.time_ns()}"
        span = TraceSpan(span_id=span_id, name=node_name)
        self._spans[span_id] = span
        self._log("NODE_START", {"node": node_name, "input": input_summary[:120]})
        return span_id

    def trace_node_end(self, span_id: str, output_summary: str = "") -> None:
        span = self._spans.pop(span_id, None)
        latency = (time.time() - span.started_at) * 1000 if span else 0
        self._log("NODE_END", {
            "node": span.name if span else "unknown",
            "latency_ms": round(latency, 1),
            "output": output_summary[:120],
        })

    def trace_llm(self, model: str, latency_ms: float, token_count: int = 0) -> None:
        self._log("LLM_CALL", {
            "model": model,
            "latency_ms": round(latency_ms, 1),
            "tokens": token_count,
        })

    def trace_retrieval(self, query: str, hit_count: int, latency_ms: float) -> None:
        self._log("RETRIEVAL", {
            "query": query[:80],
            "hits": hit_count,
            "latency_ms": round(latency_ms, 1),
        })

    def flush(self) -> None:
        self._spans.clear()

    @staticmethod
    def _log(event: str, data: dict[str, Any]) -> None:
        print(f"[Trace] {event} {json.dumps(data, ensure_ascii=False)}")
