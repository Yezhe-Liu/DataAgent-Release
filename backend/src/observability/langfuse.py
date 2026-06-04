"""LangFuse Tracer — 对接开源可观测性平台

需要自部署 LangFuse (https://langfuse.com) 或使用 Cloud 版。
未配置时自动降级为 ConsoleTracer。
"""

from __future__ import annotations

from src.observability.console import ConsoleTracer
from src.observability.tracer import Tracer


class LangFuseTracer(Tracer):
    """LangFuse 追踪器。

    环境变量:
      LANGFUSE_PUBLIC_KEY  — 公钥
      LANGFUSE_SECRET_KEY  — 私钥
      LANGFUSE_HOST        — 自部署地址 (可选)

    未配置任一必填项时, 自动降级为 ConsoleTracer。
    """

    def __init__(self):
        import os

        pk = os.getenv("LANGFUSE_PUBLIC_KEY", "")
        sk = os.getenv("LANGFUSE_SECRET_KEY", "")

        if not pk or not sk:
            self._fallback: Tracer = ConsoleTracer()
            print("[Observability] LangFuse not configured, using console tracer")
            return

        try:
            from langfuse import Langfuse
            self._client = Langfuse(
                public_key=pk,
                secret_key=sk,
                host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
            )
            self._fallback = None
            self._trace = self._client.trace(name="DataAgent")
            print("[Observability] LangFuse connected")
        except Exception as err:
            self._fallback = ConsoleTracer()
            print(f"[Observability] LangFuse init failed: {err}, using console tracer")

    def trace_node_start(self, node_name: str, input_summary: str = "") -> str:
        if self._fallback:
            return self._fallback.trace_node_start(node_name, input_summary)

        span = self._trace.span(name=node_name, input={"summary": input_summary})
        return span.id or f"{node_name}-{id(span)}"

    def trace_node_end(self, span_id: str, output_summary: str = "") -> None:
        if self._fallback:
            self._fallback.trace_node_end(span_id, output_summary)
            return
        # LangFuse span auto-closes; record output via update
        self._trace.update(output={"summary": output_summary})

    def trace_llm(self, model: str, latency_ms: float, token_count: int = 0) -> None:
        if self._fallback:
            self._fallback.trace_llm(model, latency_ms, token_count)
            return
        self._trace.span(
            name="llm_call",
            metadata={"model": model, "latency_ms": latency_ms, "tokens": token_count},
        )

    def trace_retrieval(self, query: str, hit_count: int, latency_ms: float) -> None:
        if self._fallback:
            self._fallback.trace_retrieval(query, hit_count, latency_ms)
            return
        self._trace.span(
            name="retrieval",
            metadata={"query": query[:100], "hits": hit_count, "latency_ms": latency_ms},
        )

    def flush(self) -> None:
        if self._fallback:
            self._fallback.flush()
            return
        try:
            self._client.flush()
        except Exception:
            pass
